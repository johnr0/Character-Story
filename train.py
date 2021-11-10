from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AdamW, AutoConfig
from prompt_embedding import PROMPTEmbedding
import torch
import argparse
import deepspeed

parser = argparse.ArgumentParser()


## Required parameters
parser.add_argument("--train_data_file", default=None, type=str, required=True,
                    help="The input training data file (a text file).")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--eval_data_file", default=None, type=str,
                    help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

parser.add_argument("--model_name", default="t5-small", type=str,
                    help="The model architecture to be fine-tuned.")
# parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
#                     help="The model checkpoint for weights initialization.")

# parser.add_argument("--config_name", default="", type=str,
#                     help="Optional pretrained config name or path if not the same as model_name_or_path")
# parser.add_argument("--tokenizer_name", default="", type=str,
#                     help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
# parser.add_argument("--cache_dir", default="", type=str,
#                     help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
# parser.add_argument("--block_size", default=-1, type=int,
#                     help="Optional input sequence length after tokenization."
#                             "The training dataset will be truncated in block of this size for training."
#                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
# parser.add_argument("--do_train", action='store_true',
#                     help="Whether to run training.")
# parser.add_argument("--do_eval", action='store_true',
#                     help="Whether to run eval on the dev set.")
# parser.add_argument("--evaluate_during_training", action='store_true',
#                     help="Run evaluation during training at each logging step.")
# parser.add_argument("--do_lower_case", action='store_true',
#                     help="Set this flag if you are using an uncased model.")

# Below applied in the code
parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--num_train_epochs", default=1.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--save_steps', type=int, default=50,
                    help="Save checkpoint every X updates steps.")


# Below not yet applied in the code
parser.add_argument('--logging_steps', type=int, default=50,
                    help="Log every X updates steps.")



parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")


parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")


parser.add_argument('--save_total_limit', type=int, default=None,
                    help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
parser.add_argument("--eval_all_checkpoints", action='store_true',
                    help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
parser.add_argument("--no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
parser.add_argument('--overwrite_output_dir', action='store_true',
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")


parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

model_name = args.model_name

####################### Load model and tokenizer #######################

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

_input1 = tokenizer.encode('Start Story: Characters:')
_input2 = tokenizer.encode('Continued Story:')

_input1 = torch.LongTensor(_input1).unsqueeze(0).to(model.device)
_input2 = torch.LongTensor(_input2).unsqueeze(0).to(model.device)

print(_input1.size())
print(_input2.size())

n_tokens1=(4,3)
prompt_emb1 = PROMPTEmbedding(model.get_input_embeddings(), n_tokens=n_tokens1, prompt_token_id=len(tokenizer), initialize_from_vocab=True, initialize_tokens=_input1[0])

n_tokens2=(5,)
prompt_emb2 = PROMPTEmbedding(model.get_input_embeddings(), n_tokens=n_tokens2, prompt_token_id=len(tokenizer), initialize_from_vocab=True, initialize_tokens=_input2[0])

model.encoder.set_input_embeddings(prompt_emb1)
model.decoder.set_input_embeddings(prompt_emb2)

for param in model.parameters():
    param.requires_grad = False
    if param.size(0)==sum(n_tokens1):
      param.requires_grad = True
      print('token1')
    if param.size(0)==sum(n_tokens2):
      param.requires_grad = True
      print('token2')


#################### Setting optimizer #####################
optimizer = AdamW(params = filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, eps=args.adam_epsilon)


if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

# model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
#                                                      model=model,
#                                                      model_parameters=params)


####################### Load data #######################
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset('csv', data_files={"train": "training_character_train.csv", "eval": "training_character_eval.csv"})
checkpoint = model_name
tokenizer = AutoTokenizer.from_pretrained(checkpoint, additional_special_tokens=['[Prompt]'], extra_ids=0)
# tokenizer.vocab_size = tokenizer.vocab_size+1

def tokenize_function(example):
    return tokenizer(example["input"], truncation=True, padding='max_length')

def tokenize_function2(example):
    return tokenizer(example["output"], truncation=True, padding='max_length')

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

tokenized_datasets2 = raw_datasets.map(tokenize_function2, batched=True)

# print('t len:', len(tokenized_datasets['train']['input_ids'][0]))
# print('e len:', len(tokenized_datasets['eval']['input_ids'][0]))

# print('t len:', len(tokenized_datasets['train']['input_ids'][1]))
# print('e len:', len(tokenized_datasets['eval']['input_ids'][1]))

# print('t len:', len(tokenized_datasets2['train']['input_ids'][0]))
# print('e len:', len(tokenized_datasets2['eval']['input_ids'][0]))


for t in ['train', 'eval']:
  tokenized_datasets[t] = tokenized_datasets[t].add_column('labels', tokenized_datasets2[t]['input_ids'])
#   tokenized_datasets[t] = tokenized_datasets[t].add_column('decoder_input_ids', tokenized_datasets2[t]['input_ids'])
  tokenized_datasets[t] = tokenized_datasets[t].add_column('decoder_attention_mask', tokenized_datasets2[t]['attention_mask'])

tokenized_datasets = tokenized_datasets.remove_columns(['Unnamed: 0', 'input', 'output'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets.set_format("torch")

from torch.utils.data import DataLoader
from transformers import get_scheduler

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=args.per_gpu_train_batch_size, collate_fn=data_collator
)

num_epochs = args.num_train_epochs
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(num_training_steps)/20,
    num_training_steps=num_training_steps
)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir='./', 
    per_device_train_batch_size=args.per_gpu_train_batch_size,
    # train_micro_batch_size=args.per_gpu_train_batch_size,
    per_device_eval_batch_size=args.per_gpu_eval_batch_size, 
    # eval_micro_batch_size=args.per_gpu_eval_batch_size,
    evaluation_strategy='steps',
    eval_steps = args.save_steps,
    save_strategy='steps', 
    save_steps = args.save_steps, 
    max_grad_norm=1.0, 
    # deepspeed = args.deepspeed_config,
    num_train_epochs=args.num_train_epochs,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['eval'],
    data_collator = data_collator, 
    optimizers= (optimizer, lr_scheduler)
)
trainer.train()
eval_result = trainer.evaluate()
print(eval_result)
torch.save(model.encoder.embed_tokens, 'encoder_tokens.pt')
torch.save(model.decoder.embed_tokens, 'decoder_tokens.pt')

# eval_dataloader = DataLoader(
#     tokenized_datasets["eval"], shuffle=True, batch_size=args.per_gpu_eval_batch_size, collate_fn=data_collator
# )

# #################### Training ########################
# from tqdm.auto import tqdm

# 






# num_epochs = args.num_train_epochs
# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps
# )
# print(num_training_steps)

# progress_bar = tqdm(range(int(num_training_steps)))
# training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
# trainer = Trainer(...)
# trainer.train()

# for epoch in range(int(num_epochs)):
#     model.train()
#     for step, batch in enumerate(train_dataloader):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()
        
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)
#         if (step+1)%args.save_steps==0:
#             print('step', step, 'loss:', loss.item())
#             torch.save(model.encoder.embed_tokens, 'encoder_tokens_{step}.pt')
#             torch.save(model.decoder.embed_tokens, 'decoder_tokens_{step}.pt')
    
#     model.eval()
#     sum_loss = 0
#     steps = 0
#     for step, batch in enumerate(eval_dataloader):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)
#         loss = outputs.loss
#         sum_loss += loss.item()
#         steps += 1
#     torch.save(model.encoder.embed_tokens, 'encoder_tokens.pt')
#     torch.save(model.decoder.embed_tokens, 'decoder_tokens.pt')
#     print('epoch', epoch, 'loss:', sum_loss/steps)