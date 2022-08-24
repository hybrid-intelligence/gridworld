import sys
import json
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import wandb
from nltk.translate.bleu_score import sentence_bleu


class CommandsDataset(Dataset):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
    def __len__(self):
        return len(self.inputs)
    
    
def prepare_inputs(batch, tokenizer, task_prefix="implement given instructions: "):
    
    input_sequences = [x[0] for x in batch]
    output_sequences = [x[1] for x in batch]
    
    encoding = tokenizer(
        [task_prefix + sequence for sequence in input_sequences],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    
    target_encoding = tokenizer(
        output_sequences, padding="longest", max_length=max_target_length, truncation=True
    )
    labels = target_encoding.input_ids

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    labels = torch.tensor(labels)
    labels[labels == tokenizer.pad_token_id] = -100

    return input_ids, attention_mask, labels

def prepare_data(data_path, history_len=None):
    a = len('<Architect> ')
    b = len('<Builder> ')
    data = json.load(open(data_path))

    input_seqs, output_seqs = [], []
    for key, event in tqdm(data.items()):
        cur_line = ''
        architect_lines, builder_lines = [], []

        for line in event.split('\n'):
            if line.startswith('<Architect>'):
                architect_lines.append(line[a:])
                if cur_line:
                    builder_lines.append(cur_line[:-1])
                    cur_line = ''
            else:
                #line = line.replace('cube', 'block')
                cur_line += line[b:] + '. '

        if len(architect_lines) != len(builder_lines):
            architect_lines = architect_lines[:-1]

        assert len(architect_lines) == len(builder_lines)

        for i in range(len(architect_lines)):
            context = ''
            if history_len:
                range_start = max(i - history_len, 0)
                range_end = min(range_start + history_len, i)
            else:
                range_start, range_end = 0, i
                
            for j in range(range_start, range_end):
                context += '<Architect> ' + architect_lines[j] + ' <Builder> ' + builder_lines[j] + ' <sep1> '

            input_seqs.append(context + '<Architect> ' + architect_lines[i])
            output_seqs.append(builder_lines[i])

    return input_seqs, output_seqs

tokenizer = T5Tokenizer.from_pretrained("t5-large")
max_source_length = 512
max_target_length = 128
special_tokens_dict = {'additional_special_tokens': ['<Architect>', '<Builder>', '<sep1>']}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
task_prefix = "implement given instructions: "

history_len = None
if len(sys.argv) > 1:
    history_len = int(sys.argv[1])

wandb.init(project='txt2act', entity='antonvoronov', name=f'autoregressive_history_{history_len}')

train_inputs, train_outputs = prepare_data('train_data_augmented_part1.json', history_len=history_len)
train_dataset = CommandsDataset(train_inputs, train_outputs)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, 
                          collate_fn=lambda x : prepare_inputs(x, tokenizer))

val_input_seqs, val_output_seqs = prepare_data('val_data_part1.json', history_len=history_len)
val_dataset = CommandsDataset(val_input_seqs, val_output_seqs)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, drop_last=True, 
                          collate_fn=lambda x : prepare_inputs(x, tokenizer))

model = T5ForConditionalGeneration.from_pretrained("t5-large")
model.resize_token_embeddings(len(tokenizer))
model.cuda()

optimizer = AdamW(model.parameters(), lr=1e-4)

model.train()

n_epochs = 1
min_avg_loss = 1e9

for epoch in range(n_epochs):
    for i, (ids, mask, labels) in enumerate(train_loader):
        # forward pass
        loss = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda()).loss
        #losses.append(loss.item())
        wandb.log({"train_loss" : loss.item()}, step=i)
        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 200 == 199:
            model.eval()
            val_losses = []
            for ids, mask, labels in val_loader:
                with torch.no_grad():
                    loss = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda()).loss
                    wandb.log({"val_loss" : loss.item()})
                    val_losses.append(loss.item())
            
            if sum(val_losses)/len(val_losses) < min_avg_loss:
                min_avg_loss = sum(val_losses)/len(val_losses)
                torch.save(model.state_dict(), f't5-autoregressive-history-{history_len}-best.pt')
                    
            test_examples = np.random.choice(len(val_dataset), 3)
            
            text_table = wandb.Table(columns=["inputs", "predictions", "ground_truth", "BLEU"])
            for idx in test_examples:
                with torch.no_grad():
                    inputs = f"{task_prefix}{val_input_seqs[idx]}"
                    input_ids = tokenizer(f"{task_prefix}{val_input_seqs[idx]}", return_tensors="pt").input_ids
                    
                    outputs = model.generate(input_ids.cuda(), min_length=2, max_length=max_target_length)
                    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    bleu = sentence_bleu([tokenizer.tokenize(val_output_seqs[idx])], tokenizer.tokenize(outputs))
                    text_table.add_data(inputs, outputs, val_output_seqs[idx], bleu)
                    
            wandb.log({"validation_samples" : text_table})
            model.train()
    
torch.save(model.state_dict(), f't5-autoregressive-history-{history_len}-last.pt')
