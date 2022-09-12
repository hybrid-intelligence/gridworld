import os
import torch
import numpy as np
from argparse import ArgumentParser
from transformers import T5Tokenizer, T5ForConditionalGeneration

import sys
sys.path.append("agents/mhb_baseline/")
from nlp_model.utils import parse_logs, update_state_from_action, logging


def load_model(args):
    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained("./agents/mhb_baseline/nlp_model/tokenizer/", 
                                            max_source_length=args.max_source_length, 
                                            max_target_length=args.max_target_length,
                                            model_max_length=args.max_source_length)
    #tokenizer.save_vocabulary(".")

    special_tokens_dict = {'additional_special_tokens': ['<Architect>', '<Builder>', '<sep1>']}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    # model
    model = T5ForConditionalGeneration.from_pretrained("./agents/mhb_baseline/nlp_model/model/")
   # model.save_pretrained("model")
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()
    
    return model, tokenizer

def generate_actions(args, model, tokenizer, command, history, log_file=None):
    context = parse_logs(history) + '<Architect> ' + command.strip()
    
    input_ids = tokenizer(f"{args.task_prefix} {context}", return_tensors="pt").input_ids

    outputs = model.generate(input_ids.cuda(), min_length=2, max_length=args.max_target_length)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if args.verbose > 0:
        logging(log_file, f'model input: {context}')
        logging(log_file, f'model output: {prediction}')
    
    return prediction

def update_state(args, last_state, actions, log_file=None):
    if args.init_block:
        init_block = tuple(args.init_block)
    else:
        init_block = (5, 0, 5)
        
    n_actions, valid_actions = 0, 0
    
    actions = actions.split('.')
    for action in actions:
        if action:
            n_actions += 1
            last_state, ok = update_state_from_action(args, last_state, action, init_block, log_file)
            if ok:
                valid_actions += 1
                
    return last_state, {'n_actions' : n_actions, 'n_valid' : valid_actions}
    
    
def predict_voxel(command, model,tokenizer, history, voxel, args):
	
	actions = generate_actions(args, model, tokenizer, command, history, None)
	voxel, new_stats = update_state(args, voxel, actions, None)
	history.append((command, actions))
	right_voxel = np.transpose(voxel, (1, 0, 2))
	
	return history, right_voxel, voxel
	
        
def init_models(args):
     model, tokenizer = load_model(args)
     history = []
     voxel = np.zeros((11, 9, 11))
     stats = {'n_actions' : 0, 'n_valid' : 0}
     return model, tokenizer, history, stats, voxel
     
     
	
def main(args):
    if args.logs_path:
        log_file = open(args.logs_path, 'a')
    else:
        log_file = None
        
    model, tokenizer = load_model(args)

    history = []
    voxel = np.zeros((11, 9, 11))
    stats = {'n_actions' : 0, 'n_valid' : 0}
    
    command = input('type your command in one line: ')
    i = 0
    while command:
        i += 1
        actions = generate_actions(args, model, tokenizer, command, history, log_file)
        voxel, new_stats = update_state(args, voxel, actions, log_file)
        
        history.append((command, actions))
        
    #    print(len(np.where(voxel!=0)[0]))
      #  print(voxel.sum(axis = 1))
        
        right_voxel = np.transpose(voxel, (1, 0, 2))
        
       # print(len(np.where(right_voxel!=0)[0]))
        #print(right_voxel.sum(axis = 0))
        
	        
        
       # np.save(f'{args.output_dir}/voxel-{i}.npy', voxel)
        
        stats['n_actions'] += new_stats['n_actions']
        stats['n_valid'] += new_stats['n_valid']
        
       # if args.verbose > 0:
         #   n_actions = stats['n_actions']
          #  n_valid = stats['n_valid']
           # logging(log_file, f'turn: {i}, n_actions: {n_actions}, n_valid_actions: {n_valid}')
            
    #    command = input('type your command in one line: ')
        
class DefArgs():
	def __init__(self):
		self.model = "agents/mhb_baseline/nlp_model/t5-autoregressive-history-3-best.pt"
		self.out_dir = None
		self.verbose = 0
		self.logs_path = None
		self.histrory_len = None
		self.max_source_length = 512
		self.max_target_length = 128
		self.init_block = [5,0,5]
		self.task_prefix = 'implement given instructions: '
		
if __name__ == '__main__':
    args = DefArgs()
    
    command = input('type your command in one line: ')
    model, tokenizer, history, stats, voxel = init_models(args)
    history,rv,voxel = predict_voxel(command, model,tokenizer, history, voxel, args)
    
    
    print(len(np.where(rv!=0)[0]))
    print(rv.sum(axis = 0))
    
    print("history: ", history)
    print(voxel.sum(axis = 0))
    
   # main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
