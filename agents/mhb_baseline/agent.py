import sys
import json
import os
from os.path import join

import torch
import wandb
import re
import sys
sys.path.append("agents/run_baseline/")
from models.models import ResnetEncoderWithTarget

from step_by_step_enjoy import APPOHolder, make_agent, download_weights
from generator import DialogueFigure
from nlp_model.agent import DefArgs, init_models, predict_voxel
from generator import DialogueFigure, target_to_subtasks 

import numpy as np

class APPOAgent:
    def __init__(self):
        self.actions_space = None
        self.target_grid = None
        self.steps = 0
        self.args = DefArgs()
        self.figure = DialogueFigure()
        self.generator = None
        self.min_inventory_value = 5
        self.max_inventory_value = 12
        self.color_space = (self.min_inventory_value, self.max_inventory_value)
        self.model, self.tokenizer, self.history, self.stats, self.voxel = init_models(self.args)
        self.jump_flag = 0
        self.action = -1
        download_weights()
        self.agent = make_agent()

    def set_action_space(self, action_space):
        self.actions_space = action_space
        
    def act(self, observation, reward, done, info):
        user_termination = False        
        count = re.findall("<Architect>", observation['dialog'])
        
        # If bad phrase do random actions
        if len(count) > 1:
            action = self.actions_space.sample()
            # Terminate early with 1% chance
            user_termination = np.random.choice([True, False], p=[0.01, 0.99]) 
            return action, user_termination
        
        if self.jump_flag == 0:
            
            # First iteration (predict full target + generate first subtask)
            if self.steps == 0:
                print()
                print("-----------------")           
             
                command = observation['dialog'].replace("<Architect>", "").replace("<Builder>", "")
                print("Command: ", command)
                # Predict full target and transform it to baseline(multitask) format
                self.figure.load_figure(command)
                self.generator = target_to_subtasks(self.figure)
                try:
                        coord, task = next(self.generator)
                        self.target_grid = task[:,:,:]
                except StopIteration:
                        user_termination = True
                        self.steps = 0
                        print("stop")
                        
                print("======================================================================")
                print("Update task: ", self.target_grid.sum(axis = 0))
                print("======================================================================")
                print("-----------------")
                print()
               
            if ((self.action > self.color_space[0]) and (self.action < self.color_space[1]) > 0):
                print("Field was updated: ")
                print(observation['grid'].sum(axis = 0))                
                
            self.steps += 1 
            obs = {'agentPos': observation['agentPos'],
                    'grid': observation['grid'],
                    'inventory': observation['inventory'],
                    'target_grid': self.target_grid}   
            action = self.agent.act([obs])
            self.action = action
            
    	# If action is put-block
        if  self.jump_flag==1 or ((self.action > self.color_space[0]) and (self.action < self.color_space[1]) > 0):
        	print("building ... ")
        	tcolor = np.sum(self.target_grid)
        	self.jump_flag += 1
        	print(self.jump_flag)
            
            # Baseline need two jump
        	if self.jump_flag< 3: 
        	        	print("Jump!") 
        	        	return (5, False)
            
        	self.jump_flag %= 3
        	print(self.jump_flag)      	

        	action = int(self.color_space[0] + tcolor)
        	print("Build!")
        	print(action)       	
    		
            # After trying to put a block, we generate a new task
        	try:
                    coord, task = next(self.generator)
                    self.target_grid = task[:,:,:]
                    print("======================================================================")
                    print("Update task: ", self.target_grid.sum(axis = 0))
                    print("======================================================================")
        	except StopIteration:
                    user_termination = True
                    self.steps = 0
                    print("stop")
        if done:
        	self.steps = 0
        	self.generator = None
        	self.target_grid = None        
        return action, user_termination


    
