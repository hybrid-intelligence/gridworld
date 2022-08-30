import sys
import json
import os
from os.path import join

import torch
import wandb
import re
import sys
sys.path.append("agents/mhb_baseline/")
from models.models import ResnetEncoderWithTarget

from step_by_step_enjoy import APPOHolder, make_agent
from generator import DialogueFigure
from nlp_model.agent import DefArgs, init_models, predict_voxel
from generator import DialogueFigure, target_to_subtasks 

import numpy as np

def color_random(action_space):
    action = actions_space.sample()
    text = observation['dialog']
    colors_to_hotbar = {
        'blue': 1,  # blue
        'green': 2,  # green
        'red': 3,  # red
        'orange': 4,  # orange
        'purple': 5,  # purple
        'yellow': 6,  # yellow
    }
    hotbars_turned_off = []
    for color, hotbar in colors_to_hotbar.items():
        if color not in text:
            hotbars_turned_off.append(hotbar)
    hotbars_turned_off = set(hotbars_turned_off)
    while (action - 5) in hotbars_turned_off: # actions 6 to 11 are block placements
        action = actions_space.sample()
    # Terminate early with 1% chance
    user_termination = np.random.choice([True, False], p=[0.01, 0.99]) 
    return action, user_termination
	

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
        self.log = False
     #   download_weights()
        self.agent = make_agent()

    def set_action_space(self, action_space):
        self.actions_space = action_space
        
    def act(self, observation, reward, done, info):
        user_termination = False        
        count = re.findall("<Architect>", observation['dialog'])
        
        # If bad phrase do random actions
        if len(count) > 1:            
            return color_random(self.actions_space)
        
        if self.jump_flag == 0:
            
            # First iteration (predict full target + generate first subtask)
            if self.steps == 0:
                if self.log:
                    print()
                    print("-----------------")           
             
                command = observation['dialog'].replace("<Architect>", "").replace("<Builder>", "")
                if self.log: print("Command: ", command)
                # Predict full target and transform it to baseline(multitask) format
                try:
                	self.figure.load_figure(command)
                except Exception as e:
                	print(e)
                	print(command)
                	return color_random(self.actions_space)
                	
                self.generator = target_to_subtasks(self.figure)
                try:
                        coord, task = next(self.generator)
                        self.target_grid = task[:,:,:]
                except StopIteration:
                        user_termination = True
                        self.steps = 0
                        
                        if self.log:
                            print("stop")
                if self.log:
                    print("======================================================================")
                    print("Update task: ", self.target_grid.sum(axis = 0))
                    print("======================================================================")
                    print("-----------------")
                    print()
               
            if ((self.action > self.color_space[0]) and (self.action < self.color_space[1]) > 0):
                if self.log: print("Field was updated: ")
                if self.log: print(observation['grid'].sum(axis = 0))                
                
            self.steps += 1 
            obs = {'obs': observation['pov'],
                    'compass': observation['compass'],
                    'inventory': observation['inventory'],
                    'target_grid': self.target_grid}   
            action = self.agent.act([obs])
            self.action = action
            
    	# If action is put-block
        if  self.jump_flag==1 or ((self.action > self.color_space[0]) and (self.action < self.color_space[1]) > 0):
        	if self.log: print("building ... ")
        	tcolor = np.sum(self.target_grid)
        	colors_to_hotbar = {
                'blue': 1,  # blue
                'green': 2,  # green
                'red': 3,  # red
                'orange': 4,  # orange
                'purple': 5,  # purple
                'yellow': 6,  # yellow
            }
            
        	colors = list(colors_to_hotbar.keys())
        	idx = list(colors_to_hotbar.values())
            
        	hotbar_to_color = dict(zip(idx, colors))
        	text = observation['dialog']
        	if self.log: print("ADD BLOCK with color: ", tcolor, hotbar_to_color[tcolor], text)
           
            
            
           
        	self.jump_flag += 1
        	if self.log: print(self.jump_flag)
            
            # Baseline need two jump
        	if self.jump_flag< 3: 
        	        	if self.log: print("Jump!") 
        	        	return (5, False)
            
        	self.jump_flag %= 3
        	if self.log: print(self.jump_flag)      	
            
        	for color in colors:
        	        	if color in text:
        	        	        	tcolor = colors_to_hotbar[color]
        	action = int(self.color_space[0] + tcolor)
        	if self.log: print("Build!")
        	if self.log: print(action)       	
    		
            # After trying to put a block, we generate a new task
        	try:
                    coord, task = next(self.generator)
                    self.target_grid = task[:,:,:]
                    if self.log: print("======================================================================")
                    if self.log: print("Update task: ", self.target_grid.sum(axis = 0))
                    if self.log: print("======================================================================")
        	except StopIteration:
                    user_termination = True
                    self.steps = 0
                    if self.log: print("stop")
        if done:
        	self.steps = 0
        	self.generator = None
        	self.target_grid = None        
        return action, user_termination


    
