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
import matplotlib.pyplot as plt
from step_by_step_enjoy import APPOHolder, make_agent
from generator import DialogueFigure
from nlp_model.agent import DefArgs, init_models, predict_voxel
from generator import DialogueFigure, target_to_subtasks 
import cv2
import numpy as np

def color_random(observation, actions_space):
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
        self.subtasks = None        
        self.move_action = [6,7,8,9,10,11]
        self.model, self.tokenizer, self.history, self.stats, self.voxel = init_models(self.args)        
        self.jump_flag = 0
        self.action = None
        self.last_action = None
        self.log = False
        self.jump_count = 2
        self.obs_stack = [None, None]
        self.start = True
        self.last_was_action = False
        self.termation = False
        self.action_queue = []
        self.commands = None
     #   download_weights()
        self.agent = make_agent()

    def set_action_space(self, action_space):
        self.actions_space = action_space
        
    def act(self, observation, reward, done, info):
      if self.start:
              commands = self.dialogue_to_commands(observation['dialog'])
              self.commands = commands
              self.figure.load_figure(commands)
              self.subtasks = target_to_subtasks(self.figure)
              #print("do subtasks")
              self.target_grid, self.termation = self.try_update_task()
              self.start = False                
      action_generation, action = self.do_action_from_stack()          
      if action_generation:
          self.last_action = self.action
          self.update_obs_stack(observation)
          self.target_grid, self.termation = self.try_update_task()
         # print(self.termation)
          observation_for_model = self.obs_for_model(observation)
          action = self.agent.act([observation_for_model])              
          self.action = action[0]    
          if self.action == 17:
             #  print("Pass action")
               action = 0
               self.action = 0
          if action in self.move_action:
            action = self.choose_right_color(action)
            jumps = [5 for _ in range(self.jump_count - 1)]
            self.put([ action,*jumps])
            action = 5
          elif self.termation is not True:
            self.termation = False  
      if self.termation == True:
        self.start = True
        self.figure.clear_history()
     # print(self.termation)
      return action, self.termation
    
    def put(self, actions):
        self.action_queue = actions
        return
    
    def update_obs_stack(self, observation):
        kernel = np.ones((5,5),np.float32)/25
        self.obs_stack[-1], self.obs_stack[0] = self.obs_stack[0], self.obs_stack[-1]
        img = observation['pov']
        smooth_img = cv2.filter2D(img,-1,kernel)
        self.obs_stack[-1] = smooth_img
        return
    
    def put_success(self):
        v = np.random.random()
        if np.sum(self.target_grid[0])==0 and v>0.3:
            return True
        if (self.obs_stack[0] is None) or (self.obs_stack[1] is None):
            print("Not enoth pictures!")
            return False
        diff = abs(self.obs_stack[-1].mean(axis = 2) - self.obs_stack[0].mean(axis = 2))
        num = np.random.randint(0,100)
        xc, yc = np.array(diff.shape)//2
        thresh = 60
        if diff[xc, yc] >= thresh:
           # plt.imsave("imgs/true_diff%d.png"%num,diff)
            return True
      #  plt.imsave("imgs/false_diff%d.png"%num,diff)
        return False
        
    def try_update_task(self):
      #  print(self.start)
        if (self.last_action is not None) or self.start:
           # print("Here")
            if (self.last_action in self.move_action and self.put_success()) or self.start:
                try:
                  #  print("I am here!")
                    _, target_grid = next(self.subtasks)
                    print("Change task!")
                  #  print(target_grid.sum(axis = 0))
                    return target_grid, False
                except Exception as e:
                    print(e)
                    return self.target_grid, True 
       # print("Fail!")
        return self.target_grid, self.termation
    
    def dialogue_to_commands(self, full_dialogue):
        commands = re.split("(<Architect>)|(<Builder>)", full_dialogue)
       # print(commands)
        no_none = [command for command in commands if command is not None]
       # print(no_none)
        no_zero_len = [command for command in no_none if len(command)>0]
      #  print(no_zero_len)
        atchitect = no_zero_len[1::4]
       # print(atchitect)
        return atchitect
    
    def obs_for_model(self, observation):
        obs = {'obs': observation['pov'],
                'compass': observation['compass'],
                'inventory': observation['inventory'],
                'target_grid': self.target_grid}  
        return obs
    
    def do_action_from_stack(self):
        try:
            action = self.action_queue.pop()
            return False, action 
        except:
            return True, None
        
    def choose_right_color(self, action):
        
        #tcolor = np.sum(self.target_grid)
        colors_to_hotbar = {
                'blue': 1,  # blue
                'green': 2,  # green
                'red': 3,  # red
                'orange': 4,  # orange
                'purple': 5,  # purple
                'yellow': 6,  # yellow
            }
        print(self.commands[-1])
        for key_color in colors_to_hotbar:
            if key_color in self.commands[-1]:
                tcolor = colors_to_hotbar[key_color]
                break
        colors = list(colors_to_hotbar.keys())
        idx = list(colors_to_hotbar.values())
        hotbar_to_color = dict(zip(idx, colors))
        action = int(5 + tcolor)
        return 6
        

               
                
          
          
            
          
          
    
