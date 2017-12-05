#!/usr/bin/env python2

"""
Created on Tue Nov 21 21:32:59 2017

@author: Dmayna
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating architecture of Neural Network as object
class Network(nn.Module): #inheritance from module parent clas
    
    def __init__(self, numOfInputs, numOfOutputs):
        super(Network, self).__init__() # allows to use tools of Module class
        self.numOfInputs = numOfInputs
        self.numOfOutputs = numOfOutputs
        self.fullConnection1 = nn.Linear(numOfInputs, 30) #creates connection bewteen input neurons and first hidden layer with 30 hidden neurons
        self.fullConnection2 = nn.Linear(30, numOfOutputs) # creates connectiion between hidden layer with 30 hidden neurons and output layer 
        
    def feedForward(self, state): # returns q values based off of input state
        x = F.relu(self.fullConnection1(state)) # ativate hidden neurons with rectifier function from current state with torch.nn.functional
        q_values = self.fullConnection2(x) 
        return q_values # returns q_values(outout) to make decision of direction
    

# Implementing Experience Replay
class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity = capacity # memory capacity
        self.memory = [] #initilize memory list of 100 events
        
    def push(self, event): # append event input into memory
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, sample_size):
        # if list =  ((1,2,3),(4,5,6), then zip(*list) = ((1,4), (2,3), (5,6))
        # ex) sample =  ((state1,action1,reward1), ((state2,action2,reward2)), then zip(*sample) = ((state1,state2), (action1,action2), (reward1,reward2))
        samples = zip(*random.sample(self.memory, sample_size))
        # lmabda function will take samples and concatenate from first dimension (states) and then convert tensors into torch variables with tensors and gradients
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
# Implementing Deep Q Learning

class DeepQNetwork():
    
    def __init__(self, numOfInputs, numOfOutputs, gamma):
        self.gamma = gamma
        self.reward_window = [] # mean of last 100 rewards
        self.model = Network(numOfOutputs,numOfOutputs) # model of network for deep q learning model
        self.memory = ReplayMemory(100000) # object of memory class
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) #connects optimizer to neural network with a learning rate
        self.lastState = torch.Tensor(numOfInputs).unsqueeze(0) # creates a tensor for input  states and creates a fake dimension corresponding to the samples
        self.lastAction = 0
        self.lastReward = 0
        
    def selectAction(self, state):
        probabilities = F.softmax(self.model(Variable(state, volatile = True))*100) #softmax function to give probalilities to q values. Variable turns Tensor into a Pytorch variable anf Volatile excludes input gradient state.T = 7
        # softmax([1,2,3]) = ([0.04,0.11,0.85]) => softmax([1,2,3]*T) = ([0.0,0.02,0.98])
        action = probabilities.multinomial() # random draw from this distribution 
        return action.data[0,0]
    
    def learn(self,sampleState,sampleNextState,sampleReward,sampleAction):
        outputs = self.model(sampleState).gather(1,sampleAction.unsqueeze(1)).squeeze(1) # gathers choosen actions from each state
        next_outputs = self.model(sampleNextState).detach().max(1)[0] # maximum of Q values with respect to the action(1) from Q values of the next state[0]
        target = self.gamma*next_outputs + sampleReward
        temporalDifferenceLoss = F.smooth_l1_loss(outputs, target) # calculates temporal difference between outputs and targets
        self.optimizer.zero_grad() # Reinitilize optimizer at each iteration of the loop
        temporalDifferenceLoss.backward(retain_variables = True) #back propagates error into neural network
        self.optimizer.step() # Updates weights 
        
    def update(self, reward, newSignal):
        newState = torch.Tensor(newSignal).float().unsqueeze(0)
        self.memory.push((self.lastState,newState,torch.LongTensor([int(self.lastAction)]),torch.Tensor([self.lastReward])))
        action = self.selectAction(newState)
        if len(self.memory.memory) > 100:
            sampleSate, sampleNextState, sampleReward, sampleAction = self.memory.sample(100) # learning will be from 100 transitions
            self.learn(sampleSate, sampleNextState, sampleReward, sampleAction)
        self.lastAction = action
        self.lastState = newState
        self.lastReward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/len(self.reward_window+1.)
    
    def save(self):
        torch.save({'state_dict':self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict()
                    },'lastBrain.pth')
    
    def load(self):
        if os.path.isfile('lastBrain.pth'):
            print("Loading checkpoint ...")
            checkpoint = torch.load('lastBrain.pth')
            self.model.load_state_dict(checkpoint['state_dict']) # will update model (weights and parameters)
            self.optimizer.load_state_dict(checkpoint['optimizer']) # will update optomizers for model
        else:
            print("Sorry no checkpoint found ...")
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        