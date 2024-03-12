#!/usr/bin/env python3
from itertools import count
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from bothelper import read_board, up, down, left, right, botsetup, rotate_board, merge_count, print_board, restart
from dqnlearner_class import DQNLearner
import os
import numpy as np
import matplotlib.pyplot as plt
import time
# register name
s = botsetup("dqn-train")
    

counter = 0
GAMMA = 0.99        # POUR ALEXIS
EPS_START = 0.9     # POUR ALEXIS
EPS_END = 0.05      # POUR ALEXIS
EPS_DECAY = 2000    # POUR ALEXIS
TAU = 0.005         # POUR ALEXIS
LR = 0.1            # POUR ALEXIS
device = 'cpu'
NB_EPISODES = 25  # POUR ALEXIS

steps_done = 0

if os.path.isfile('./policy_scripted.pt'):
    policy_net = torch.jit.load('./policy_scripted.pt')
    policy_net.eval()
else:
    policy_net = DQNLearner(16, 4)

if os.path.isfile('./target_scripted.pt'):
    target_net = torch.jit.load('./target_scripted.pt')
else:
    target_net = DQNLearner(16, 4)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net.forward(state).max(0).indices.view(1, 1)[0]
    else:
        return torch.tensor([np.floor(np.random.random() * 4)], device=device, dtype=torch.int64)

def optimize_model(state, next_state, action, reward):

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net.forward(state).gather(0, action)

    with torch.no_grad():
        next_state_values = target_net.forward(next_state).max(0).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(0))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

dirs = [down, left, up, right]

def training(num_episodes, render = False):
    min_it = np.inf
    max_it = 0
    max_ep_val = 0
    ep_max_val = 1
    for i in range(num_episodes):
        if (i % 50 == 0 and i != 0):
            print(f"Épisode n°{i}, score moyen des 50 dernières étapes : {np.mean(data[0])}, score de la dernière itération : {data[1][i-1]}. Meilleur score : {np.max(data[1])}")
            print(f"Plus petit score des 50 dernières étapes : {min_it}. Meilleur score des 50 dernières étapes : {max_it}")
            print(f"Score médian des 50 dernières itérations : {np.median(data[0])}")
            print(f"Meilleure valeure atteinte : {max_ep_val} à l'épisode {ep_max_val}")
            data[2] += [np.array([np.mean(data[0])] * 10)]
            data[0] = []
            min_it = np.inf
            max_it = 0
        # Initialize the environment and get its state
        state, info = read_board(s)
        state = np.asarray(state).astype(int)
        state = torch.tensor(state.flatten(), dtype=torch.float, device=device)
        alert_2048 = False
        for t in count():
            action = select_action(state)
            dirs[action](s)
            next_state, info = read_board(s, False)

            if not next_state:
                next_state = None
                break
            
            next_state = np.array(next_state)
            next_state = next_state.astype(int)
            next_state = torch.tensor(next_state.flatten(), dtype=torch.float, device=device)

            try:
                zer_count = dict(zip(*np.unique(next_state, return_counts=True)))[0.0]
                
                max_val = max(next_state)
                if max_val == 2048.0 and not alert_2048:
                    print(f"2048 atteint à l'époque {i}")
                    alert_2048 = True

                reward = (zer_count + info / 100 + max_val) * (sum(state) != sum(next_state)) # POUR ALEXIS
            except Exception:
                reward = 0

            optimize_model(state, next_state, action, reward)

            state = next_state

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if state == None:
                break
        if info > max_it: max_it = info
        if info < min_it: min_it = info
        if max_val > max_ep_val:
            max_ep_val = max_val
            ep_max_val = i
        data[0].append(info)
        data[1].append(info)
        restart(s)
        time.sleep(0.001)
    policy_scripted = torch.jit.script(policy_net)
    policy_scripted.save('policy_scripted.pt')
    target_scripted = torch.jit.script(target_net)
    target_scripted.save('target_scripted.pt')

data = [[],[],[]]

training(num_episodes=NB_EPISODES)

plt.xlabel('Episode')
plt.ylabel('Score')
plt.plot(data[2], label="Moyenne d'itérations")
plt.show()