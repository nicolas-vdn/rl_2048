from itertools import count
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from bothelper import read_board, up, down, left, right, botsetup, rotate_board, merge_count, print_board, restart


class DQNLearner(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQNLearner, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)    # POUR ALEXIS
        self.layer2 = nn.Linear(128, 128)               # POUR ALEXIS
        self.layer3 = nn.Linear(128, 256)               # POUR ALEXIS
        self.layer4 = nn.Linear(256, 256)               # POUR ALEXIS
        self.layer5 = nn.Linear(256, 128)               # POUR ALEXIS
        self.layer6 = nn.Linear(128, 128)               # POUR ALEXIS
        self.layer7 = nn.Linear(128, n_actions)         # POUR ALEXIS
        # Definition des couches

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)