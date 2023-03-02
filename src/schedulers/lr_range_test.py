from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
from torch import nn, optim
import json
import torch
from architectures.rise_mobile_v3 import get_rise_v33_model_by_train_config
from configs.train_config import TrainConfig
from preprocessing.bughouse_dataset import BughouseDataset
import torch.nn.functional as F

from utils.constants import NUM_BUGHOUSE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH

with open('../../data/filtered_games.json', 'r') as file:
    games = list(json.load(file).values())
print(len(games))
dataset = BughouseDataset(games)
dataset.load_chunk()
trainloader = DataLoader(dataset, batch_size=16, shuffle=True)


def criterion(output, target):
    value = output[0]
    policy = output[1]
    y_value = target[0]
    y_policy = target[1]
    value_loss = F.mse_loss(value.ravel(), y_value)
    policy_loss = F.cross_entropy(policy, y_policy.float())
    loss = 0.01 * value_loss + 0.99 * policy_loss
    return loss


model = get_rise_v33_model_by_train_config((NUM_BUGHOUSE_CHANNELS, BOARD_HEIGHT, 2*BOARD_WIDTH), TrainConfig).cuda()
#optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, nesterov=True, momentum=0.95, weight_decay=1e-4)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(trainloader, end_lr=100, num_iter=100)
lr_finder.plot()  # to inspect the loss-learning rate graph
lr_finder.reset()  # to reset the model and optimizer to their initial state