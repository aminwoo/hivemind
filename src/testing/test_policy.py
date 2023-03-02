import torch
import torch.nn.functional as F
import json
import os
import math

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from architectures.rise_mobile_v3 import get_rise_v33_model_by_train_config
from configs.train_config import TrainConfig
from preprocessing.bughouse_dataset import BughouseDataset
from utils.constants import BOARD_WIDTH, BOARD_HEIGHT, NUM_BUGHOUSE_CHANNELS, POLICY_LABELS


if __name__ == '__main__':
    device = torch.device('cuda')
    torch.manual_seed(TrainConfig.seed)
    model = get_rise_v33_model_by_train_config((NUM_BUGHOUSE_CHANNELS, BOARD_HEIGHT, 2 * BOARD_WIDTH),
                                               TrainConfig).cuda()
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load('../../weights/checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open('../../data/games.json', 'r') as file:
        games = list(json.load(file).values())

    train_games, val_games = train_test_split(games, test_size=0.001, random_state=42)
    print(len(val_games))
    dataset = BughouseDataset(val_games)
    correct = 0
    total = 0
    while dataset.load_chunk(TrainConfig.batch_size * 100):
        dataloader = DataLoader(dataset, batch_size=TrainConfig.batch_size, shuffle=True)
        for input_planes, (y_value, y_policy) in dataloader:
            input_planes = input_planes.view(-1, NUM_BUGHOUSE_CHANNELS, BOARD_HEIGHT, 2 * BOARD_WIDTH).to(
                device, dtype=torch.float)
            y_value = y_value.to(device, dtype=torch.float)
            y_policy = y_policy.to(device, dtype=torch.float)
            board_num = torch.argmax(y_policy, dim=1) >= len(POLICY_LABELS)
            with torch.no_grad():
                value, policy = model.forward(input_planes)
            correct += torch.sum(torch.argmax(policy, dim=1) == torch.argmax(y_policy, dim=1))
            total += torch.sum(board_num == (torch.argmax(policy, dim=1) >= len(POLICY_LABELS)))


    print(correct, total, correct / total)


