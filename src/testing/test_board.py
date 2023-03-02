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


def train(model, optimizer, scheduler):
    model.train()  # Set model to train mode

    batch = 0
    steps = 0
    train_value_loss = 0
    train_policy_loss = 0
    train_value_acc = 0
    train_policy_acc = 0
    checkpoint = None

    for epoch in range(TrainConfig.nb_training_epochs):
        print('Training epoch: {}/{}'.format(epoch + 1, TrainConfig.nb_training_epochs))
        epoch_loss = 0

        train_set.reset()
        while train_set.load_chunk(TrainConfig.batch_size * 100):
            dataloader = DataLoader(train_set, batch_size=TrainConfig.batch_size, shuffle=True)
            for input_planes, (y_value, y_policy) in dataloader:
                input_planes = input_planes.view(-1, NUM_BUGHOUSE_CHANNELS, BOARD_HEIGHT, 2 * BOARD_WIDTH).to(
                    device, dtype=torch.float)
                y_value = y_value.to(device, dtype=torch.float)
                y_policy = y_policy.to(device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                value, policy = model(input_planes)
                value = value.ravel()

                value_loss = F.mse_loss(value, y_value)
                policy_loss = F.cross_entropy(policy, y_policy)

                loss = TrainConfig.val_loss_factor * value_loss + TrainConfig.policy_loss_factor * policy_loss
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

                win = value > 0
                value[win] = 1
                value[~win] = -1

                train_value_loss += value_loss
                train_policy_loss += policy_loss
                train_policy_acc += torch.sum(torch.argmax(policy, dim=1) == torch.argmax(y_policy, dim=1)) / \
                                    policy.size()[0]
                train_value_acc += torch.sum(value == y_value) / y_value.size()[0]

                steps += 1
                batch += 1
                if batch % TrainConfig.batch_steps == 0:
                    if os.path.exists('../data/history.json'):
                        with open('../data/history.json', 'r') as f:
                            history = json.load(f)
                    else:
                        history = {'train_value_loss': [], 'train_policy_loss': [],
                                   'train_value_acc': [], 'train_policy_acc': [],
                                   'val_value_loss': [], 'val_policy_loss': [],
                                   'val_value_acc': [], 'val_policy_acc': [],
                                   }

                    history['train_value_loss'].append(train_value_loss.item() / steps)
                    history['train_policy_loss'].append(train_policy_loss.item() / steps)
                    history['train_value_acc'].append(train_value_acc.item() / steps)
                    history['train_policy_acc'].append(train_policy_acc.item() / steps)

                    val_loss = evaluate(model, history)

                    if checkpoint and val_loss > TrainConfig.spike_thresh * checkpoint['val_loss']:
                        model.load_state_dict(checkpoint['model_state_dict'])  # Revert to previous weights
                        val_loss = checkpoint['val_loss']

                    checkpoint = {
                        'val_loss': val_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }
                    torch.save(checkpoint, '../weights/checkpoint.pth')

                    steps = 0
                    train_value_loss = 0
                    train_policy_loss = 0
                    train_value_acc = 0
                    train_policy_acc = 0

    torch.save(model.state_dict(), '../weights/model_weights.pth')
    return model


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
            #print(torch.argmax(y_policy, dim=1) >= len(POLICY_LABELS))
            board_num = torch.argmax(y_policy, dim=1) >= len(POLICY_LABELS)
            with torch.no_grad():
                value, policy = model.forward(input_planes)
            correct += torch.sum(board_num == (torch.argmax(policy, dim=1) >= len(POLICY_LABELS)))
            total += 16

    print(correct, total, correct / total)


