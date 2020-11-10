from torch import nn
import torch
from torch.cuda import amp
import os
import numpy as np
from tqdm import tqdm
from model import AlexNet
from preprocessing import get_dataloaders


EPOCHS = 90
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128
PATH = 'ImageNet'

def train(model,
          criterion,
          optimizer,
          scaler,
          train_loader,
          device,
          epochs):

    train_losses = []

    for ep in tqdm(range(epochs)):
        train_batch_loss = []
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_batch_loss.append(loss.item())
        train_losses.append(np.mean(train_batch_loss))

    return np.array(train_losses)


if __name__=='__main__':
    model = AlexNet()
    scaler = amp.GradScaler()
    model = nn.DataParallel(model)

    optimizer = nn.optim.SDG(model.parameters(), lr=LR, 
                             momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropy()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader = get_dataloaders(path=PATH, batch_size=BATCH_SIZE)
    train(model, criterion, optimizer, scaler, train_loader, device, EPOCHS)

