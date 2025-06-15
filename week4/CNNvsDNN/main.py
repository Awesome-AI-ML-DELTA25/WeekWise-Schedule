import torch
import torch.nn as nn
from dataloader import get_loaders
from train import train

from models.model_dense import DenseNet
from models.model_conv import ConvNet

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['dense', 'conv'], default='dense', help='Model type to use')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(batch_size=args.batch_size)

    model = DenseNet() if args.model == 'dense' else ConvNet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    train(model, device, train_loader, test_loader, criterion, optimizer, args.epochs)

    torch.save(model.state_dict(), f"./wts/{args.model}_model.pth")
