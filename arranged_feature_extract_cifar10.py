import argparse
import json
import sagemaker_containers
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def _train(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    input_dir = os.environ.get('SM_CHANNEL_TRAINING')
    print("input_dir is",input_dir)

    train_data = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)
    print("train_data is",train_data)
    print("type(train_data) is",type(train_data))
    test_data = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform)
    print("test_data is",test_data)
    print("type(test_data) is",type(test_data))

    torch.manual_seed(42)  # for reproducible results
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader), len(train_loader.dataset),
        100. * len(train_loader) / len(train_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader), len(test_loader.dataset),
        100. * len(test_loader) / len(test_loader.dataset)
    ))

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_ftrs, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(1024, 10),
                                nn.LogSoftmax(dim=1))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    epochs = args.epochs
    batch_size = args.batch_size

    since = time.time()

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 12)
        running_loss = 0.0
        running_corrects = 0

        for i, (X_train, y_train) in enumerate(train_loader):
            i += 1
            optimizer.zero_grad()

            y_pred = model(X_train)
            _, preds = torch.max(y_pred, 1)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_train.size(0)
            running_corrects += torch.sum(preds == y_train)

#         scheduler.step()

            epoch_loss = running_loss / (batch_size*i)
            epoch_acc = running_corrects.double() * 100/(batch_size*i)
        # statistics
            if i % 100 == 0:
                print(
                    f'Loss: {epoch_loss:4f} batch: {i:4} [{batch_size*i:6}/50000] Accuracy: {epoch_acc:7.3f}%')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return _save_model(model, args.model_dir)


def model_fn(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_ftrs, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(1024, 10),
                                nn.LogSoftmax(dim=1))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def _save_model(model, model_dir):
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str,
                        default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int,
                        default=os.environ['SM_NUM_GPUS'])

    _train(parser.parse_args())