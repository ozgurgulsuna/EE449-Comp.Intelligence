## HamsterNET: A Convolutional Neural Network that is small and fast, like a hamster running on a wheel.

# 
#           ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                                
#         ▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓    ▒▒▒▒▒▒                                          
#       ░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    ░░░░▒▒██▒▒                                      
#     ░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓        ░░▓▓▓▓▓▓██                                    
#     ▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░      ▒▒▓▓▒▒██▓▓░░██                                  
#   ▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓        ▓▓▒▒  ░░▒▒▓▓  ▓▓                                
#   ▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓        ▓▓▒▒  ▒▒▓▓▓▓▒▒  ██                              
#   ▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓        ▓▓▓▓▓▓▒▒▓▓▒▒  ▓▓▓▓  ██                            
#   ▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓        ▓▓▓▓▓▓▓▓▓▓▒▒▒▒▓▓▓▓  ██                            
#     ▒▒▓▓▓▓▓▓▓▓▓▓▓▓        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ██                            
#       ▒▒▒▒▒▒▒▒▓▓▓▓        ▒▒░░▒▒▒▒▓▓▓▓▓▓▓▓▓▓██                              
#       ▒▒  ░░░░▒▒▒▒░░▒▒░░░░▒▒  ░░░░██████████                                
#         ▒▒▒▒▒▒              ▒▒▒▒░░                                          
                                                    
# CUDA tutorial: https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# PyTorch MLP tutorial: https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb#scrollTo=FTvjOcbLREwM
# PyTorch CNN tutorial: https://github.com/bentrevett/pytorch-image-classification/blob/master/2_lenet.ipynb


# Imports --------------------------------------------------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm

import time
import json

# Parameters ------------------------------------------------------------------------------------------------------------------------------------#
validation_ratio = 0.1
batch_size = 50
epoch_size = 15
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model_name = 'mlp_1'

DISPLAY = True

# Record ----------------------------------------------------------------------------------------------------------------------------------------------#
save = True
save_path = './HamsNET.pt'
training_loss_record = []
training_loss_gradient_record = []
test_acc_record = []


# Transformations ------------------------------------------------------------------------------------------------------------------------------------#
transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            torchvision.transforms.Grayscale()
            ])

# Data ----------------------------------------------------------------------------------------------------------------------------------------------#
# test set
test_data = torchvision.datasets.CIFAR10('./data', train = False, download = True, transform = transform)

# training set
train_data_original = torchvision.datasets.CIFAR10('./data', train = True, download = True,transform = transform)

# split the training set into training and validation set
train_data, val_data = torch.utils.data.random_split(train_data_original, [int(len(train_data_original)*(1-validation_ratio)), int(len(train_data_original)*validation_ratio)])

# lenghts of each set
print('train_data: ', len(train_data))
print('val_data: ', len(val_data))
print('test_data: ', len(test_data))

# +-----------------------------------------------------------------------------------+
# |                                       Dataset                                     |
# | █████████████████████████████████████████████████████████████████████████████████ |
# |                                        60K                                        |
# +-----------------------------------------------------------------------------------+
# |                           Training                           | Val. |    Test     |
# | =============================================================|======|============ |
# | █████████████████████████████████████████████████████████████|██████|████████████ |
# |                              45K                             |  5K  |     10K     |
# +-----------------------------------------------------------------------------------+


# Data loader ----------------------------------------------------------------------------------------------------------------------------------------#
train_generator = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_generator = torch.utils.data.DataLoader(val_data, batch_size = batch_size )
test_generator = torch.utils.data.DataLoader(test_data, batch_size = batch_size )

# Architectures ---------------------------------------------------------------------------------------------------------------------------------------#
# "mlp_1" is a simple multi-layer perceptron with one hidden layer
class mlp_1(nn.Module):
    def __init__(self, input_size, output_size):
        super(mlp_1,self).__init__()
        self.input_size = input_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),                      # 1024x32
            nn.ReLU())                                      
        self.prediction_layer = nn.Linear(32, output_size)  # 32x10
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc(x)
        x = self.prediction_layer(x)
        return x
    
# "mlp_2" is a simple multi-layer perceptron with two hidden layers
class mlp_2(nn.Module):
    def __init__(self, input_size, output_size):
        super(mlp_2,self).__init__()
        self.input_size = input_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),                      # 1024x32
            nn.ReLU(),
            nn.Linear(32, 64))                              # 32x64
        self.prediction_layer = nn.Linear(64, output_size)  # 64x10
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc(x)
        x = self.prediction_layer(x)
        return x
    
# "cnn_3" is a simple convolutional neural network with three convolutional layers
class cnn_3(nn.Module):
    def __init__(self, output_size):
        super(cnn_3,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 1x32x32 -> 16x32x32
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, padding=2)  # 16x32x32 -> 8x32x32
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)                                       # 8x32x32 -> 8x16x16                        
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, padding=3)  # 8x16x16 -> 16x16x16
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)                                       # 16x16x16 -> 16x8x8
        self.prediction_layer = nn.Linear(16 * 8 * 8, output_size)                        # 16x8x8 -> 10

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.prediction_layer(x)
        return x
    
# "cnn_4" is a simple convolutional neural network with four convolutional layers
class cnn_4(nn.Module):
    def __init__(self, output_size):
        super(cnn_4,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 1x32x32 -> 16x32x32
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)  # 16x32x32 -> 8x32x32
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2)  # 8x32x32 -> 16x32x32
        self.relu3 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)                                       # 16x32x32 -> 16x16x16
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2) # 16x16x16 -> 16x16x16
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)                                       # 16x16x16 -> 16x8x8
        self.prediction_layer = nn.Linear(16 * 8 * 8, output_size)                        # 16x8x8 -> 10

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.prediction_layer(x)
        return x
    
# "cnn_5" is a simple convolutional neural network with six convolutional layers
class cnn_5(nn.Module):
    def __init__(self, output_size):
        super(cnn_5,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)   # 1x32x32 -> 8x32x32
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)  # 8x32x32 -> 16x32x32
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)  # 16x32x32 -> 8x32x32
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)  # 8x32x32 -> 16x32x32
        self.relu4 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)                                       # 16x32x32 -> 16x16x16
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) # 16x16x16 -> 16x16x16
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)  # 16x16x16 -> 8x16x16 
        self.relu6 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)                                       # 8x16x16 -> 8x8x8
        self.prediction_layer = nn.Linear(8 * 8 * 8, output_size)                         # 8x8x8 -> 10

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool1(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.prediction_layer(x)
        return x


# Training --------------------------------------------------------------------------------------------------------------------------------------------#

# initialize your model
if model_name == "mlp_1":
    model = mlp_1(input_size=32*32, output_size=10)
elif model_name == "mlp_2":
    model = mlp_2(input_size=32*32, output_size=10)
elif model_name == "cnn_3":
    model = cnn_3(output_size=10)
elif model_name == "cnn_4":
    model = cnn_4(output_size=10)
elif model_name == "cnn_5":
    model = cnn_5(output_size=10)
else:
    print("Error: model name is not correct!")

# create loss: use cross entropy loss
criterion = torch.nn.CrossEntropyLoss()

# create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.0)
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    accuracy = correct.float() / y.shape[0]
    return accuracy

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    step_loss = 0
    step_acc = 0
    step_loss_gradient = 0
    model.train()
    i = 0
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        step_loss += loss.item()
        step_acc += acc.item()
        if model_name == "mlp_1" or model_name == "mlp_2":
            step_loss_gradient += torch.norm(model.fc[0].weight.grad).item()
        elif model_name == "cnn_3" or model_name == "cnn_4" or model_name == "cnn_5":
            step_loss_gradient += torch.norm(model.conv1.weight.grad).item()
        if i % 10 == 9:
            if DISPLAY is True:                                                        # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, (i+1), step_loss / 10))    # each epoch has 5000/50 = 100 steps
                print('training accuracy: %.2f' % (step_acc*100 / (10)) )            # printed at 10 step intervals
            training_loss_record.append(step_loss / 10)                          # save training loss with 10 step intervals
            training_loss_gradient_record.append(step_loss_gradient / 10)            # save training loss gradient with 10 step intervals
            step_loss = 0
            step_acc = 0
            step_loss_gradient = 0 
        i += 1
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device,sv=0):
    epoch_loss = 0
    epoch_acc = 0
    step_loss = 0
    step_acc = 0
    model.eval()
    with torch.no_grad():
        i = 0
        for (x, y) in tqdm(iterator, disable=True):
            i += 1
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            step_loss += loss.item()
            step_acc += acc.item()
            if i % 10 == 9:
                if DISPLAY is True:                                                      # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, (i+1), step_loss/10))    # each epoch has 5000/50 = 100 steps
                    print('validation accuracy: %.2f' % (step_acc*100/10) )          # printed at 10 step intervals
                step_loss = 0
                step_acc = 0
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Training loop ---------------------------------------------------------------------------------------------------------------------------------------#

best_valid_loss = float('inf')

for epoch in trange(epoch_size,disable=True):

    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_generator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, val_generator, criterion, device, sv=0)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './results/trained_models/'+ model_name+'['+'ReLU-AF'+'].pt')

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'+---------------------------------------+')
    print(f'Epoch:        {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'Train Loss: {train_loss:.3f} |  Train Acc: {train_acc*100:.2f} %')
    print(f'Val.  Loss: {valid_loss:.3f} |   Val. Acc: {valid_acc*100:.2f} %')
    print(f'+---------------------------------------+')

# Testing-----------------------------------------------------------------------------------------------------------------------------------------------#
# Load the best model in run
model.load_state_dict(torch.load('./results/trained_models/'+ model_name+'['+'ReLU-AF'+'].pt'))

# Evaluate the model on the test set
test_loss, test_acc = evaluate(model, test_generator, criterion, device, sv=0)
print(f'+---------------------------------------+')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
print(f'+---------------------------------------+')
test_acc_record = test_acc*100

# End of training loop ----------------------------------------------------------------------------------------------------------------------------------#

# Load the best model in run -----------------------------------------------------------------------------------------------------------------------------#
model.load_state_dict(torch.load('./results/trained_models/'+ model_name+'['+'ReLU-AF'+'].pt'))

# Save the results ----------------------------------------------------------------------------------------------------------------------------------#
with open("./results/["+ model_name +']ReLU-AF_training_loss_record', "w") as fp:
    json.dump(training_loss_record, fp)
with open("./results/["+ model_name +']ReLU-AF_test_acc_record', "w") as fp:
    json.dump(test_acc_record, fp)
with open("./results/["+ model_name +']ReLU-AF_training_loss_gradient_record', "w") as fp:
    json.dump(training_loss_gradient_record, fp)


# FMI : for my information
# https://stackoverflow.com/questions/72724452/mat1-and-mat2-shapes-cannot-be-multiplied-128x4-and-128x64
