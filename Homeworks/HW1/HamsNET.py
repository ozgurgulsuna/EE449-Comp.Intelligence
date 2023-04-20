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

# Parameters ------------------------------------------------------------------------------------------------------------------------------------#
validation_ratio = 0.1
batch_size = 50
epoch_size = 15
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
val_generator = torch.utils.data.DataLoader(val_data, batch_size = batch_size)
test_generator = torch.utils.data.DataLoader(test_data, batch_size = batch_size)

# Architectures ---------------------------------------------------------------------------------------------------------------------------------------#
# "mlp_1" is a simple multi-layer perceptron
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
    
# "mlp_2" is a simple multi-layer perceptron



    

# Training --------------------------------------------------------------------------------------------------------------------------------------------#

# initialize your model
model = mlp_1(input_size=32*32, output_size=10)

# get the parameters 1024x32 layer as numpy array
# we used sequential model, so we can access the layers by index: model_mlp.fc[0].weight.data.numpy()
params_1024x32 = model.fc[0].weight.data.numpy()

# create loss: use cross entropy loss)
criterion = torch.nn.CrossEntropyLoss()

# create optimizer
# optimizer = torch.optim.SGD(model_mlp.parameters(), lr = 0.01, momentum = 0.0)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

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
    model.train()
    i = 0
    for (x, y) in tqdm(iterator, disable=True):
        i += 1
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
        if i % 10 == 9:                                                          # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, (i+1), step_loss / 10))    # each epoch has 5000/50 = 100 steps
            print('training accuracy: %.2f' % (step_acc*100 / (10)) )            # printed at 10 step intervals
            step_loss = 0
            step_acc = 0        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
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
            if i % 10 == 9:                                                          # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, (i+1), step_loss / 10))    # each epoch has 5000/50 = 100 steps
                print('validation accuracy: %.2f' % (step_acc*100 / (10)) )          # printed at 10 step intervals
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
    valid_loss, valid_acc = evaluate(model, val_generator, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'mlp_1-model.pt')

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch:         {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'Train Loss: {train_loss:.3f} |  Train Acc: {train_acc*100:.2f}%')
    print(f'Val. Loss:  {valid_loss:.3f} |   Val. Acc: {valid_acc*100:.2f}%')
    print(f'+-----------------------------------------------------------------------------+')


# for epoch in range(epoch_size):  # loop over the dataset multiple times
#     running_loss = 0.0
#     training_accuracy = 0.0
#     validation_accuracy = 0.0
#     for i, data in enumerate(train_generator, 0):
#         # get the inputs
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = model_mlp(inputs)
#         loss_size = criterion(outputs, labels)
#         loss_size.backward()
#         optimizer.step()
        
#         # print statistics
#         training_accuracy += (outputs.argmax(1) == labels).sum().item()

#         running_loss += loss_size.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#             print('Training accuracy: ', training_accuracy / (batch_size * 2000))
#             print('Validation accuracy: ', validation_accuracy / (batch_size * 2000))
#             training_accuracy = 0.0

# print('Finished Training')

# save your model
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)





# # transfer your model to train mode
# model_mlp.train()

# # transfer your model to eval mode
# model_mlp.eval()



# dataiter = iter(test_generator)
# images, labels = next(dataiter)
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
















# class FullyConnected(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(FullyConnected, self).__init__()
#         self.input_size = input_size
#         self.fc1 = torch.nn.Linear(input_size, hidden_size)
#         self.fc2 = torch.nn.Linear(hidden_size, num_classes)
#         self.relu = torch.nn.ReLU()

#     def forward(self, x):
#         x = x.view(-1, self.input_size)
#         hidden = self.fc1(x)
#         relu = self.relu(hidden)
#         output = self.fc2(relu)
#         return output

# # initialize your model
# model_mlp = FullyConnected(1024,128,10)

# # get the parameters 1024x128 layer as numpy array
# params_784x128 = model_mlp.fc1.weight.data.numpy()

# # create loss: use cross entropy loss)
# loss = torch.nn.CrossEntropyLoss()

# # create optimizer
# # optimizer = torch.optim.SGD(model_mlp.parameters(), lr = 0.01, momentum = 0.0)
# optimizer = torch.optim.Adam(model_mlp.parameters(), lr = 0.001)


# # transfer your model to train mode
# model_mlp.train()

# # transfer your model to eval mode
# model_mlp.eval()

# FMI : for my information
# https://stackoverflow.com/questions/72724452/mat1-and-mat2-shapes-cannot-be-multiplied-128x4-and-128x64
