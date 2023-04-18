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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import time

batch_size = 40
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            torchvision.transforms.Grayscale()
            ])

# test set
test_data = torchvision.datasets.CIFAR10('./data', train = False,transform = transform)
print('test_data: ', len(test_data))

# training set
train_data = torchvision.datasets.CIFAR10('./data', train = True, download = True,transform = transform)

# split the training set into training and validation set
random_split = torch.utils.data.random_split(train_data, [int(len(train_data)*90/100), int(len(train_data)*10/100)])
train_data = random_split[0]
val_data = random_split[1]
print('train_data: ', len(train_data))
print('val_data: ', len(val_data))


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



# data loader
train_generator = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
test_generator = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = False)



# scale the data input to [-1,1]
#train_data.data = (train_data.data - 127.5) / 127.5
#val_data.data = (val_data.data - 127.5) / 127.5
# test_data.data = (test_data.data - 127.5) / 127.5

# visualize the data
# image = train_data[0]
# image = image.reshape(3,32,32)
# plt.imshow(image)

dataiter = iter(train_generator)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
















class FullyConnected(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

# initialize your model
model_mlp = FullyConnected(1024,128,10)

# get the parameters 1024x128 layer as numpy array
params_784x128 = model_mlp.fc1.weight.data.numpy()

# create loss: use cross entropy loss)
loss = torch.nn.CrossEntropyLoss()

# create optimizer
# optimizer = torch.optim.SGD(model_mlp.parameters(), lr = 0.01, momentum = 0.0)
optimizer = torch.optim.Adam(model_mlp.parameters(), lr = 0.001)


# transfer your model to train mode
model_mlp.train()

# transfer your model to eval mode
model_mlp.eval()
