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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import time

batch_size = 2
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




# Hyper-parameters
transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            torchvision.transforms.Grayscale()
            ])

# test set
test_data = torchvision.datasets.CIFAR10('./data', train = False,transform = transform)

# training set
train_data_original = torchvision.datasets.CIFAR10('./data', train = True, download = True,transform = transform)

# split the training set into training and validation set
train_data, val_data = torch.utils.data.random_split(train_data_original, [int(len(train_data_original)*90/100), int(len(train_data_original)*10/100)])

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


# data loader
train_generator = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_generator = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True)
test_generator = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = True)


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
    

      

# initialize your model
model_mlp = mlp_1(input_size=32*32, output_size=10)

# get the parameters 1024x32 layer as numpy array
# we used sequential model, so we can access the layers by index: model_mlp.fc[0].weight.data.numpy()
params_1024x32 = model_mlp.fc[0].weight.data.numpy()

# create loss: use cross entropy loss)
criterion = torch.nn.CrossEntropyLoss()

# create optimizer
# optimizer = torch.optim.SGD(model_mlp.parameters(), lr = 0.01, momentum = 0.0)
optimizer = torch.optim.Adam(model_mlp.parameters(), lr = 0.001)

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = model_mlp.to(device)
# criterion = criterion.to(device)


for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_generator, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_mlp(inputs)
        loss_size = criterion(outputs, labels)
        loss_size.backward()
        optimizer.step()

        # shuffling the data
        # train_generator = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)



        
        # print statistics
        running_loss += loss_size.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        
print('Finished Training')

# save your model
PATH = './cifar_net.pth'
torch.save(model_mlp.state_dict(), PATH)





# transfer your model to train mode
model_mlp.train()

# transfer your model to eval mode
model_mlp.eval()



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
