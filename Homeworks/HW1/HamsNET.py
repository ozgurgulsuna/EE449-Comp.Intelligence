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


import torchvision
transform = transforms.Compose([
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
torchvision.transforms.Grayscale()
])

# training set
train_data = torchvision.datasets.CIFAR10('./data', train = True, download = True,
transform = transform)

# test set
test_data = torchvision.datasets.CIFAR10('./data', train = False,
transform = transform)
