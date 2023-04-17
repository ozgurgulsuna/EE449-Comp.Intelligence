"""
--------------------------------------------------------------------------------
2D Convolution with NumPy
--------------------------------------------------------------------------------
Author: Ozgur Gulsuna
Date: 2023-04-14

Description:
This is a simple implementation of 2D convolution with NumPy.

input: [batch size, input_channels, input_height, input_width]
kernel: [output_channels, input_channels, filter_height, filter width]

source: https://www.youtube.com/watch?v=Lakz2MoHy6o
        https://github.com/TheIndependentCode/Neural-Network
--------------------------------------------------------------------------------
"""



import numpy as np
import utils



def my_conv2d(input, kernel):
    # kernel transformation done via flip
    for i in range(kernel.shape[0]):
        kernel[i] = np.flip(kernel[i])
    
    image_height = input.shape[2]
    image_width = input.shape[3]

    filter_height = kernel.shape[2]
    filter_width = kernel.shape[3]

    # floor division to get the padding size 
    h = filter_height // 2
    w = filter_width // 2

    out = np.zeros((input.shape))

    for i in range(h, image_height-h):
        for j in range(w, image_width-w):
            for k in range(kernel.shape[0]):
                out[:, k, i, j] = np.sum(input[:, :, i-h:i+h+1, j-w:j+w+1] * kernel[k], axis=(1,2,3))
    return out


# input shape: [batch size, input_channels, input_height, input_width]
input=np.load('./data/samples_8.npy')

# input shape: [output_channels, input_channels, filter_height, filter width]
kernel=np.load('./data/kernel.npy')

out = my_conv2d(input, kernel)


plot = utils.part2Plots(input[2], nmax=64, save_dir='./out', filename='a')

print(kernel)




    