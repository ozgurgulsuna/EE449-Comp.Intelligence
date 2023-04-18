"""-----------------------------------------------------------------------------
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
------------------------------------------------------------------------------"""

import numpy as np
import utils

def my_conv2d(input, kernel):
    # kernel transformation done via flip
    for i in range(kernel.shape[0]):
        kernel[i] = np.flip(kernel[i])
    
    input_height = input.shape[2]
    input_width = input.shape[3]
    input_depth = input.shape[0]
    #print(input_depth)

    kernel_height = kernel.shape[2]
    kernel_width = kernel.shape[3]
    kernel_depth = kernel.shape[0]
    # print(kernel_depth)

    # floor division to get the padding size 
    h = kernel_height // 2
    w = kernel_width // 2

    # out = np.zeros((input.shape))
    out = np.zeros((input_depth, kernel_depth,input_height - kernel_height + 1, input_width - kernel_width + 1))

    # correlate the kernel with the input
    for i in range(input_depth):
        for l in range(kernel_depth):
            for j in range(h,input_height - h):
                for k in range(w,input_width -w):
                    sum = 0
                    for m in range(kernel_height):
                        for n in range(kernel_width):
                            sum += input[i,0, j-h+m, k-w+n] * kernel[l,0, m, n]
                    out[i,l, j-h, k-w] = sum
    return out

# input shape: [batch size, input_channels, input_height, input_width]
input=np.load('./data/samples_8.npy')

# input shape: [output_channels, input_channels, filter_height, filter width]
kernel=np.load('./data/kernel.npy')

# sum = my_conv2d(input, kernel)
out = my_conv2d(input, kernel)

plot = utils.part2Plots(out, nmax=64, save_dir='./out', filename='a')


    