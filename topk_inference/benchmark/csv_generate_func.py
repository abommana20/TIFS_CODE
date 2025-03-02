import argparse
import os
import time

import torch
import torch.nn as nn
# from cifar import model

import numpy as np
import csv
from subprocess import call
import subprocess
import csv

def count_conv_operations(ifm_height, ifm_width, filter_height, filter_width, stride, padding):
    # Calculate the dimensions of the output feature map (OFM)
    ofm_height = (ifm_height - filter_height + 2 * padding) // stride + 1
    ofm_width = (ifm_width - filter_width + 2 * padding) // stride + 1
    
    # The number of times the filter is applied is equal to the number of positions
    # in the output feature map
    num_operations = ofm_height * ofm_width
    
    return num_operations

# # Example usage:
# ifm_height = 32  # Height of the input feature map (e.g., 32 pixels)
# ifm_width = 32   # Width of the input feature map (e.g., 32 pixels)
# filter_height = 3  # Height of the convolution filter (e.g., 3 pixels)
# filter_width = 3   # Width of the convolution filter (e.g., 3 pixels)
# stride = 1        # Stride of the convolution
# padding = 0       # Padding applied to the input feature map

# num_operations = count_conv_operations(ifm_height, ifm_width, filter_height, filter_width, stride, padding)
# print(f"Number of convolution operations: {num_operations}")

def print_feature_sizes(model, input_tensor, model_name, csv_filename='layer_feature_sizes.csv'):
    # Forward hook to capture the input and output dimensions of each layer
    csv_data = []
    csv_operations = []
    num_operations = {}
    def hook_fn(module, input, output):
        if isinstance(module, nn.Conv2d):
            # Extract IFM (Input Feature Map) details
            ifm_channels = input[0].size(1)
            ifm_height = input[0].size(2)  #+ 2 * module.padding[0]
            ifm_width = input[0].size(3) #+ 2 * module.padding[1]
            
            # Extract kernel details
            kernel_height = module.kernel_size[0]
            kernel_width = module.kernel_size[1]

            #get stride
            stride = module.stride[0]
            
            # Extract output channels
            ofm_channels = module.out_channels

            # # Determine if the next layer is a MaxPool
            # is_followed_by_maxpool = 0
            # for next_layer in model.children():
            #     if isinstance(next_layer, nn.MaxPool2d):
            #         is_followed_by_maxpool = 1
            #         break
            if ifm_height < kernel_height:
                ifm_height = input[0].size(2)  + 2 * module.padding[0]
            if ifm_width < kernel_width:
                ifm_width = input[0].size(3) + 2 * module.padding[1]
            csv_operations.append([count_conv_operations(ifm_height, ifm_width, kernel_height, kernel_width, stride, module.padding[0])*8])
            

            # Add the row to csv_data
            csv_data.append([ifm_width, ifm_height, ifm_channels, kernel_width, kernel_height, ofm_channels,0,stride])
            # print("padding",module.padding)

            # print(ifm_width, ifm_height, ifm_channels, kernel_width, kernel_height, ofm_channels,0,1)
        elif isinstance(module, nn.Linear):
            ifm_channels = input[0].size(1)
            ofm_channels = module.out_features
            csv_data.append([1, 1, ifm_channels, 1, 1, ofm_channels,0,1])
            csv_operations.append([8])
            # print(1, 1, ifm_channels, 1, 1, ofm_channels,0,1)

    # Register hooks on each layer of the model
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook_fn))

    # Pass the input tensor through the model to trigger the hooks
    model(input_tensor)

    # Remove all hooks
    for hook in hooks:
        hook.remove()

    # Post-process to check for ReLU and MaxPool2d sequences
    csv_index = 0  # To keep track of corresponding index in csv_data
    modules = list(model.modules())
    for i, module in enumerate(modules):
        if isinstance(module, nn.Conv2d):
            if i+1 < len(modules)  and i+2 < len(modules) and i+3 < len(modules):
                if isinstance(modules[i+1], nn.MaxPool2d) or isinstance(modules[i+2], nn.MaxPool2d) or isinstance(modules[i+3], nn.MaxPool2d)  \
                or isinstance(modules[i+1], nn.AvgPool2d) or isinstance(modules[i+2], nn.AvgPool2d) or isinstance(modules[i+3], nn.AvgPool2d)  \
                    or isinstance(modules[i+1], nn.AdaptiveAvgPool2d) or isinstance(modules[i+2], nn.AdaptiveAvgPool2d) or isinstance(modules[i+3], nn.AdaptiveAvgPool2d):
                    # Find the corresponding row in csv_data and set the last but one element (index -2) to 1
                    if csv_index < len(csv_data):
                        csv_data[csv_index][-2] = 1
            csv_index += 1
                # csv_data[i][-1] = 1


    # Write the collected data to a CSV file
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
    with open(f'num_operations_{model_name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_operations)

    # torch.save(num_operations, f'num_operations_{model_name}.pt')

    # # Write the collected data to a CSV file
    # with open(csv_filename, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(csv_data)



