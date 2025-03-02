import torch.optim as optim
import numpy as np
import random
import time
import os
import sys
import copy
import math
import matplotlib.pyplot as plt
from resnet import *
from mobilenet import *
from densenet import *
from torx.module.layer import *


## this code replaces the conv2d and linear layers of the given model with the crxb_conv2d and crxb_linear layers from torx library

def replace_layers(trained_model, device, config_path,path, thermal_noise_mode, topk_list, tm, model_name, enable_fec):
    model_copy = copy.deepcopy(trained_model)
    # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
    layer_count = 0
    for name, module in trained_model.named_modules():
        parent_name, child_name = name.rsplit('.', 1) if '.' in name else (None, name)
        if isinstance(module, nn.Conv2d):
            custom_conv = crxb_Conv2d(module.in_channels, module.out_channels,
                                        module.kernel_size, module.stride,
                                        padding=module.padding, dilation=module.dilation, bias=module.bias is not None,device=device,config_path=config_path, layer_count = layer_count, path_to_temp=path, thermal_noise_mode=thermal_noise_mode, topk_list=topk_list, tm=tm, model_name=model_name, enable_fec=enable_fec)
            # Explicitly copy weights and biases
            custom_conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                custom_conv.bias.data = module.bias.data.clone()

            if parent_name:
                setattr(dict(model_copy.named_modules())[parent_name], child_name, custom_conv)
            else:
                setattr(model_copy, child_name, custom_conv)
            layer_count += 1
        elif isinstance(module, nn.Linear):
            custom_fc = crxb_Linear(module.in_features, module.out_features,bias= module.bias is not None,device=device,config_path=config_path, layer_count = layer_count, path_to_temp=path, thermal_noise_mode=thermal_noise_mode, topk_list=topk_list,tm=tm, model_name=model_name, enable_fec=enable_fec)
            # Explicitly copy weights and biases
            custom_fc.weight.data = module.weight.data.clone()
            if module.bias is not None:
                custom_fc.bias.data = module.bias.data.clone()

            if parent_name:
                setattr(dict(model_copy.named_modules())[parent_name], child_name, custom_fc)
            else:
                setattr(model_copy, child_name, custom_fc)
            layer_count += 1
        # print("layer_count",layer_count)
    return model_copy

