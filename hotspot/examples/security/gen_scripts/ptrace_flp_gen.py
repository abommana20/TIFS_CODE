import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import sys
import copy
import math
import matplotlib.pyplot as plt
import argparse
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16
# from torchsummary import summary
from resnet import *
from densenet import * 
from VGG import *
from functions import *
import json

parser = argparse.ArgumentParser(description='PyTorch CNN Inference')
parser.add_argument('--model', type=str, default='resnet18', help='model to use')
parser.add_argument('--tilepower', type=float, default=0.33, help='tile power')
parser.add_argument('--mempower', type=float, default=0.15, help='memory power')
parser.add_argument('--crxb_size', type=int, default=128, help='crxb size')
parser.add_argument('--cell_res', type=int, default=2, help='cell resolution')
parser.add_argument('--weight_res', type=int, default=8, help='weight resolution')
parser.add_argument('--pe_row', type=int, default=3, help='pe row')
parser.add_argument('--pe_col', type=int, default=2, help='pe col')
parser.add_argument('--array_row', type=int, default=2, help='array row')
parser.add_argument('--array_col', type=int, default=2, help='array col')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--tx', type=float, default=1.0, help='tile width')
parser.add_argument('--ty', type=float, default=1.0, help='tile height')
parser.add_argument('--tsvy', type=float, default=1.0, help='tsv height')

args = parser.parse_args()


# param init:
crxb_size = args.crxb_size
cell_res = args.cell_res
weight_res = args.weight_res
pe_row = args.pe_row
pe_col = args.pe_col
array_row = args.array_row
array_col = args.array_col
tile_power = args.tilepower
mem_power = args.mempower
tx = args.tx
ty = args.ty
tsvy = args.tsvy

#  model init:
if args.model == 'resnet18':
    model = ResNet18(num_classes=args.num_classes)
elif args.model  == 'vgg16':
    model = VGG('VGG16',num_classes=args.num_classes)
elif args.model == 'densenet121':
    model = densenet121(num_classes=args.num_classes)

tile_start_idx = 0
tile_dict = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        # print(name)
        layer_shape = module.weight.shape
        # print(layer_shape)
        num_tile = generate_tile_power(crxb_size, cell_res, weight_res, "conv", layer_shape, pe_row, pe_col, array_row, array_col, tile_dict, tile_power, tile_start_idx)
        tile_start_idx += num_tile
    elif isinstance(module, nn.Linear):
        # print(name)
        layer_shape = module.weight.shape
        # print(layer_shape)
        num_tile = generate_tile_power(crxb_size, cell_res, weight_res, "fc", layer_shape, pe_row, pe_col, array_row, array_col,  tile_dict, tile_power, tile_start_idx)
        tile_start_idx += num_tile
    # print("tile_start_idx",tile_start_idx)
# print(tile_start_idx)
# print(tile_dict.keys())
Nx = math.ceil(math.sqrt((tile_start_idx)))
Ny = math.ceil(tile_start_idx/Nx)
unit_name_tile = "tile"
unit_name_tsv = "tsv"
##floorplan generation.

##copy the generated files to the designated path 
path = f'{args.model}'
#check if the path exists, if not create it
if not os.path.exists(path):
    os.makedirs(path)
#generate_floorplan(Nx, Ny, tx, ty, tsvy, unit_name_tile, unit_name_tsv):
generate_floorplan(Nx, Ny, tx, ty, tsvy, unit_name_tile, unit_name_tsv, args.model, args.model)
##generate memory floorpla just by replacing tile with memtile and tsv with memtsv
generate_floorplan(Nx, Ny, tx, ty, tsvy, "memtile", "memtsv", args.model + str("_mem"), args.model)

##Power trace generation
updated_tile_dict = pad_tile(tile_dict, Nx)
# print(updated_tile_dict.keys())
tsv_dict = {}
length = len(tile_dict)
tsv_dict = add_keys_based_on_name(updated_tile_dict, "tsv",length, 0, tile_start_idx)
mem_tile_d = {}
mem_tile_d = add_keys_based_on_name(tsv_dict, "memtile",length, mem_power, tile_start_idx)
mem_tile_tsv = {}
mem_tile_tsv = add_keys_based_on_name(mem_tile_d, "memtsv",length, 0, tile_start_idx)

# print(mem_tile_tsv.keys())

##filelist
file_list = [f'{args.model}/{args.model}_mem.flp', f'{args.model}/{args.model}.flp']

##save the dict 
# Filename to write to
filename = f'{args.model}/{args.model}.ptrace'
# List of files to read from
output_dict = create_dict_from_files(file_list, mem_tile_tsv)
#printthe keys of thedict 
# print(output_dict.keys())
# Write the dictionary to the file
write_dict_to_file(output_dict, filename)


# #copy the files to the path
# os.system(f'cp {args.model}.ptrace {path}')
# os.system(f'cp {args.model}_mem.flp {path}')
# os.system(f'cp {args.model}.flp {path}')








