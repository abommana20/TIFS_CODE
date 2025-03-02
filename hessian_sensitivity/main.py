import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
from resnet import *
from densenet import *
from VGG import *
from alexnet import *
from lenet import *
from vgg8 import *
from functions import *
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True}
dataset_name = "cifar10"
# model_names = ['resnet18', 'vgg16', 'densenet121', 'densenet40', 'alexnet', 'lenet', 'vgg8']
model_names = ['densenet121', 'densenet40', 'alexnet', 'lenet', 'vgg8']
max_power_array = 8e-3
peripheral_power = 2.7e-3
mean = (0.4914, 0.4822, 0.4465)
std= (0.2470, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, **kwargs)
for model_name in model_names:
    print(f"Generating hessian for {model_name} on {dataset_name}...")
    if model_name == 'resnet18':
        model = ResNet18(num_classes=10).to(device)
    elif model_name == 'vgg16':
        model = VGG('VGG16',num_classes=10).to(device)
    elif model_name == 'densenet121':
        model = densenet121(num_classes=10).to(device)
    elif model_name == 'alexnet':
        model = AlexNet(num_classes=10).to(device)
    elif model_name == 'lenet':
        model = LeNet(num_classes=10).to(device)
    elif model_name == 'vgg8':
        model = VGG8(num_classes=10).to(device)
    elif model_name == 'densenet40':
        model = densenet40(num_classes=10).to(device)
    save_path = f"hessian_pt/"
    # Load the model
    model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
    model.to(device)

    test_loader_b = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, **kwargs)
    
    criterion = nn.CrossEntropyLoss()


    model.eval()
    hessian_layer = []
    first_batch = True
    # hessian_filter_b[f'batch_size_{b}'] = []
    # hessian_filter = hessian_filter_b[f'batch_size_{b}']
    start_time = time.time()
    for batch_idx, (data, targets) in enumerate(test_loader_b, 0):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)          
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            # ntrace = get_trace_hut(model, loss,100, device, layerwise = False, filterwise=False)
            

            ltrace = get_trace_hut(model, loss,10, device, layerwise = True, filterwise=False)
            if first_batch:
                ##initialize the hessian for the first batch
                for i in range(len(ltrace)):
                    hessian_layer.append(torch.zeros(ltrace[i].shape).to(device))
                first_batch = False
    
            for i in range(len(ltrace)):
                hessian_layer[i] += ltrace[i]
    
            # crct_pred.append((correct,batch_idx))
           
            if batch_idx % 500 == 0:
                print(f'no of completed images {batch_idx+1}') 
                # break
            end_time = time.time()
            # print(f'Time taken for {batch_idx} is {end_time-start_time}')
            # quit(0)
            # print(M)
    average_of_hessian = [ hessian_layer[i]/len(test_loader_b) for i in range(len(hessian_layer))]
    torch.save(average_of_hessian,save_path + f'hessian_layer_{model_name}.pt')