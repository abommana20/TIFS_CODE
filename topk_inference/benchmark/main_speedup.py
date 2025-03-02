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
from torx.module.layer import *
from benchmark import replace_layers
import argparse
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16
# from torchsummary import summary
from resnet import *
# from mobilenet import *
from alexnet import *
from vgg8 import *
from lenet import *
from densenet import * 
from VGG import *
from densenet import *
# from 
from energy_eval_sup.energy_eval import energy_eval, baseline_eval
import json

def load_config(file_path):
    """Load the config file."""
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, file_path):
    """Save the modified config file."""
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
##arguments
# Arguments setup

def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # break

    test_accuracy = 100. * correct / total
    avg_test_loss = running_loss / len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        avg_test_loss, correct, total, test_accuracy))
    return avg_test_loss, test_accuracy

def main(args, topk_list):
    ##load dataset
    cuda = args.cuda
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    dataset= args.dataset
    kwargs = {'num_workers': 2, 'pin_memory': True}
    # Data Preparation
    if dataset == 'cifar10':
        # Define the folder name
        folder_name = "cifar-10"

        # Check if the folder exists, and create it if it doesn't
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")
        else:
            print(f"Folder '{folder_name}' already exists.")
        save_path = os.path.join(os.getcwd(), f'{folder_name}') + '/'
        num_classes = 10
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
            testset, batch_size=args.batch_size, shuffle=True, **kwargs)
    elif dataset == 'cifar100':
        folder_name = "cifar-100"

        # Check if the folder exists, and create it if it doesn't
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")
        else:
            print(f"Folder '{folder_name}' already exists.")

        save_path = os.path.join(os.getcwd(), f'{folder_name}') + '/'
        num_classes = 100
        mean =  (0.5071, 0.4867, 0.4408)
        std =  (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, **kwargs)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, **kwargs)

    ##define model
    model_name = args.model

    criterion = nn.CrossEntropyLoss()
    #pth file
    trained_model_path = args.trained_model_path
    model_pth_file = os.path.join(trained_model_path , f'{model_name}_{dataset}.pth')
    config_file_path = os.path.join(os.path.dirname(__file__), f'../config/{args.fault_dist}')
    os.makedirs(os.path.join(os.path.dirname(__file__), f'../config/{args.fault_dist}/{model_name}'), exist_ok=True)
    new_config_file_path = os.path.join(os.path.dirname(__file__), f'../config/{args.fault_dist}/{model_name}')
    # Here, you would add your code to run the experiment based on the config
    if model_name == 'resnet18':
            model = ResNet18(num_classes=10)
            # model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
            model.to(device)
    elif model_name == 'vgg16':
        model = VGG('VGG16',num_classes=10).to(device)
        # model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
        model.to(device)
    elif model_name == 'densenet121':
        model = densenet121(num_classes=10).to(device)
        # model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
        model.to(device)
    elif model_name == 'alexnet':
        model = AlexNet(num_classes=10).to(device)
        # model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
        model.to(device)
    elif model_name == 'lenet':
        model = LeNet(num_classes=10).to(device)
        # model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
        model.to(device)
    elif model_name == 'vgg8':
        model = VGG8(num_classes=10).to(device)
        # model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
        model.to(device)
    elif model_name == 'densenet40':
        model = densenet40(num_classes=10).to(device)
        # model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
        model.to(device)

    # for name, module in replaced_model.named_modules():
    #             if isinstance(module, crxb_Conv2d) or isinstance(module, crxb_Linear):
    #                 module.thermal_noise_mode = args.thermal_mode
    # model = model.to(device)
    # Load the original configuration
    # print(model)
    config = load_config(config_file_path+'/config.json')
    tile_size=96
    def run_experiment(replaced_model,list_acc=[], num_runs=6):
        # global activation_means
        # activation_means={}
        # for name, module in replaced_model.named_modules():
        #         if isinstance(module, crxb_Conv2d) or isinstance(module, crxb_Linear):
        #             module.thermal_noise_mode = args.thermal_mode
        model_tile_fraction = []
        for name, module in replaced_model.named_modules():
                if isinstance(module, crxb_Conv2d) or isinstance(module, crxb_Linear):
                    model_tile_fraction += module.tile_fraction_list
        # print(model_tile_fraction)
        # weight_crxb_shape = []
        # num_weights = []
        # for name, module in replaced_model.named_modules():
        #         if isinstance(module, crxb_Conv2d) or isinstance(module, crxb_Linear):
        #             num_weights.append(module.weight.numel())
        #             weight_crxb_shape.append(module.weight_shape)
        # torch.save(weight_crxb_shape, f"weight_crxb_{model_name}.pt")
        # torch.save(num_weights,f"num_weights_{model_name}.pt")

        test_loss, test_accuracy = validate(replaced_model, test_loader, criterion, device)
        list_acc.append(test_accuracy)
        # torch.save(test_accuracy, save_path + f'{model_name}/acc_run_{num_runs}_{model_name}_{dataset}_temp_{fault_rate}_layer_{0}.pth')
        return model_tile_fraction
        
    
   
        # Run experiments with different configurations
    dict_acc = {}
    dict_loss = {}
    num_runs=1
    # Count the Conv2d and Linear layers
    conv_count = 0
    fc_count = 0

    # Iterate over all modules and check instances
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_count += 1
        elif isinstance(module, nn.Linear):
             fc_count += 1
    total_count = conv_count + fc_count
    print(f"Total number of layers: {total_count}")
    # for inject_layer_index in range(0,total_count):
       
    accuracy_list = []
    # new_config_file_path = config_file_path
    new_config= config
    save_config(new_config, new_config_file_path+'/config.json')
    # print(replaced_model)
    # print(replaced_model.children())
    # quit(0)
    # Run the experiment
    tile_info_path = f"tile_info/tile_info_{model_name}_cifar10.pt"
    tile_info = torch.load(tile_info_path, weights_only = True)
    ##inject noise to only one layer
    #create a list of all ##Just for checking onlyyyyyy
    temp = []
    layer_inject_40 = [11]  # Example list of layers to inject
    layer_inject_20 = [5]
    for i in range(0, total_count):
        if i in [layer - 1 for layer in layer_inject_40]:  # Check if `i` is in the adjusted layer_inject
            temp.append([350] * tile_info[i])
        elif i in [layer - 1 for layer in layer_inject_20]:
            temp.append([350] * tile_info[i])
        else:
            temp.append([350] * tile_info[i])
    torch.save(temp, '.'+'/temp.pt')
    # print(f"running experiment with temp and model: ", temp, 0, model_name)
    # args.path = '.'+'/temp.pt'
    ##Get speedu up each layer
    weight_crxb_path = f"weight_crxb_{args.model}.pt"
    crxb_shape = torch.load(weight_crxb_path, weights_only = True)
    speed_up = []
    for layer in range(len(crxb_shape)):
        layer_shape = crxb_shape[layer]
        total_crxbs_per_layer = layer_shape[0]*layer_shape[1]
        tiles_per_layer = tile_info[layer]
        total_xbs = tiles_per_layer*96 ## number of xbs tile * number of tiles for layer.
        speed_up_layer= math.floor(total_xbs/total_crxbs_per_layer) 
        speed_up.append(min(speed_up_layer,args.speed_up_factor))
    print("speed_up",speed_up)
    torch.save(speed_up, f"speed_up_{args.model}.pt")
    # model_tile_fraction = []
    # avg_accuracy = 0
    for i in range(num_runs):
        set_seed(i)
        print(f"Run {i+1}/{num_runs}")
        replaced_model = replace_layers(model, device,new_config_file_path,path=args.path, thermal_noise_mode=args.thermal_mode, topk_list=topk_list, tm=args.tm, model_name=model_name, enable_fec=args.enable_fec)
        replaced_model.to(device)
        replaced_model.load_state_dict(torch.load(model_pth_file, map_location = device))
        replaced_model.to(device)
        model_tile_fraction = run_experiment(replaced_model,accuracy_list,num_runs)
        # torch.save(dict_acc,save_path+ f'{model_name}_{dataset}_{args.fault_dist}_error.pth')
        ##avg over all runs using torch
        avg_accuracy = torch.tensor(accuracy_list).mean().item()
        print(f'Average accuracy : {avg_accuracy:.4f}%')
        # torch.save(avg_accuracy, save_path + f'{model_name}/acc_{model_name}_{dataset}_temp_{temp}_layer_{0}_avg.pth')
        print(model_tile_fraction,"\n", len(model_tile_fraction))
    return avg_accuracy, model_tile_fraction

   
if __name__ == '__main__':
    # args.model = 'resnet18'
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='resnet18', type=str, help='model name')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--trained_model_path', type=str, default='trained_models', help='path to trained model')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='dataset to use')
    parser.add_argument('--fault_dist', type=str, default='uniform', help='fault dist to be used')
    parser.add_argument('--path', type=str, default='.', help='path to save the traces')
    parser.add_argument('--tm', type=float, default=0.1, help="max fault tolerance in tile")
    parser.add_argument('--thermal_mode', type=int, default=2, help='for analysis of three (linear, exp, 2011) paper noise injection')
    parser.add_argument('--enable_fec', type=int, default=0, help='enable error correction')
    parser.add_argument('--speed_up_factor', type=int, default=2, help='max speed up given')
    args = parser.parse_args()
    thermal_modes = [1]
    
    models = [ "resnet18", "vgg8","densenet40",]
    speed_ups=[96,32,16,8,4,2,1]
    # models = []
    results_dict = {}
    for model_name in models:
        args.model = model_name
        results_dict[model_name]={}
        for speed_up in speed_ups:
            # results_dict[model_name][speed_up]
            args.speed_up_factor = speed_up
            tile_info = torch.load(f"tile_info/tile_info_{args.model}_cifar10.pt", weights_only= True)
            total_count = len(tile_info)
            topk_list = []
            # layer_with_0_1 = []6561279296875, 0.0006561279296875
            layer_with_0_4 = [1,0,4,5,6,7]
            layer_with_0_2 = [2,3]
            for i in range(0, total_count):
                if i in [layer-1 for layer in layer_with_0_4]:
                    topk_list.append(0.1)
                elif i in [layer-1 for layer in layer_with_0_2]:
                    topk_list.append(0.1)
                else:
                    topk_list.append(0.1)
            print(len(topk_list))
            # args.model=model_name
            args.path = f'/data/abommana/research_work/hw_security_CIM/power_trace_gen/results/weight_input_based_attack/{args.model}/20_80/new_tile_temp_attacker.pt'
            # args.path = f"/data/abommana/research_work/hw_security_CIM/power_trace_gen/results3d/weight_input_based_attack/{args.model}/50_50/bottom_tim/new_tile_temp_attacker.pt"
            args.thermal_mode = 2
            args.enable_fec = 1
            avg_accuracy,tile_fraction_list = main(args, topk_list)
            baseline_eval(model_name,"cifar10","./tile_info","./latency", "./power_trace", "./", args.speed_up_factor)
            # energy_eval()
            energy_overhead = energy_eval(tile_fraction_list,model_name, "cifar10", "./tile_info","./crossbar_layers", "./latency", f"./energy_eval_sup/baseline_energy_spup_{args.speed_up_factor}.pt","./", args.tm, 17)
            print("modelname:", model_name,"\n","avg_accuracy:", round(avg_accuracy,2), "energy overhead:", round(energy_overhead[1],3)*100 )
            results_dict[model_name][speed_up]=(round(avg_accuracy,2), round(energy_overhead[1],3)*100)
    # for thermal_mode in thermal_modes:
    #     args.thermal_mode = thermal_mode
    #     avg_accuracy, model_tile_fraction  = main(args,topk_list)
    #     print(f"avg accuracy for {thermal_mode}", avg_accuracy)
    torch.save(results_dict, "results_speed_up.pt")
