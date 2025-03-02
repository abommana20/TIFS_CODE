from extract_input_weight_traces import *
import torch
from resnet import *
from densenet import *
from VGG import *
from create_floorplan import *
from extract_tile_info import *
import subprocess
import shutil
import os
import re

def extract_selected_tile_values(filename, indices):
    adjusted_indices = [i + 1 for i in indices]  # Adjust indices by adding 1
    tile_values = []  # List to hold the values of selected tiles
    tile_pattern = re.compile(r'tile(\d+)')  # Regex to extract the tile number

    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()  # Split each line into parts based on whitespace
                if len(parts) == 2:
                    match = tile_pattern.search(parts[0])  # Search for the tile pattern
                    if match:
                        tile_number = int(match.group(1))  # Extract the numeric part
                        if tile_number in adjusted_indices:  # Check if the tile number is in the list
                            value = float(parts[1])  # Convert the second part (value) to float
                            tile_values.append(value)  # Append the value to the list
    except FileNotFoundError:
        print("File not found. Please check the file path and name.")
    except ValueError:
        print("There was an issue converting a value to float. Check the file format.")

    return tile_values

def trim_list(tile_values, x, from_front=True):
    """
    Trims the list to only keep a specified number of elements from either the front or back.

    Parameters:
    tile_values (list): The original list of tile values.
    x (int): Number of elements to keep.
    from_front (bool): True to keep elements from the front, False to keep from the back.

    Returns:
    list: The trimmed list.
    """
    if from_front:
        return tile_values[:x]
    else:
        return tile_values[-x:]

def create_sublists(tile_values, counts):
    """
    Creates sublists from the list of tile values based on a list of counts.

    Parameters:
    tile_values (list): The list of tile values to split into sublists.
    counts (list): A list of integers where each integer specifies the number of elements in each sublist.

    Returns:
    list: A list of sublists, each containing the specified number of elements.
    """
    sublists = []
    start = 0
    for count in counts:
        if start + count > len(tile_values):  # Check if there are enough elements left
            break
        sublists.append(tile_values[start:start + count])
        start += count

    return sublists
def extract_tile_values(filename):
    tile_values = []  # List to hold the values of each tile
    tile_pattern = re.compile(r'tile\d+')  # Regular expression to find 'tile' followed by any number

    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()  # Split each line into parts based on whitespace
                if len(parts) == 2 and tile_pattern.search(parts[0]):  # Check if 'tile{i}' is in the first part
                    value = float(parts[1])  # Convert the second part (value) to float
                    tile_values.append(value)  # Append the value to the list
    except FileNotFoundError:
        print("File not found. Please check the file path and name.")
    except ValueError:
        print("There was an issue converting a value to float. Check the file format.")
    
    return tile_values

model_names = ['resnet18', 'densenet40',  'vgg8']
dataset_name = 'cifar10'
save_path = "./trace_data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# extract traces for all models
for model_name in model_names:
    if model_name == 'resnet18':
        model = ResNet18(num_classes=10)
        model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
    elif model_name == 'vgg16':
        model = VGG('VGG16', num_classes=10)
        model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
    elif model_name == 'densenet121':
        model = densenet121(num_classes=10)
        model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
    
    model.to(device)
    get_input_traces(dataset_name, model_name, model, save_path, device)

attack_names = ['input_based_attack'] 

configs  = ['20_80', '50_50', '80_20', 'special']


# #normal_mode for attacker tiles all zero power
max_power_array = 8e-3
peripheral_power = 2.71e-3
# extract tile info
for model_name in model_names:
    if model_name == 'resnet18':
        model = ResNet18(num_classes=10)
        model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
    elif model_name == 'vgg16':
        model = VGG('VGG16', num_classes=10)
        model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
    elif model_name == 'densenet121':
        model = densenet121(num_classes=10)
        model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
    
    # model.to(device)

    tile_info_and_weighted_sum_gen(model, model_name, dataset_name, max_power_array)

    
# ##attacking mode for attacker tiles power of tile is same as max power
original_directory = os.getcwd()
for mode in ['attacker', 'normal']:
    for model_name in model_names:
        # gen_floor_plan_2d_and_power_trace(model_name, dataset_name, max_power_array, peripheral_power)
        for attack_name in attack_names:
            for config in configs:
                print(f'Running HotSpot for {attack_name} attack on {model_name} model with {config} config')
                path = original_directory + f'/results_bs2/{attack_name}/{model_name}/{config}'
                ptrace_path = path + f'/power_trace_{model_name}_{dataset_name}_{mode}.ptrace'
                floorplan_file = path + f'/reramcore.flp'
                
                hotspot_executable_path = original_directory+'/../hotspot/examples/security'
                args1 = '.'
                args2=path
                args3=mode

                ##cp trace and floorplan to hotspot directory
                hotspot_trace_path = f'{hotspot_executable_path}/power.ptrace'
                hotspot_flp_path = f'{hotspot_executable_path}/reramcore.flp'

                # Copy the power trace file and floorplan file to the HotSpot directory
                shutil.copy(ptrace_path, hotspot_trace_path)
                shutil.copy(floorplan_file, hotspot_flp_path)

                hotspot_executable_file_path = hotspot_executable_path + '/main_exec.sh'

                output_path = hotspot_executable_path + '/command_output.txt'
                ##

# Open a file to redirect the output
                # new_directory = '/path/to/new/directory'
                os.chdir(hotspot_executable_path)
                with open(output_path, 'w') as f:
   
                    ##run hotspot
                    subprocess.run([hotspot_executable_file_path, args3],stdout=f, text=True)
                    output_folder = hotspot_executable_path + '/outputs'
                    output_results_folder_in_path = path + f'/hotspot_output_{args3}'
                        # If the destination exists, remove it first
                    if os.path.exists(output_results_folder_in_path):
                        shutil.rmtree(output_results_folder_in_path)
                    shutil.copytree(output_folder, output_results_folder_in_path)
                os.chdir(original_directory)
                shutil.copy(output_path, path + f'/hotspot_output_{mode}.txt')

        
for mode in ['attacker', 'normal']:
    for model_name in model_names:
        load_tile_info = torch.load(f"./trace_data/tile_info_{model_name}_{dataset_name}.pt")
    
        ## get the total number of tiles for the model
        for i in range(len(load_tile_info)):
            if i == 0:
                total_tile = load_tile_info[i]
            else:
                total_tile += load_tile_info[i]
        for attack_name in attack_names:
            for config in configs:
                path = original_directory + f'/results_bs2/{attack_name}/{model_name}/{config}'
                output_results_folder_in_path = path + f'/hotspot_output_{mode}'
                
                ## read the example.steady files
                steady_file_path = output_results_folder_in_path + '/example.steady'
                steady_values = extract_tile_values(steady_file_path)

                if config == 'special':
                    get_indx_list = torch.load(path+f'/interior_indices.pt')
                    trimmed_list = extract_selected_tile_values(steady_file_path, get_indx_list)
                    sublists = create_sublists(trimmed_list, load_tile_info)
                else:
                    trimmed_list = trim_list(steady_values, total_tile, from_front=True)
                    sublists = create_sublists(trimmed_list, load_tile_info)
                # if mode=='normal':
                    # print(sublists)
                
                torch.save(sublists, path + f'/new_tile_temp_{mode}.pt')


## Run the accuracy evaluation now accuracy framework is in /data/abommana/research_work/hw_security_CIM/inference/benchmark
'''
the main file: main_temp.py

Parsing arguments:
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

'''
accuracy_path = original_directory + '/../topk_inference/benchmark'
os.chdir(accuracy_path)
for mode in ['attacker', 'normal']:
    for model_name in model_names:
        for attack_name in attack_names:
            for config in configs:
                print(f'Running accuracy evaluation for {attack_name} attack on {model_name} model with {config} config in {mode} mode')
                command = [
                    'python', 'main_acc.py',
                    '--model', model_name,
                    '--batch_size', '128',
                    '--trained_model_path', 'trained_models',
                    '--dataset', dataset_name,
                    '--cuda', 'cuda:1',
                    '--fault_dist', 'uniform',
                    '--path', f'{original_directory}/results_bs2/{attack_name}/{model_name}/{config}/new_tile_temp_{mode}.pt'
                ]
                output_file_path = f"{original_directory}/results_bs2/{attack_name}/{model_name}/{config}/accuracy_output_{mode}.txt"
                with open(output_file_path, 'w') as file:
                    try:
                        # Run command and capture output
                        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
                        file.write(output.decode())  # Write the decoded output to file
                        print(f"Output written to {output_file_path}")
                    except subprocess.CalledProcessError as e:
                        # Write error to file and print it
                        error_message = e.output.decode()
                        file.write(f"Command failed with return code {e.returncode}\n{error_message}")
                        print(f"Command failed with return code {e.returncode}")
                        print(error_message)


os.chdir(original_directory)


