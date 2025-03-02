from extract_input_weight_traces import *
import torch
from resnet import *
from densenet import *
from VGG import *
from create_floor3d import *
from extract_tile_info import *
import subprocess
import shutil
import os

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
model_names = ['resnet18']
dataset_name = 'cifar10'
save_path = "./trace_data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## extract traces for all models
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

attack_type = ["input_based_attack"]
tile_configs = ["50_50"]
flp_configs = ["top_tim", "top_wo_tim", "bottom_tim", "bottom_wo_tim"]




# #normal_mode for attacker tiles all zero power
max_power_array = 8e-3
peripheral_power = 2.71e-3
## extract tile info
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
max_power_array = 8e-3
peripheral_power = 2.71e-3
original_directory = os.getcwd()
for mode in ['attacker', 'normal']:

    for model_name in model_names:
        # gen_floor_plan_3d_and_power_trace(model_name, dataset_name, max_power_array, peripheral_power,mode)
        for attack_name in attack_names:
            for tile in tile_configs:
                for flp_config in flp_configs:
                    print(f'Running HotSpot for {attack_name} attack on {model_name} model with {flp_config} flp config and {tile} tile config')
                    path = original_directory + f'/results_3d_bs2/{attack_name}/{model_name}/{tile}/{flp_config}'
                    ptrace_path = path + f'/power_trace_{model_name}_{dataset_name}_{mode}.ptrace'
                    floorplan_file1 = path + f'/reramcore1.flp'
                    floorplan_file2 = path + f'/TIM.flp'
                    floorplan_file3 = path + f'/reramcore2.flp'
                    
                    if 'wo' in flp_config:
                        hotspot_executable_path = original_directory + '/../hotspot/examples/security_3d_wout_tim'
                    else:
                        hotspot_executable_path = original_directory+'/../hotspot/examples/security_3d_w_tim'
                
                    args2=path
                    args3=mode

                    ##cp trace and floorplan to hotspot directory
                    hotspot_trace_path = f'{hotspot_executable_path}/power.ptrace'
                    hotspot_flp1_path = f'{hotspot_executable_path}/reramcore1.flp'
                    hotspot_flp2_path = f'{hotspot_executable_path}/TIM.flp'
                    hotspot_flp3_path = f'{hotspot_executable_path}/reramcore2.flp'

                    # Copy the power trace file and floorplan file to the HotSpot directory
                    shutil.copy(ptrace_path, hotspot_trace_path)
                    shutil.copy(floorplan_file1, hotspot_flp1_path)
                    shutil.copy(floorplan_file2, hotspot_flp2_path)
                    shutil.copy(floorplan_file3, hotspot_flp3_path)

                    hotspot_executable_file_path = hotspot_executable_path + '/run.sh'

                    output_path = hotspot_executable_path + '/command_output.txt'
                    ##

    # Open a file to redirect the output
                    # new_directory = '/path/to/new/directory'
                    os.chdir(hotspot_executable_path)
                    with open(output_path, 'w') as f:
    
                        ##run hotspot
                        subprocess.run([hotspot_executable_file_path, args2, args3],stdout=f, text=True)
                        output_folder = hotspot_executable_path + '/outputs'
                        output_results_folder_in_path = path + f'/hotspot_output_{args3}'
                         # If the destination exists, remove it first
                        if os.path.exists(output_results_folder_in_path):
                            shutil.rmtree(output_results_folder_in_path)
                        shutil.copytree(output_folder, output_results_folder_in_path)
                    os.chdir(original_directory)
                    shutil.copy(output_path, path + '/hotspot_command_output.txt')


after_attack_user_temp = []

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
            for tile in tile_configs:
                for flp_config in flp_configs:
                    path = original_directory + f'/results_3d_bs2/{attack_name}/{model_name}/{tile}/{flp_config}'
                    output_path = path +  f'/hotspot_output_{mode}'
                    file = output_path + '/example.steady'
                    ##read the stead state temp from hotspot output to get the new temperatures for the model (example.steady)
                    steady_state_temp = []
                    steady_state_temp = extract_tile_values(file)

                    if flp_config == "top_tim": 
                        # combined_trace_1 = attac_power_trace_1 + user_power_trace 
                        ##take the last tiles in example steady state temp
                        trmmied_list = trim_list(steady_state_temp, total_tile, from_front=False)
                        tile_temp = create_sublists(trmmied_list, load_tile_info)


                    elif flp_config == "top_wo_tim":

                        # combined_trace_1 = attac_power_trace_1 + user_power_trace 
                        ##take the last tiles in example steady state temp
                        trmmied_list = trim_list(steady_state_temp, total_tile, from_front=False)
                        tile_temp = create_sublists(trmmied_list, load_tile_info)

                    elif flp_config == "bottom_tim":
                        # combined_trace_1 = user_power_trace + attac_power_trace_1
                        #take the first tiles in example steady state temp
                        trmmied_list = trim_list(steady_state_temp, total_tile, from_front=True)
                        tile_temp = create_sublists(trmmied_list, load_tile_info)

                    elif flp_config == "bottom_wo_tim":

                        # combined_trace_1 = user_power_trace + attac_power_trace_1
                        #take the first tiles in example steady state temp
                        trmmied_list = trim_list(steady_state_temp, total_tile, from_front=True)
                        tile_temp = create_sublists(trmmied_list, load_tile_info)
                    
                    ##save the new temperatures for the model
                    torch.save(tile_temp, path + f'/new_tile_temp_{mode}.pt')
                        
                



# ## Run the accuracy evaluation now accuracy framework is in /data/abommana/research_work/hw_security_CIM/inference/benchmark
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
            for tile in tile_configs:
                for flp_config in flp_configs:
                    print(f'Running accuracy evaluation for {attack_name} attack on {model_name} model with {tile} tileconfig and {flp_config} flp config in {mode} mode')
                    command = [
                        'python', 'main_temp.py',
                        '--model', model_name,
                        '--batch_size', '128',
                        '--trained_model_path', 'trained_models',
                        '--dataset', dataset_name,
                        '--cuda', 'cuda:2',
                        '--fault_dist', 'uniform',
                        '--path', f'{original_directory}/results_3d_bs2/{attack_name}/{model_name}/{tile}/{flp_config}/new_tile_temp_{mode}.pt'
                    ]
                    output_file_path = f"{original_directory}/results_3d_bs2/{attack_name}/{model_name}/{tile}/{flp_config}/accuracy_output_{mode}.txt"
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