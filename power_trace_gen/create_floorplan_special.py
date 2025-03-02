from floorplan_gen import *
import os 
import torch
import math
from extract_tile_info import *


def gen_attacker_power_info(input_list, weight_list, model_name, dataset_name, attack_type, max_power, path,mode):
    crxb_size = 128
    tile_size = 96 ## ISSAC config has 96 arrays in one tile
    max_weighted_sum_const = crxb_size*8*3*crxb_size
    col_size = int(crxb_size/(quantization_res/2))
    power_trace = []
    n_lvl_w = 2 ** quantization_res-1
    h_lvl_w = (n_lvl_w - 2) / 2
    # print(avg_weighted_sum)
    # print(max_weighted_sum_const)
    on_off_ratio = 17
    K = torch.tensor([1/on_off_ratio,(1/3+2/(3*on_off_ratio)),(2/3+1/(3*on_off_ratio)),1])
    if attack_type == "input_based_attack":
        for i in range(len(input_list)):
            ##get the power info for the input-based attack
            crxb_row, crxb_row_pads = num_pad(weight_list[i].shape[1], crxb_size)
            crxb_col, crxb_col_pads = num_pad_col(weight_list[i].shape[0], crxb_size)
            w_pad = (0,  crxb_col_pads, 0,crxb_row_pads)
            input_pad = (0, 0, 0, crxb_row_pads)
            delta_w = weight_list[i].abs().max() / h_lvl_w
            weight_quan = quantize_dac(weight_list[i], n_lvl_w)
            weight_flatten = weight_quan.transpose(0, 1)
            # 2.2. add paddings
            weight_padded = F.pad(weight_flatten, w_pad,  mode='constant', value=0)
            weight_crxb = weight_padded.reshape(crxb_row,  crxb_col, crxb_size, col_size)
            mapped_weights = bit_slice_tensor(weight_crxb, quantization_res, 2)

            ## now find the weighted sum
            input_layer = input_list[i]
            delta_x = input_layer.abs().max() / h_lvl_w
            input_padded = F.pad(input_layer, input_pad,mode='constant', value=0)
            input_crxb = input_padded.view(input_layer.shape[0], 1, crxb_row, crxb_size, 1)
            ## change dtype to int16
            input_crxb = input_crxb.to(torch.int16)
            count_ones = count_bits(input_crxb)
            weight_sum = weight_sum_along_rows(mapped_weights, K)
            # print("weight_sum",weight_sum)
            A_squeezed = weight_sum.squeeze(-1)  # Shape becomes (2, 3, 4)
            B_squeezed = count_ones.squeeze(0).squeeze(0)  # Shape becomes (2, 4, 3)
            weighted_sum = torch.matmul(A_squeezed.float(), B_squeezed.float())
            
           
            ##do avg across last dimension 
            avg_weighted_sum = torch.mean(weighted_sum, dim=-1)
            ratio = avg_weighted_sum / max_weighted_sum_const
            # print("avg_weighted_sum for input", torch.sum(avg_weighted_sum))
            power = ratio * max_power
            # print("power",power)    
            power_list = power.flatten().tolist()
            power_trace.append(power_list)
            # print("power",power_list)
            # print("power len",len(power_list))
        torch.save(power_trace, path+f"/power_trace_attck_{model_name}_{dataset_name}_{mode}.pt")
    elif attack_type == "weight_based_attack":
        for i in range(len(input_list)):
            ##get the power info for the input-based attack
            crxb_row, crxb_row_pads = num_pad(weight_list[i].shape[1], crxb_size)
            crxb_col, crxb_col_pads = num_pad_col(weight_list[i].shape[0], crxb_size)
            w_pad = (0,  crxb_col_pads, 0,crxb_row_pads)
            input_pad = (0, 0, 0, crxb_row_pads)
            # weight_quan = quantize_dac(weight_list[i], n_lvl_w)
            weight_flatten = weight_list[i].transpose(0, 1)
            # 2.2. add paddings
            weight_padded = F.pad(weight_flatten, w_pad,  mode='constant', value=0)
            weight_crxb = weight_padded.reshape(crxb_row,  crxb_col, crxb_size, col_size)
            mapped_weights = bit_slice_tensor(weight_crxb, quantization_res, 2)

            ## now find the weighted sum
            input_layer = input_list[i]
            input_quan = quantize_dac(input_layer, n_lvl_w)
            input_padded = F.pad(input_quan, input_pad,mode='constant', value=0)
            input_crxb = input_padded.view(input_layer.shape[0], 1, crxb_row, crxb_size, 1)
            ## change dtype to int16
            input_crxb = input_crxb.to(torch.int16)
            count_ones = count_bits(input_crxb)
            weight_sum = weight_sum_along_rows(mapped_weights, K)
            A_squeezed = weight_sum.squeeze(-1)  # Shape becomes (2, 3, 4)
            B_squeezed = count_ones.squeeze(0).squeeze(0)  # Shape becomes (2, 4, 3)
            weighted_sum = torch.matmul(A_squeezed.float(), B_squeezed.float())
            ##do avg across last dimension 
            avg_weighted_sum = torch.mean(weighted_sum, dim=-1)
            # print(avg_weighted_sum)
            # print(max_weighted_sum_const)
            # print("avg_weighted_sum for weight", torch.sum(avg_weighted_sum))
            ratio = avg_weighted_sum / max_weighted_sum_const
            power = ratio * max_power
            # print("power",power)    
            power_list = power.flatten().tolist()
            power_trace.append(power_list)
            # print("power",power_list)
            # print("power len",len(power_list))
        torch.save(power_trace, path+f"/power_trace_attck_{model_name}_{dataset_name}_{mode}.pt")
    elif attack_type == "weight_input_based_attack":
        for i in range(len(input_list)):
            # print("Iam here in weight_input_based_attack")
            ##get the power info for the input-based attack
            crxb_row, crxb_row_pads = num_pad(weight_list[i].shape[1], crxb_size)
            crxb_col, crxb_col_pads = num_pad_col(weight_list[i].shape[0], crxb_size)
            w_pad = (0,  crxb_col_pads, 0,crxb_row_pads)
            input_pad = (0, 0, 0, crxb_row_pads)
            # weight_quan = quantize_dac(weight_list[i], n_lvl_w)
            weight_flatten = weight_list[i].transpose(0, 1)
            # 2.2. add paddings
            weight_padded = F.pad(weight_flatten, w_pad,  mode='constant', value=0)
            weight_crxb = weight_padded.reshape(crxb_row,  crxb_col, crxb_size, col_size)
            mapped_weights = bit_slice_tensor(weight_crxb, quantization_res, 2)

            ## now find the weighted sum
            input_layer = input_list[i]
            input_padded = F.pad(input_layer, input_pad,mode='constant', value=0)
            input_crxb = input_padded.view(input_layer.shape[0], 1, crxb_row, crxb_size, 1)
            ## change dtype to int16
            input_crxb = input_crxb.to(torch.int16)
            count_ones = count_bits(input_crxb)
            weight_sum = weight_sum_along_rows(mapped_weights, K)
            A_squeezed = weight_sum.squeeze(-1)  # Shape becomes (2, 3, 4)
            B_squeezed = count_ones.squeeze(0).squeeze(0)  # Shape becomes (2, 4, 3)
            weighted_sum = torch.matmul(A_squeezed.float(), B_squeezed.float())
            ##do avg across last dimension 
            avg_weighted_sum = torch.mean(weighted_sum, dim=-1)
            # print("avg_weighted_sum for weight_input", torch.sum(avg_weighted_sum))
            # print(avg_weighted_sum)
            # print(max_weighted_sum_const)
            ratio = avg_weighted_sum / max_weighted_sum_const
            power = ratio * max_power
            # print("power",power)    
            power_list = power.flatten().tolist()
            power_trace.append(power_list)
            # print("power",power_list)
            # print("power len",len(power_list))
        torch.save(power_trace, path+f"/power_trace_attck_{model_name}_{dataset_name}_{mode}.pt")

def combine_lists_with_specific_order(first_list, second_list, N_x, N_y, border_layers=1):
    # Validate that we can form the requested number of border layers
    # Each layer shrinks the available dimensions by 2 in both directions
    # Ensure that after border_layers, we still have a meaningful interior (or at least non-negative dimensions)
    if 2 * border_layers > N_x or 2 * border_layers > N_y:
        raise ValueError("Too many border layers for the given dimensions.")

    # Calculate the total number of border tiles across all layers
    # For each layer k (0-based), the width = N_x - 2*k and height = N_y - 2*k.
    # Number of tiles in that layer:
    #   = 2 * width + 2 * (height - 2)
    #   = 2*(N_x - 2*k) + 2*(N_y - 2*k - 2)
    #   = 2*(N_x - 2*k) + 2*(N_y - 2*k) - 4
    #   = 2*(N_x + N_y - 4*k - 2) - 4
    # Check if N_x - 2*k and N_y - 2*k are >= 3 to form a proper border. If not, that layer won't be a proper ring.
    # We'll assume the user sets border_layers appropriately so a proper border ring is formed each time.

    total_border_tiles = 0
    layer_tile_counts = []
    for k in range(border_layers):
        layer_width = N_x - 2*k
        layer_height = N_y - 2*k

        # If layer_width < 1 or layer_height < 1, it's invalid, but we checked earlier.
        # A single row or column could still be considered a 'border', but let's assume standard ring shape:
        if layer_width < 3 or layer_height < 3:
            raise ValueError("Layer {} cannot form a proper border ring with dimensions {}x{}."
                             .format(k, layer_width, layer_height))

        border_count_for_layer = 2 * layer_width + 2 * (layer_height - 2)
        layer_tile_counts.append(border_count_for_layer)
        total_border_tiles += border_count_for_layer

    # Split second_list into the border elements for each layer and the remaining (interior) elements
    if total_border_tiles > len(second_list):
        raise ValueError("Not enough elements in second_list to fill all border layers.")

    # Partition second_list for each border layer
    border_elements_per_layer = []
    start_idx = 0
    for count in layer_tile_counts:
        end_idx = start_idx + count
        border_elements_per_layer.append(second_list[start_idx:end_idx])
        start_idx = end_idx

    # The remaining elements of second_list (if any) go to the interior
    remaining_elements = second_list[start_idx:]

    # Append the remaining elements to the end of first_list before filling the interior
    first_list += remaining_elements

    # Identify border indices for each layer
    # For layer k:
    #   top row:    row = k, cols = k ... (N_x-k-1)
    #   bottom row: row = N_y-k-1, cols = k ... (N_x-k-1)
    #   left col:   col = k, rows = k+1 ... (N_y-k-2)
    #   right col:  col = N_x-k-1, rows = k+1 ... (N_y-k-2)
    all_border_indices = set()
    layer_border_indices = []

    for k in range(border_layers):
        layer_indices = set()

        top_row = k
        bottom_row = N_y - k - 1
        left_col = k
        right_col = N_x - k - 1

        # Top row indices
        for col in range(left_col, right_col + 1):
            layer_indices.add(top_row * N_x + col)

        # Bottom row indices
        if bottom_row != top_row:
            for col in range(left_col, right_col + 1):
                layer_indices.add(bottom_row * N_x + col)

        # Left column indices (excluding corners)
        for row in range(top_row + 1, bottom_row):
            layer_indices.add(row * N_x + left_col)

        # Right column indices (excluding corners)
        if right_col != left_col:
            for row in range(top_row + 1, bottom_row):
                layer_indices.add(row * N_x + right_col)

        layer_border_indices.append(sorted(layer_indices))
        all_border_indices.update(layer_indices)

    # Identify interior indices (those not in any border)
    interior_indices = [i for i in range(N_x * N_y) if i not in all_border_indices]

    # Assign border tiles from border_elements to their corresponding indices, layer by layer
    border_tiles = {}
    for k, indices in enumerate(layer_border_indices):
        for i, idx in enumerate(indices):
            border_tiles[idx] = border_elements_per_layer[k][i]

    # Assign interior tiles from first_list (already extended) to their corresponding indices
    interior_tiles = {}
    for i, idx in enumerate(interior_indices):
        interior_tiles[idx] = first_list[i]

    # Combine the tiles into one list ordered by indices
    combined_tiles = {**border_tiles, **interior_tiles}
    combined_list = [combined_tiles[i] for i in range(N_x * N_y)]

    return combined_list, interior_indices


def gen_floor_plan_2d_and_power_trace(model_name, dataset_name, max_power_array,peripheral_power, mode):

    #First create a directory for three types of attacks (input-based, weight-based, and weight-input-based), next create a subdirectory for the model. 
    # Next create a subdirectory for four different configurations of the floorplan for 2d (50_50, 20_80, 80_20, special). The traces and floor plan will be saved in the corresponding subdirectory.
    original_directory = os.getcwd()
    attack_type = ["input_based_attack", "weight_based_attack","weight_input_based_attack"]
    variable_configs = [1, 2, 3, 4, 5, 6, 7, 9]
    # variable_percentage = [variable_configs[i]*100 for i in range(len(variable_configs))]

    tile_configs = ["special"] ##variable or special

    load_tile_info = torch.load(f"./trace_data/tile_info_{model_name}_{dataset_name}.pt")
    

    ## get the total number of tiles for the model
    for i in range(len(load_tile_info)):
        if i == 0:
            user_tiles = load_tile_info[i]
        else:
            user_tiles += load_tile_info[i]
    
    
    
    unit_name_tile = "tile"
    tx = 0.0006 ##taken from ISAAC PAPER.
    ty = 0.0006
    load_power_info = torch.load(original_directory+f"/trace_data/power_trace_{model_name}_{dataset_name}.pt")
    user_power_trace = []
    for i in range(len(load_power_info)):
        if len(load_power_info[i]) > 96:
            ## divide into chunks of 96 and sum them up
            chunks = [load_power_info[i][j:j+96] for j in range(0, len(load_power_info[i]), 96)]
            for k in range(len(chunks)):
                user_power_trace.append(sum(chunks[k]) + len(chunks[k])*peripheral_power)
        else:
            user_power_trace.append(sum(load_power_info[i]) + len(load_power_info[i])*peripheral_power)
    if mode == 'attacker':
            max_power_array = 8e-3
            peripheral_power = 2.71e-3
    else:
            max_power_array = 0.00
            peripheral_power = 0
    
    for attack in attack_type:
        for tile_config in variable_configs:
            attack_path = original_directory+f"/results_spe/{attack}"
            os.makedirs(attack_path, exist_ok=True)
            model_path = f"{attack_path}/{model_name}"
            os.makedirs(model_path, exist_ok=True)
            tile_config_path = f"{model_path}/{tile_config*100}_{(1-tile_config)*100}"
            
            os.makedirs(tile_config_path, exist_ok=True)
            print(f"Generating floorplan and power trace for {model_name} with {attack} attack and {tile_config} tile configuration")
            if tile_configs[0] == "special":
                Nx_user = math.ceil(math.sqrt((user_tiles))) 
                Ny_user = math.ceil(user_tiles/Nx_user)
                Nx = Nx_user+ tile_config*2
                Ny = Ny_user + tile_config*2
                total_tiles = Nx*Ny
                # dummy_tiles = Nx*Ny - user_tiles
                tiles_for_attacker = total_tiles - user_tiles 
                ##check if total tiles for attacker + user tiles is equal to the total tiles raise error if they are not equal
                if tiles_for_attacker + user_tiles != total_tiles:
                    raise ValueError("The total tiles for the attacker and the user do not match the total tiles")

            else:
                total_tiles = math.ceil(user_tiles/tile_config)
                Nx = math.ceil(math.sqrt((total_tiles)))
                Ny = math.ceil(total_tiles/Nx)
                dummy_tiles = Nx*Ny - total_tiles
                tiles_for_attacker = total_tiles - user_tiles + dummy_tiles
            generate_2d_floorplan(Nx, Ny, tx, ty, unit_name_tile, tile_config_path)
            gen_input_weights_traces(attack, model_name, dataset_name, tiles_for_attacker, max_power_array, tile_config_path,mode)

            ##get the final power of each tile for both the attacker and the model.
            load_attack_power_info = torch.load(f"{tile_config_path}/power_trace_attck_{model_name}_{dataset_name}_{mode}.pt")
            attac_power_trace = []
            for i in range(len(load_attack_power_info)):
                attac_power_trace.append(sum(load_attack_power_info[i]) + len(load_attack_power_info[i])*peripheral_power)
            ##check if tiles for attacker and length of the power trace are equal
            if len(attac_power_trace) != tiles_for_attacker:
                raise ValueError("The number of tiles for the attacker and the length of the power trace do not match")
            
            ## apend two list till the special case
            if tile_configs[0] == 'special':
                copy_user_power_trace = user_power_trace.copy()
                combined_trace, interior_indices = combine_lists_with_specific_order(copy_user_power_trace, attac_power_trace, Nx, Ny, border_layers=tile_config)
                interior_indices = torch.tensor(interior_indices)
                torch.save(interior_indices, f"{tile_config_path}/interior_indices.pt")
            else:
                combined_trace = user_power_trace + attac_power_trace
            torch.save(combined_trace, f"{tile_config_path}/combined_power_trace_{model_name}_{dataset_name}_{mode}.pt")
            output_trace_file= f"{tile_config_path}/power_trace_{model_name}_{dataset_name}_{mode}.ptrace"
            create_ptrace_file(combined_trace, output_trace_file)


def gen_input_weights_traces(attack_type, model_name, dataset_name, tiles_for_attacker, max_power, path, mode):

    #assuming 8-bit weights and 2bit per cell in each tile you will have 8x12 arrays that why the layer size is 1024x384

    if attack_type == "input_based_attack":
        input_list = []
        weight_list = []
        for i in range(int(tiles_for_attacker)):
            ##set all inputs to 255 (8-bit) all linear layers which fit in the tile. linear layer (1024,384) so the input size is (1,1024)
            input_list.append(torch.full((1,1,1024),255))
            ##weights can be random
            weight_list.append(torch.rand((384,1024)))
        gen_attacker_power_info(input_list, weight_list, model_name, dataset_name, attack_type, max_power, path,mode)
    elif attack_type == "weight_based_attack":
        input_list = []
        weight_list = []
        ##here inputcan random whereas weights are set to 255
        for i in range(int(tiles_for_attacker)):
            input_list.append(torch.rand((1,1,1024)))
            weight_list.append(torch.full((384,1024),255))
        gen_attacker_power_info(input_list, weight_list, model_name, dataset_name, attack_type, max_power, path,mode)
    elif attack_type == "weight_input_based_attack":
        input_list = []
        weight_list = []
        ##here both inputs and weights are set to 255
        for i in range(int(tiles_for_attacker)):
            input_list.append(torch.full((1,1,1024),255))
            weight_list.append(torch.full((384,1024),255))
        gen_attacker_power_info(input_list, weight_list, model_name, dataset_name, attack_type, max_power, path,mode)

    ##get the power info. 

def create_ptrace_file(list_values, filename='output.ptrace'):
    num_tiles = len(list_values)
    tile_names = [f"tile{i+1}" for i in range(num_tiles)]
    data_lines = ['\t'.join(map(str, list_values))] * 5  # Repeat the list values line 5 times
    
    with open(filename, 'w') as file:
        # Write tile names separated by tabs
        file.write('\t'.join(tile_names) + '\n')
        # Write the data lines
        file.write('\n'.join(data_lines) + '\n')
        # Add an empty line at the end
        # file.write('\n')



def main():
    max_power_array = 8e-3
    peripheral_power=2.7e-3
    model_names = ['resnet18','densenet40', 'vgg8']
    # model_names = ['resnet18']
    for mode in ['attacker', 'normal']:
        for model_name in model_names:
            gen_floor_plan_2d_and_power_trace(model_name, "cifar10", max_power_array,peripheral_power,mode )

if __name__ == "__main__":
    main()