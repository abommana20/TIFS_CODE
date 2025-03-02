from floorplan_gen import *
import os 
import torch
import math
from extract_tile_info import *
import re

def gen_attacker_power_info(input_list, weight_list, model_name, dataset_name, attack_type, max_power, path):
    crxb_size = 128
    tile_size = 96 ## ISSAC config has 96 arrays in one tile
    max_weighted_sum_const = crxb_size*8*3*crxb_size
    col_size = int(crxb_size/(quantization_res/2))
    power_trace = []
    n_lvl_w = 2 ** quantization_res-1
    h_lvl_w = (n_lvl_w - 2) / 2
    
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
        torch.save(power_trace, path+f"/power_trace_attck_{model_name}_{dataset_name}.pt")
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
            
            # print("avg_weighted_sum for weight", torch.sum(avg_weighted_sum))
            ratio = avg_weighted_sum / max_weighted_sum_const
            power = ratio * max_power
            # print("power",power)    
            power_list = power.flatten().tolist()
            power_trace.append(power_list)
            # print("power",power_list)
            # print("power len",len(power_list))
        torch.save(power_trace, path+f"/power_trace_attck_{model_name}_{dataset_name}.pt")
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
            ratio = avg_weighted_sum / max_weighted_sum_const
            power = ratio * max_power
            # print("power",power)    
            power_list = power.flatten().tolist()
            power_trace.append(power_list)
            # print("power",power_list)
            # print("power len",len(power_list))
        torch.save(power_trace, path+f"/power_trace_attck_{model_name}_{dataset_name}.pt")
def combine_lists_with_specific_order(first_list, second_list, N_x, N_y):
    # Calculate the number of border tiles
    num_border_tiles = 2 * N_x + 2 * (N_y - 2)

    # Split the second_list into border elements and remaining elements
    border_elements = second_list[:num_border_tiles]
    remaining_elements = second_list[num_border_tiles:]  # Extra elements beyond the border requirement
    
    # Append remaining elements of second_list to the end of first_list before filling the interior
    first_list += remaining_elements  # This appends the remaining elements to the end of first_list
    
    # Prepare to collect indices for border and interior
    border_indices = set()
    interior_indices = []

    # Top and bottom row indices
    border_indices.update(range(0, N_x))  # Top row
    border_indices.update(range((N_y - 1) * N_x, N_y * N_x))  # Bottom row
    
    # Left and right column indices, excluding already counted corners
    if N_y > 2:
        border_indices.update(range(N_x, (N_y - 1) * N_x, N_x))  # Left column
        border_indices.update(range(2 * N_x - 1, N_y * N_x - 1, N_x))  # Right column

    # Collect interior indices
    for i in range(N_x * N_y):
        if i not in border_indices:
            interior_indices.append(i)

    # Assign border tiles from border_elements to their corresponding indices
    border_tiles = {index: border_elements[i] for i, index in enumerate(sorted(border_indices))}
    interior_tiles = {index: first_list[i] for i, index in enumerate(interior_indices)}

    # Combine the tiles into one list ordered by indices
    combined_tiles = {**border_tiles, **interior_tiles}
    combined_list = [combined_tiles[i] for i in range(N_x * N_y)]
    
    return combined_list, interior_indices

def gen_floor_plan_3d_and_power_trace(model_name, dataset_name, max_power_array,peripheral_power,mode):

    #First create a directory for three types of attacks (input-based, weight-based, and weight-input-based), next create a subdirectory for the model. 
    # Next create a subdirectory for four different configurations of the floorplan for 2d (50_50, 20_80, 80_20, special). The traces and floor plan will be saved in the corresponding subdirectory.
    attack_type = ["input_based_attack"]
    tile_config = ["50_50", "20_80", "80_20"]
    flp_config = ["top_tim", "top_wo_tim", "bottom_tim", "bottom_wo_tim"]

    num_of_tiers = 2

    load_tile_info = torch.load(f"./trace_data/tile_info_{model_name}_{dataset_name}.pt")
    attacker_model_names = ['resnet18','vgg16','densenet121','densenet40', 'alexnet', 'lenet', 'vgg8']

    attacker_tile_info = {}
    for attacker_model in attacker_model_names:
        attacker_tile_info[attacker_model] = torch.load(f"./trace_data/tile_info_{attacker_model}_{dataset_name}.pt")
    


    ## get the total number of tiles for the model
    for i in range(len(load_tile_info)):
        if i == 0:
            total_tile = load_tile_info[i]
        else:
            total_tile += load_tile_info[i]
    
    ## get the total number of tiles for the attacker
    total_tile_attacker = {}
    for attacker_model in attacker_model_names:
        total_tile_attacker[attacker_model] = 0
        for i in range(len(attacker_tile_info[attacker_model])):
            if i == 0:
                total_tile_attacker[attacker_model] = attacker_tile_info[attacker_model][i]
            else:
                total_tile_attacker[attacker_model] += attacker_tile_info[attacker_model][i]
    sorted_total_tile_attacker = dict(sorted(total_tile_attacker.items(), key=lambda item: item[1], reverse=True))
    print("sorted_total_tile_attacker",sorted_total_tile_attacker)
    
    unit_name_tile = "tile"
    tx = 0.0006
    ty = 0.0006

    ##create folders for the attack
    for attack in attack_type:
        os.makedirs(f"./results_3d_bs2/{attack}", exist_ok=True)
        attack_path = f"./results_3d_bs2/{attack}"
        for tile in tile_config:
            os.makedirs(f"./results_3d_bs2/{attack}/{model_name}/{tile}", exist_ok=True)
            tile_path = attack_path + f"/{model_name}/{tile}"
            for flp in flp_config:
                os.makedirs(f"./results_3d_bs2/{attack}/{model_name}/{tile}/{flp}", exist_ok=True)
                flp_path = tile_path + f"/{flp}"
    dummy_tiles_left = {}
    attack_model_mapped = {}
    for mode in ["attacker", "normal"]:
        dummy_tiles_left[mode] = {}
        attack_model_mapped[mode] = {}          
        for attack in attack_type:
            dummy_tiles_left[mode][attack] = {}
            attack_model_mapped[mode][attack] = {}

            attack_path = f"./results_3d_bs2/{attack}"
            for tile in tile_config:
                dummy_tiles_left[mode][attack][tile] = {}
                attack_model_mapped[mode][attack][tile] = {}
                tile_path = attack_path + f"/{model_name}/{tile}"
                attack_model = []
                if tile == "50_50":
                    total_tile_estimated = 2*total_tile
                    tile_per_tier = math.ceil(total_tile_estimated/num_of_tiers)
                    Nx = math.ceil(math.sqrt((tile_per_tier))) 
                    Ny = math.ceil(tile_per_tier/Nx)
                    dummy_tiles = 2*Nx*Ny - total_tile_estimated ## All dummy tiles can be used by attacker so no need to explictly add them and have them in the floorplan.
                    tiles_for_attacker = total_tile_estimated - total_tile + dummy_tiles
                elif tile == "20_80":
                    total_tile_estimated = math.ceil(total_tile/0.2)
                    tile_per_tier =math.ceil(total_tile_estimated/num_of_tiers)
                    Nx = math.ceil(math.sqrt((tile_per_tier))) 
                    Ny = math.ceil(tile_per_tier/Nx)
                    dummy_tiles = 2*Nx*Ny - total_tile_estimated
                    tiles_for_attacker = total_tile_estimated - total_tile + dummy_tiles
                elif tile == "80_20":
                    total_tile_estimated = math.ceil(total_tile/0.8)
                    tile_per_tier = math.ceil(total_tile_estimated/num_of_tiers)
                    Nx = math.ceil(math.sqrt((tile_per_tier)))
                    Ny = math.ceil(tile_per_tier/Nx)
                    dummy_tiles = 2*Nx*Ny - total_tile_estimated
                    tiles_for_attacker = total_tile_estimated - total_tile + dummy_tiles
                for attacker_model in sorted_total_tile_attacker.keys():
                    if sorted_total_tile_attacker[attacker_model] <= tiles_for_attacker:
                        tiles_for_attacker -= total_tile_attacker[attacker_model]
                        attack_model.append(attacker_model)
                    else:
                        continue
                
                dummy_tiles_left[mode][attack][tile]= tiles_for_attacker
                attack_model_mapped[mode][attack][tile] = attack_model
                for flp in flp_config:
                    path = tile_path + f"/{flp}"
                    generate_3d_floorplan_separate_files(Nx, Ny, tx, ty, num_of_tiers, unit_name_tile, path)
                    # gen_input_weights_traces(attack, model_name, dataset_name, tiles_for_attacker, max_power_array, path)
 

  ##################################################################
    ##get the final power of each tile for both the attacker and the model.
    load_power_info = torch.load(f"./trace_data/power_trace_{model_name}_{dataset_name}.pt")
    user_power_trace = []
    for i in range(len(load_power_info)):
        if len(load_power_info[i]) > 96:
            ## divide into chunks of 96 and sum them up
            chunks = [load_power_info[i][j:j+96] for j in range(0, len(load_power_info[i]), 96)]
            for k in range(len(chunks)):
                user_power_trace.append(sum(chunks[k]) + len(chunks[k])*peripheral_power)
        else:
            user_power_trace.append(sum(load_power_info[i]) + len(load_power_info[i])*peripheral_power)    
    torch.save(user_power_trace, f"./results_3d_bs2/final_power_trace_{model_name}_{dataset_name}")
    for mode in ["attacker", "normal"]:
        for attack in attack_type:
            for tile in tile_config:
                if tile == "50_50":
                    total_tile_estimated = 2*total_tile
                    tile_per_tier = math.ceil(total_tile_estimated/num_of_tiers)
                    Nx = math.ceil(math.sqrt((tile_per_tier))) 
                    Ny = math.ceil(tile_per_tier/Nx)
                elif tile == "20_80":
                    total_tile_estimated = math.ceil(total_tile/0.2)
                    tile_per_tier = math.ceil(total_tile_estimated/num_of_tiers)
                    Nx = math.ceil(math.sqrt((tile_per_tier))) 
                    Ny = math.ceil(tile_per_tier/Nx)
                elif tile == "80_20":
                    total_tile_estimated = math.ceil(total_tile/0.8)
                    tile_per_tier = math.ceil(total_tile_estimated/num_of_tiers)
                    Nx = math.ceil(math.sqrt((tile_per_tier)))
                    Ny = math.ceil(tile_per_tier/Nx)
                attac_power_trace_1 = []
                for attacker_model in attack_model_mapped[mode][attack][tile]:
                    attack_power_trace = []
                    if mode == "attacker":
                        load_attack_power_info = torch.load(f"trace_data/power_trace_{attacker_model}_synthetic.pt")
                    else:
                        load_attack_power_info = torch.load(f"trace_data/power_trace_{attacker_model}_{dataset_name}.pt")
                    
                    for i in range(len(load_attack_power_info)):
                        # print("length of the power info",len(load_power_info[i]))
                        # print("length of the power info",len(load_attack_power_info_1[i]))
                        if len(load_attack_power_info[i]) > 96:
                            ## divide into chunks of 96 and sum them up
                            chunks = [load_attack_power_info[i][j:j+96] for j in range(0, len(load_attack_power_info[i]), 96)]
                            for k in range(len(chunks)):
                                attack_power_trace.append(sum(chunks[k]) + len(chunks[k])*peripheral_power)
                        else:
                            attack_power_trace.append(sum(load_attack_power_info[i]) + len(load_attack_power_info[i])*peripheral_power)
                    attac_power_trace_1 += attack_power_trace
                attac_power_trace_1 += [0]*dummy_tiles_left[mode][attack][tile]
                for flp in flp_config:
                    path = f"./results_3d_bs2/{attack}/{model_name}/{tile}/{flp}"
                    
        
                 
                    if flp == "top_tim": ## user tile is on top and attacker tile is on the bottom 
                        ##there is also TIM layer in the power trace one must be cautious where to insert it 
                        ## apend two list till the special case
                        combined_trace_1 = attac_power_trace_1 + user_power_trace 
                        TIM_enabled = True

                    elif flp == "top_wo_tim":

                        combined_trace_1 = attac_power_trace_1 + user_power_trace 
                        TIM_enabled = False

                    elif flp == "bottom_tim":
                        combined_trace_1 = user_power_trace + attac_power_trace_1
                        TIM_enabled = True

                    elif flp == "bottom_wo_tim":

                        combined_trace_1 = user_power_trace + attac_power_trace_1
                        TIM_enabled = False
                    
                    
                    torch.save(combined_trace_1, f"{path}/combined_power_trace_{model_name}_{dataset_name}_{mode}.pt")
                    
                    output_trace_file_1= f"{path}/power_trace_{model_name}_{dataset_name}_{mode}.ptrace"

                    create_ptrace_file(combined_trace_1,2,Nx*Ny,TIM_enabled,output_trace_file_1)
    




def gen_input_weights_traces(attack_type, model_name, dataset_name, tiles_for_attacker, max_power, path):

    #assuming 8-bit weights and 2bit per cell in each tile you will have 8x12 arrays that why the layer size is 1024x384

    if attack_type == "input_based_attack":
        input_list = []
        weight_list = []
        for i in range(int(tiles_for_attacker)):
            ##set all inputs to 255 (8-bit) all linear layers which fit in the tile. linear layer (1024,384) so the input size is (1,1024)
            input_list.append(torch.full((1,1,1024),255))
            ##weights can be random
            weight_list.append(torch.rand((384,1024)))
        gen_attacker_power_info(input_list, weight_list, model_name, dataset_name, attack_type, max_power, path)
    elif attack_type == "weight_based_attack":
        input_list = []
        weight_list = []
        ##here inputcan random whereas weights are set to 255
        for i in range(int(tiles_for_attacker)):
            input_list.append(torch.rand((1,1,1024)))
            weight_list.append(torch.full((384,1024),255))
        gen_attacker_power_info(input_list, weight_list, model_name, dataset_name, attack_type, max_power, path)
    elif attack_type == "weight_input_based_attack":
        input_list = []
        weight_list = []
        ##here both inputs and weights are set to 255
        for i in range(int(tiles_for_attacker)):
            input_list.append(torch.full((1,1,1024),255))
            weight_list.append(torch.full((384,1024),255))
        gen_attacker_power_info(input_list, weight_list, model_name, dataset_name, attack_type, max_power, path)

    ##get the power info. 
def create_ptrace_file(list_values, num_tiers, tiles_per_tier, tim_enabled=True, filename='output.ptrace'):
    total_tiles = num_tiers * tiles_per_tier
    tile_names = []

    # Construct tile names and insert TIM names if enabled
    for tier in range(num_tiers):
        for i in range(tiles_per_tier):
            tile_names.append(f"tile{tier * tiles_per_tier + i + 1}")
        # if tim_enabled:
            # Insert the TIM tile name after each tier
            # tile_names.append("TIM_1")

    # Prepare the list values by inserting 0 for TIM if enabled
    enhanced_list_values = []
    for i in range(num_tiers):
        enhanced_list_values.extend(list_values[i * tiles_per_tier:(i + 1) * tiles_per_tier])
        # if tim_enabled:
            # Insert a 0 power value for the TIM layer
            # enhanced_list_values.append(0)

    data_lines = ['\t'.join(map(str, enhanced_list_values))] * 5  # Repeat the list values line 5 times
    
    with open(filename, 'w') as file:
        # Write tile names separated by tabs
        file.write('\t'.join(tile_names) + '\n')
        # Write the data lines
        file.write('\n'.join(data_lines) + '\n')
        # Add an empty line at the end
        # file.write('\n')

    return f"Ptrace file '{filename}' created with TIM layer {'enabled' if tim_enabled else 'disabled'}."



def main():
    max_power_array = 8e-3
    peripheralpower=2.7e-3
    model_names = ['resnet18','densenet40', 'vgg8']

    for model_name in model_names:
        for mode in ['attacker', 'normal']:
            print("model_name",model_name)
            print("mode",mode)
            gen_floor_plan_3d_and_power_trace(model_name, "cifar10", max_power_array,peripheralpower,mode)

if __name__ == "__main__":
    main()