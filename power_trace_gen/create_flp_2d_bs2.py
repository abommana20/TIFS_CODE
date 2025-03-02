from floorplan_gen import *
import os 
import torch
import math
from extract_tile_info import *


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
        for model in input_list.keys():
            model_input_list = input_list[model]
            model_weight_list = weight_list[model]    
            for i in range(len(model_input_list)):
                ##get the power info for the input-based attack
                
                
                if len(model_weight_list[i].shape) > 2:
                    crxb_row, crxb_row_pads = num_pad(model_weight_list[i].shape[1]*model_weight_list[i].shape[2]*model_weight_list[i].shape[3], crxb_size)
                    crxb_col, crxb_col_pads = num_pad_col(model_weight_list[i].shape[0], crxb_size)
                    
                    input_pad = (0, 0, 0, crxb_row_pads)
                else:
                    input_pad = (0, crxb_row_pads)
                w_pad = (0,  crxb_col_pads, 0,crxb_row_pads)
                delta_w = model_weight_list[i].abs().max() / h_lvl_w
                weight_quan = quantize_dac(model_weight_list[i], n_lvl_w)
                weight_flatten = weight_quan.transpose(0, 1)
                # 2.2. add paddings
                weight_padded = F.pad(weight_flatten, w_pad,  mode='constant', value=0)
                weight_crxb = weight_padded.reshape(crxb_row,  crxb_col, crxb_size, col_size)
                mapped_weights = bit_slice_tensor(weight_crxb, quantization_res, 2)

                ## now find the weighted sum
                input_layer = model_input_list[i]
                delta_x = input_layer.abs().max() / h_lvl_w
                if len(model_weight_list[i].shape) > 2:
                    input_unfold = F.unfold(input_layer, kernel_size=model_weight_list[i].shape[2],
                                dilation=1, padding=(1,1),stride=(1,1))
                    input_padded = F.pad(input_layer, input_pad,mode='constant', value=0)
                    input_crxb = input_padded.view(input.shape[0], 1, crxb_row, crxb_size, input_padded.shape[2])
                    sum_input_crxb =  torch.sum(input_crxb, dim=-1)/input_padded.shape[2]
                else: 
                    input_padded = F.pad(input_layer, input_pad,mode='constant', value=0)
                    input_crxb = input_padded.view(input_layer.shape[0], 1, crxb_row, crxb_size, 1)
                    sum_input_crxb = input_crxb


                
                # ## Need to avg the inputs across the sliding windows in case of conv layer
                # input_crxb = input_padded.view(input_layer.shape[0], 1, crxb_row, crxb_size, 1)
                ## change dtype to int16
                sum_input_crxb = sum_input_crxb.to(torch.int32)
                count_ones = count_bits(sum_input_crxb)
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
            torch.save(power_trace, path+f"/power_trace_attack_{model}_synthetic_bs2.pt")
    
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

def gen_floor_plan_2d_and_power_trace(model_name, dataset_name, max_power_array,peripheral_power, mode):

    #First create a directory for three types of attacks (input-based, weight-based, and weight-input-based), next create a subdirectory for the model. 
    # Next create a subdirectory for four different configurations of the floorplan for 2d (50_50, 20_80, 80_20, special). The traces and floor plan will be saved in the corresponding subdirectory.

    attack_path_1 = f"./results_bs2/input_based_attack"
    # attack_path_2 = f"./results/weight_based_attack"
    # attack_path_3 = f"./results/weight_input_based_attack"

    ##create folders for the attack
    os.makedirs(attack_path_1, exist_ok=True)

    model_path_1 = f"{attack_path_1}/{model_name}"

    ##create folders for the model
    os.makedirs(model_path_1, exist_ok=True)

    ##create folders for the floorplan for different configs
    floorplan_path_1_attack_1 = f"{model_path_1}/50_50"
    floorplan_path_2_attack_1 = f"{model_path_1}/20_80"
    floorplan_path_3_attack_1 = f"{model_path_1}/80_20"
    floorplan_path_4_attack_1 = f"{model_path_1}/special"

    os.makedirs(floorplan_path_1_attack_1, exist_ok=True)
    os.makedirs(floorplan_path_2_attack_1, exist_ok=True)
    os.makedirs(floorplan_path_3_attack_1, exist_ok=True)
    os.makedirs(floorplan_path_4_attack_1, exist_ok=True)

    
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
 
    
    unit_name_tile = "tile"
    tx = 0.0006
    ty = 0.0006

    ## For 50_50 configuration
    total_tile_50_50 = 2*total_tile
    Nx_50_50 = math.ceil(math.sqrt((total_tile_50_50))) 
    Ny_50_50 = math.ceil(total_tile_50_50/Nx_50_50)
    # dummy_tiles_50_50 = Nx_50_50*Ny_50_50 - total_tile_50_50 ## All dummy tiles can be used by attacker so no need to explictly add them and have them in the floorplan.
    tiles_for_attacker_50_50 = Nx_50_50*Ny_50_50 - total_tile 

    tiles_left_after_attact_50_50 = tiles_for_attacker_50_50 - total_tile
    attack_model_mapped_50_50 = [model_name]
    print("----------------------------------------------------------------------------------------------")
    ##check if you can seqqueze another model in the same floorplan
    for attacker_model in sorted_total_tile_attacker.keys():
        if sorted_total_tile_attacker[attacker_model] <= tiles_left_after_attact_50_50:
            tiles_left_after_attact_50_50 -= total_tile_attacker[attacker_model]
            attack_model_mapped_50_50.append(attacker_model)
        else:
            continue
            # print("attacker_model",attacker_model)
            # print("tiles_left_after_attact",tiles_left_after_attact_50_50)

    dummy_tiles = tiles_left_after_attact_50_50
    generate_2d_floorplan(Nx_50_50, Ny_50_50, tx, ty, unit_name_tile, floorplan_path_1_attack_1)

    # gen_input_weights_traces("input_based_attack", attack_model_mapped_50_50, dataset_name, tiles_for_attacker_50_50, max_power_array, floorplan_path_1_attack_1)

    print("----------------------------------------------------------------------------------------------")
    ## For 20_80 configuration
    total_tile_20_80 = math.ceil(total_tile/0.20)
    Nx_20_80 = math.ceil(math.sqrt((total_tile_20_80)))
    Ny_20_80 = math.ceil(total_tile_20_80/Nx_20_80)
    # dummy_tiles_20_80 = Nx_20_80*Ny_20_80 - total_tile_20_80
    tiles_left_attacker_20_80 = Nx_20_80*Ny_20_80 - total_tile 

    attack_model_mapped_20_80 = []
    for attacker_model in sorted_total_tile_attacker.keys():
        if sorted_total_tile_attacker[attacker_model] <= tiles_left_attacker_20_80:
            tiles_left_attacker_20_80 -= total_tile_attacker[attacker_model]
            attack_model_mapped_20_80.append(attacker_model)
        else:
           continue
    generate_2d_floorplan(Nx_20_80, Ny_20_80, tx, ty, unit_name_tile, floorplan_path_2_attack_1)
    # gen_input_weights_traces("input_based_attack", attack_model_mapped_20_80, dataset_name, tiles_for_attacker_20_80, max_power_array, floorplan_path_2_attack_1)

    dummy_tiles_20_80 = tiles_left_attacker_20_80
    print("----------------------------------------------------------------------------------------------")
    ## For 80_20 configuration
    total_tile_80_20 = math.ceil(total_tile/0.8)
    Nx_80_20 = math.ceil(math.sqrt((total_tile_80_20)))
    Ny_80_20 = math.ceil(total_tile_80_20/Nx_80_20)
    tiles_left_for_attacker_80_20 = Nx_80_20*Ny_80_20 - total_tile 

    attack_model_mapped_80_20 = []
    for attacker_model in sorted_total_tile_attacker:
        if sorted_total_tile_attacker[attacker_model] <= tiles_left_for_attacker_80_20:
            tiles_left_for_attacker_80_20 -= total_tile_attacker[attacker_model]
            attack_model_mapped_80_20.append(attacker_model)
           
        else:
            continue
    generate_2d_floorplan(Nx_80_20, Ny_80_20, tx, ty, unit_name_tile, floorplan_path_3_attack_1)
    
    # gen_input_weights_traces("input_based_attack", attack_model_mapped_80_20, dataset_name, tiles_for_attacker_80_20, max_power_array, floorplan_path_3_attack_1)
    dummy_tiles_80_20 = tiles_left_for_attacker_80_20
    print("----------------------------------------------------------------------------------------------")
    ## For special configuration
    Nx_base = math.ceil(math.sqrt((total_tile))) 
    Ny_base = math.ceil(total_tile/Nx_base)
    total_tile_special = (Nx_base+2)*(Ny_base+2)
    Nx_special = Nx_base+2##columns
    Ny_special = Ny_base+2 ##rows
    # dummy_tiles_special =  - total_tile_special 
    tiles_left_for_attacker_special = Nx_special*Ny_special - total_tile 

    attack_model_special = []
    for attacker_model in sorted_total_tile_attacker:
        if sorted_total_tile_attacker[attacker_model] <= tiles_left_for_attacker_special:
            tiles_left_for_attacker_special -= total_tile_attacker[attacker_model]
            attack_model_special.append(attacker_model)
        else:
            continue

    
    ##reamining tiles are dummy just zero power
    generate_2d_floorplan(Nx_special, Ny_special, tx, ty, unit_name_tile, floorplan_path_4_attack_1)

    # gen_input_weights_traces("input_based_attack", model_name, dataset_name, tiles_for_attacker_special, max_power_array, floorplan_path_4_attack_1)
    dummy_tiles_special = tiles_left_for_attacker_special
    print("---------------------------------------Floorplan Generation done----------------------------------------------------")
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
    torch.save(user_power_trace, f"./results_bs2/final_power_trace_{model_name}_{dataset_name}")

    attac_power_trace_1 = []
    attac_power_trace_2 = []
    attac_power_trace_3 = []
    attac_power_trace_4 = []

    for attacker_model in attack_model_mapped_50_50:
        attack_power_trace = []
        if mode == "attacker":
            load_attack_power_info_1 = torch.load(f"trace_data/power_trace_{attacker_model}_synthetic.pt")
        else:
            load_attack_power_info_1 = torch.load(f"trace_data/power_trace_{attacker_model}_{dataset_name}.pt")

        for i in range(len(load_attack_power_info_1)):
            if len(load_attack_power_info_1[i]) > 96:
                ## divide into chunks of 96 and sum them up
                chunks = [load_attack_power_info_1[i][j:j+96] for j in range(0, len(load_attack_power_info_1[i]), 96)]
                for k in range(len(chunks)):
                    attack_power_trace.append(sum(chunks[k]) + len(chunks[k])*peripheral_power)
            else:
                attack_power_trace.append(sum(load_attack_power_info_1[i]) + len(load_attack_power_info_1[i])*peripheral_power)
        attac_power_trace_1 += attack_power_trace
    attac_power_trace_1 += [0]*dummy_tiles

    for attacker_model in attack_model_mapped_20_80:
        attack_power_trace = []
        if mode == "attacker":
            load_attack_power_info_2 = torch.load(f"trace_data/power_trace_{attacker_model}_synthetic.pt")
        else:
            load_attack_power_info_2 = torch.load(f"trace_data/power_trace_{attacker_model}_{dataset_name}.pt")
        for i in range(len(load_attack_power_info_2)):
            if len(load_attack_power_info_2[i]) > 96:
                ## divide into chunks of 96 and sum them up
                chunks = [load_attack_power_info_2[i][j:j+96] for j in range(0, len(load_attack_power_info_2[i]), 96)]
                for k in range(len(chunks)):
                    attack_power_trace.append(sum(chunks[k]) + len(chunks[k])*peripheral_power)
            else:
                attack_power_trace.append(sum(load_attack_power_info_2[i]) + len(load_attack_power_info_2[i])*peripheral_power)  
    
        attac_power_trace_2 += attack_power_trace
    attac_power_trace_2 += [0]*dummy_tiles_20_80

    

    for attacker_model in attack_model_mapped_80_20:
        attack_power_trace = []
        if mode == "attacker":
            load_attack_power_info_3 = torch.load(f"trace_data/power_trace_{attacker_model}_synthetic.pt")
        else:
            load_attack_power_info_3 = torch.load(f"trace_data/power_trace_{attacker_model}_{dataset_name}.pt")
        for i in range(len(load_attack_power_info_3)):
            if len(load_attack_power_info_3[i]) > 96:
                ## divide into chunks of 96 and sum them up
                chunks = [load_attack_power_info_3[i][j:j+96] for j in range(0, len(load_attack_power_info_3[i]), 96)]
                for k in range(len(chunks)):
                    attack_power_trace.append(sum(chunks[k]) + len(chunks[k])*peripheral_power)
            else:
                attack_power_trace.append(sum(load_attack_power_info_3[i]) + len(load_attack_power_info_3[i])*peripheral_power)
        attac_power_trace_3 += attack_power_trace
    attac_power_trace_3 += [0]*dummy_tiles_80_20

    for attacker_model in attack_model_special:
        attack_power_trace = []
        if mode == "attacker":
            load_attack_power_info_4 = torch.load(f"trace_data/power_trace_{attacker_model}_synthetic.pt")
        else:
            load_attack_power_info_4 = torch.load(f"trace_data/power_trace_{attacker_model}_{dataset_name}.pt")
        for i in range(len(load_attack_power_info_4)):
            if len(load_attack_power_info_4[i]) > 96:
                ## divide into chunks of 96 and sum them up
                chunks = [load_attack_power_info_4[i][j:j+96] for j in range(0, len(load_attack_power_info_4[i]), 96)]
                for k in range(len(chunks)):
                    attack_power_trace.append(sum(chunks[k]) + len(chunks[k])*peripheral_power)
            else:
                attack_power_trace.append(sum(load_attack_power_info_4[i]) + len(load_attack_power_info_4[i])*peripheral_power)
        attac_power_trace_4 += attack_power_trace
    attac_power_trace_4 += [0]*dummy_tiles_special

    ## apend two list till the special case
    combined_trace_1 = user_power_trace + attac_power_trace_1 
    combined_trace_2 = user_power_trace + attac_power_trace_2
    combined_trace_3 = user_power_trace + attac_power_trace_3

    ##check if the len of the combined trace is equal to the total number of tiles
    if len(combined_trace_1) != Nx_50_50*Ny_50_50:
        print("len of 50_50",len(combined_trace_1))
        print("total_tile_50_50",Nx_50_50*Ny_50_50)
    if len(combined_trace_2) != Nx_20_80*Ny_20_80:
        print("len of 20_80",len(combined_trace_2))
        print("total_tile_20_80",Nx_20_80*Ny_20_80)
    if len(combined_trace_3) != Nx_80_20*Ny_80_20:
        print("len of 80_20",len(combined_trace_3))
        print("total_tile_80_20",Nx_80_20*Ny_80_20)

    ## for the special case we need spcial mapping.
    copy_user_power_trace = user_power_trace.copy()
    combined_trace_4, interior_indices = combine_lists_with_specific_order(copy_user_power_trace, attac_power_trace_4, Nx_special, Ny_special)
    
    # print("len of special tiles",len(combined_trace_4))
    ##save the indices
    torch.save (interior_indices, f"{floorplan_path_4_attack_1}/interior_indices.pt")

    ##savethem in thirs respective folders
    torch.save(combined_trace_1, f"{floorplan_path_1_attack_1}/combined_power_trace_{model_name}_{dataset_name}_{mode}.pt")
    torch.save(combined_trace_2, f"{floorplan_path_2_attack_1}/combined_power_trace_{model_name}_{dataset_name}_{mode}.pt")
    torch.save(combined_trace_3, f"{floorplan_path_3_attack_1}/combined_power_trace_{model_name}_{dataset_name}_{mode}.pt")
    torch.save(combined_trace_4, f"{floorplan_path_4_attack_1}/combined_power_trace_{model_name}_{dataset_name}_{mode}.pt")

    ## create the power trace file from the lists. 
    # if mode == "attacker":
    output_trace_file_1= f"{floorplan_path_1_attack_1}/power_trace_{model_name}_{dataset_name}_{mode}.ptrace"
    output_trace_file_2= f"{floorplan_path_2_attack_1}/power_trace_{model_name}_{dataset_name}_{mode}.ptrace"
    output_trace_file_3= f"{floorplan_path_3_attack_1}/power_trace_{model_name}_{dataset_name}_{mode}.ptrace"
    output_trace_file_4= f"{floorplan_path_4_attack_1}/power_trace_{model_name}_{dataset_name}_{mode}.ptrace"

    create_ptrace_file(combined_trace_1, output_trace_file_1)
    create_ptrace_file(combined_trace_2, output_trace_file_2)
    create_ptrace_file(combined_trace_3, output_trace_file_3)
    create_ptrace_file(combined_trace_4, output_trace_file_4)


def gen_input_weights_traces(attack_type, model_name, dataset_name, tiles_for_attacker, max_power, path):

    #assuming 8-bit weights and 2bit per cell in each tile you will have 8x12 arrays that why the layer size is 1024x384

    if attack_type == "input_based_attack":

        input_list = {}
        weight_list = {}
        for model in model_name:
            input_list[model] = torch.load(f"./trace_data/inputs_trace_{model}_synthetic.pt")
            weight_list[model] = torch.load(f"./trace_data/weights_trace_{model}_{dataset_name}.pt")
        # input_list = torch.load(f"./trace_data/inputs_trace_{model_name}_synthetic.pt")
        # weight_list = torch.load(f"./trace_data/weights_trace_{model_name}_{dataset_name}.pt")
        gen_attacker_power_info(input_list, weight_list, model_name, dataset_name, attack_type, max_power, path)
    
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
    peripheralpower=2.7e-3
    model_names = ['resnet18','vgg16','densenet121','densenet40', 'alexnet', 'lenet', 'vgg8']

    for model_name in model_names:
        for mode in ['attacker', 'normal']:
            print("model_name",model_name)
            print("mode",mode)
            gen_floor_plan_2d_and_power_trace(model_name, "cifar10", max_power_array,peripheralpower, mode)

if __name__ == "__main__":
    main()