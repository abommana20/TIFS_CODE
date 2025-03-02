import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from math import ceil

quantization_res=8

def quantize_dac(input, scale_max):
    min_val = input.min()
    max_val = input.max()
    
    # Normalize the input to 0-1 range and scale it to 0-scale_max
    output = (input - min_val) / (max_val - min_val) * scale_max
    output = torch.clamp(output, 0, scale_max)  # Ensuring the range is within 0-scale_max
    output = torch.round(output)  # Quantizing to nearest integer

    return output


# quantization function of DAC module

# quantize_weight = quantize_dac
# quantize_input = quantize_dac
def weight_sum_along_rows(tensor, K):
    # Step 1: Count occurrences of 0, 1, 2, 3 along dimension d
    values_to_count = [0, 1, 2, 3]
    counts = torch.stack([torch.sum(tensor == value, dim=-1, keepdim=True) for value in values_to_count], dim=-1)

    # Step 2: Perform the required calculations and sum them
    # Assuming K is a tensor of shape (4,) containing K_00, K_01, K_02, K_03
    # print(counts)
    results = counts * K * 3  # This will broadcast K along the other dimensions
    
    final_result = torch.sum(results, dim=-1)  # Sum across the last dimension

    return final_result
def count_bits(tensor):
    # Initialize a tensor of the same shape to store the counts of ones
    counts = torch.zeros_like(tensor, dtype=torch.int)
    # Loop through each bit position (0 to 7 for 8-bit integers)
    for shift in range(8):
        counts += (tensor >> shift) & 1
    return counts

def num_pad(source, target):
    crxb_index = ceil(source / target)
    num_padding = crxb_index * target - source
    return crxb_index, num_padding
def num_pad_col( source, target):
    crxb_col_fit = target/(quantization_res/2)
    crxb_index = ceil((source * (quantization_res/2)) / target)
    num_padding =int( crxb_index * target / (quantization_res/2) - source)
    return crxb_index, num_padding
def bit_slice_tensor(tensor, wp, cp):
    # Calculate the number of slices needed
    num_slices = wp // cp
    cellRange = 2 ** cp  # The maximum value each slice can have plus one (since we are slicing 2^cp bits each time)

    # Convert tensor to integers if not already
    tensor = tensor.to(torch.int32)
    
    # Prepare the output tensor shape
    output_shape = (*tensor.shape[:-1], tensor.shape[-1] * num_slices)
    output = torch.zeros(output_shape, dtype=torch.int8, device=tensor.device)
    
    # Calculate initial division factor to get the MSB first
    initial_divisor = 2 ** (wp - cp)
    
    # Perform bit slicing
    for i in range(num_slices):
        # Shift the tensor values right by the current bit offset
        shifted_tensor = tensor // initial_divisor
        # Extract the lowest cp bits using fmod from the shifted tensor
        extracted_bits = torch.fmod(shifted_tensor, cellRange)
        # Place extracted bits in the corresponding new dimension in the output tensor
        output[..., i::num_slices] = extracted_bits.to(torch.int8)
        # Reduce the division factor for the next more significant bits
        initial_divisor //= cellRange

    return output.float()

def tile_info_and_weighted_sum_gen(model, model_name, dataset_name, max_power):
    crxb_size = 128
    tile_size = 96 ## ISSAC config has 96 arrays in one tile
    max_weighted_sum_const = crxb_size*8*3*crxb_size
    
    col_size = int(crxb_size/(quantization_res/2))
    ##for conv and linear layers only
    tile_info = []
    power_trace = []
    on_off_ratio = 17
    K = torch.tensor([1/on_off_ratio,(1/3+2/(3*on_off_ratio)),(2/3+1/(3*on_off_ratio)),1])
    
    n_lvl_w = 2 ** quantization_res-1
    h_lvl_w = (n_lvl_w - 2) / 2
    input = torch.load(f"trace_data/inputs_{model_name}_{dataset_name}.pt")
    on_off_ratio = 17
    K = torch.tensor([1/on_off_ratio,(1/3+2/(3*on_off_ratio)),(2/3+1/(3*on_off_ratio)),1])
    idx=0
    for  (name, layer) in (model.named_modules()):
        if isinstance(layer, nn.Linear):
            # max_power.append([])
            # print(layer.weight.shape[1])
            # print(layer.weight.shape[0])
            crxb_row, crxb_row_pads = num_pad(layer.weight.shape[1], crxb_size)
            crxb_col, crxb_col_pads = num_pad_col(layer.weight.shape[0], crxb_size)
            w_pad = (0,  crxb_col_pads, 0,crxb_row_pads)
            input_pad = (0,  crxb_row_pads)
            delta_w = layer.weight.abs().max() / h_lvl_w
            weight_quan = quantize_dac(layer.weight, n_lvl_w)
            weight_flatten = weight_quan.transpose(0, 1)
            # 2.2. add paddings
            weight_padded = F.pad(weight_flatten, w_pad,  mode='constant', value=0)
            weight_crxb = weight_padded.reshape(crxb_row,  crxb_col, crxb_size, col_size)
            mapped_weights = bit_slice_tensor(weight_crxb, quantization_res, 2)
            total_number_of_tiles = ceil(mapped_weights.shape[0]*mapped_weights.shape[1] / tile_size)
            tile_info.append(total_number_of_tiles)

            ## now find the weighted sum
            input_layer =input[idx]
            
            input_layer = torch.from_numpy(input_layer)
            
            # print(len(input_layer.size()))
            delta_x = input_layer.abs().max() / h_lvl_w
            ##check the mean of the input
            # mean_input = torch.mean(input_layer)
            # std = torch.std(input_layer)
            # print("mean_input",mean_input)
            # print("std",std)
            #print("input",input,input.size())
            # input_clip = F.hardtanh(input_layer, min_val=-h_lvl_w * delta_x.item(),
            #                         max_val=h_lvl_w * delta_x.item())
            #print("input_clip",input_clip,input_clip.size())
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
            print(avg_weighted_sum)
            print(max_weighted_sum_const)
            ratio = avg_weighted_sum / max_weighted_sum_const
            power = ratio * max_power
            # print("power",power)    
            power_list = power.flatten().tolist()
            power_trace.append(power_list)
            idx+=1
        if isinstance(layer, nn.Conv2d):
            weight_flatten = layer.weight.view(layer.weight.shape[0], -1)
            # print("weight_flatten",weight_flatten.size())
            crxb_row, crxb_row_pads = num_pad(weight_flatten.shape[1], crxb_size)
            crxb_col, crxb_col_pads = num_pad_col(weight_flatten.shape[0], crxb_size)
            # print("crxb_row",crxb_row)
            input_pad = (0, 0, 0, crxb_row_pads)
            w_pad = (0,  crxb_col_pads, 0,crxb_row_pads)
            delta_w = layer.weight.abs().max() / h_lvl_w
            weight_quan = quantize_dac(layer.weight, n_lvl_w)
            del weight_flatten
            weight_flatten = weight_quan.view(layer.weight.shape[0], -1).transpose(0, 1)
            # 2.2. add paddings
            weight_padded = F.pad(weight_flatten, w_pad,  mode='constant', value=0)
            weight_crxb = weight_padded.reshape(crxb_row,  crxb_col, crxb_size, col_size)
            mapped_weights = bit_slice_tensor(weight_crxb, quantization_res, 2)
            total_number_of_tiles = ceil(mapped_weights.shape[0]*mapped_weights.shape[1] / tile_size)
            tile_info.append(total_number_of_tiles)
            ## now find the weighted sum

            input_layer = input[idx]
            input_layer = torch.from_numpy(input_layer)
            # print(input_layer.size())
            ##check the mean of the input
           
            # print("input_layer",input_layer)
            delta_x = input_layer.abs().max() / h_lvl_w
            #print("input",input,input.size())
            # input_clip = F.hardtanh(input_layer, min_val=-h_lvl_w * delta_x.item(),
                                    # max_val=h_lvl_w * delta_x.item())
            #print("input_clip",input_clip,input_clip.size())
            input_quan = quantize_dac(input_layer, n_lvl_w)
            # mean_input = torch.mean(weight_quan)
            # std = torch.std(weight_quan)
            # print("mean_input",mean_input)
            # print("std",std)
            input_unfold = F.unfold(input_quan, kernel_size=layer.kernel_size[0],
                                dilation=layer.dilation, padding=layer.padding,
                                stride=layer.stride)
            # print("input_unfold",input_unfold.size())
            input_padded = F.pad(input_unfold, input_pad,mode='constant', value=0)
            # print("input_padded",input_padded.size())
            # 2.3. reshape to crxb size
            input_crxb = input_padded.view(input_layer.shape[0], 1, crxb_row, crxb_size, input_padded.shape[2])

  
            input_crxb = input_crxb.to(torch.int16)
            count_ones = count_bits(input_crxb)
            weight_sum = weight_sum_along_rows(mapped_weights, K)
            A_squeezed = weight_sum.squeeze(-1)
            B_squeezed = count_ones.squeeze(0).squeeze(0)
            weighted_sum = torch.matmul(A_squeezed.float(), B_squeezed.float())
            ##do avg across last dimension 
            avg_weighted_sum = torch.mean(weighted_sum, dim=-1)
            print(avg_weighted_sum)
            print(max_weighted_sum_const)
            ratio = avg_weighted_sum / max_weighted_sum_const
            power = ratio * max_power
            power_list = power.flatten().tolist()
            # power= (avg_weighted_sum/ max_weighted_sum_const) * max_power
            # print("power",power.size())
            power_trace.append(power_list)
            idx+=1

    ##save the tile info and power trace
    torch.save(tile_info, f"trace_data/tile_info_{model_name}_{dataset_name}.pt")
    torch.save(power_trace, f"trace_data/power_trace_{model_name}_{dataset_name}.pt")

# Example usage

from resnet import *
from densenet import *
from VGG import *
from alexnet import *
from lenet import *
from vgg8 import *


if __name__ == "__main__":
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    dataset_name = "cifar10"
    model_names = ['resnet18', 'vgg16', 'densenet121', 'densenet40', 'alexnet', 'lenet', 'vgg8']
    # model_names = ['resnet18']
    max_power_array = 8e-3

    for model_name in model_names:
        print(f"Generating tile info and power trace for {model_name} on {dataset_name}...")
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

        tile_info_and_weighted_sum_gen(model, model_name, dataset_name, max_power_array)
