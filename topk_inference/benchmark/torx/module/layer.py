import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json

from .adc import adc
from .dac import quantize_dac
from .w2g import w2g

quantize_input = quantize_dac
quantize_weight = quantize_dac
adc = adc

def top_10_percent_boolean(tensor,x, num_elements):
    """
    Create a boolean tensor where the top 10% of values (by magnitude) are True, and the rest are False.
    
    :param tensor: Input tensor
    :return: Boolean tensor with True for top 10% magnitude values
    """
    # Flatten the tensor to simplify processing
    flat_tensor = tensor.flatten()
    
    # Compute the threshold value for the top 10% by magnitude
    k = int(x * num_elements)  # Number of top elements
    if k == 0:
        raise ValueError("Tensor size is too small to compute top 10%.")
    
    # Sort the flat tensor in descending order
    sorted_values, sorted_indices = torch.sort(flat_tensor, descending=True)
    
    # Identify top-k indices
    top_k_indices = sorted_indices[:k]
    
    # Create a boolean mask initialized to False
    mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
    
    # Mark those top-k positions as True
    mask[top_k_indices] = True
    
    # Reshape mask to original shape
    mask = mask.view_as(tensor)
    return mask
    
    #   # Find the top-k values by magnitude
    # threshold = torch.topk(flat_tensor, k).values.min()
    
    # # Create a boolean tensor
    # boolean_tensor = (tensor >= threshold)
    
    # return boolean_tensor

# Example Usage
# tensor = torch.randint(1,10,(8,8)) # Example 4x4 tensor
# print("Input Tensor:\n", tensor)

# boolean_tensor = top_10_percent_boolean(tensor)
# print("Boolean Tensor (Top 10% by Magnitude):\n", boolean_tensor)
def replicate_last_dimension(tensor, num_times):
    """
    Replicate the last dimension of a tensor `num_times` times.
    
    :param tensor: Input tensor of shape (..., d)
    :param num_times: Number of times to replicate the last dimension
    :return: Tensor with the last dimension replicated, shape (..., num_times * d)
    """
    if num_times <= 0:
        raise ValueError("The number of times to replicate must be positive.")

    # Get the original shape of the tensor
    *leading_dims, last_dim = tensor.shape

    # Reshape the tensor to prepare for replication
    tensor = tensor.unsqueeze(-1)  # Add a new dimension after the last dimension

    # Repeat the tensor along the new dimension
    replicated_tensor = tensor.repeat_interleave(num_times, dim=-1)

    # Reshape the tensor to merge the replicated dimension with the original last dimension
    replicated_tensor = replicated_tensor.view(*leading_dims, num_times * last_dim)

    return replicated_tensor

def reduce_and_reshape_conv(tensor, cells, cell_range, original_value_row, original_value_col):
    # tensor: input tensor of shape (a, b, c, d)
    # cells: the size of each chunk in dimension d
    # cell_range: the base for the exponent used in the transformation

    # Get the size of the tensor along each dimension
    # print("tensor shape", tensor.shape)
    a, b, c, d = tensor.shape
    
    # Calculate the new size for dimension d
    k = d // cells

    # Reshape the tensor so that dimension d is split into chunks of size cells
    tensor_reshaped = tensor.reshape(a, b, c, k, cells)

    # Create a tensor for the weights, decreasing powers of cell_range
    weights = torch.tensor([cell_range**i for i in range(cells-1, -1, -1)]).to(tensor.device)

    # Compute the weighted sum across the last dimension (cells)
    tensor_reduced = torch.sum(tensor_reshaped * weights, dim=-1)

    # Further reshape to combine dimensions a*c and b*k
    final_tensor = tensor_reduced.reshape(a * c, b * k)[ :original_value_row, :original_value_col].transpose(0, 1)
    # print("final_tensor shape", final_tensor.shape)

    return final_tensor
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Construct the path to the config.json file
# config_path = os.path.join(os.path.dirname(__file__), '../../../config/new_config.json')

# Load the JSON configuration


# quantize = 8

class crxb_Conv2d(nn.Conv2d):
    """
    This is the custom conv layer that takes non-ideal effects of ReRAM crossbar into account. It has three functions.
    1) emulate the DAC at the input of the crossbar and qnantize the input and weight tensors.
    2) map the quantized tensor to the ReRAM crossbar arrays and include non-ideal effects such as noise, ir drop, and
        SAF.
    3) emulate the ADC at the output of he crossbar and convert the current back to digital number
        to the input of next layers

    Args:
        ir_drop(bool): switch that enables the ir drop calculation.
        device(torch.device): device index to select. It’s a no-op if this argument is a negative integer or None.
        gmax(float): maximum conductance of the ReRAM.
        gmin(float): minimun conductance of the ReRAM.
        gwire(float): conductance of the metal wire.
        gload(float): load conductance of the ADC and DAC.
        scaler_dw(float): weight quantization scaler to reduce the influence of the ir drop.
        vdd(float): supply voltage.
        enable_stochastic_noise(bool): switch to enable stochastic_noise.
        freq(float): operating frequency of the ReRAM crossbar.
        temp(float): operating temperature of ReRAM crossbar.
        crxb_size(int): size of the crossbar.
        quantize(int): quantization resolution of the crossbar.
        enable_SAF(bool): switch to enable SAF
        enable_ec_SAF(bool): switch to enable SAF error correction.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True,groups=1,dilation=1,device=None, config_path=None, layer_count = 0, path_to_temp=None,thermal_noise_mode=1,topk_list = None, tm=0.1, model_name="resnet18" , enable_fec=1):
        super(crxb_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)

        assert self.groups == 1, "currently not support grouped convolution for custom conv"

        with open(config_path+'/config.json', 'r') as config_file:
            config = json.load(config_file)

       
        # self.device = config['device']
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ################## Crossbar conversion #############################
        self.crxb_size = config['crxb_size']
        self.enable_ec_SAF = config['enable_ec_SAF']
        self.enable_ec_SAF_msb = config['enable_ec_SAF_msb']
        self.ir_drop = config['ir_drop']
        self.quantize_weights =  config['quantize']
        self.adc_resolution =  config['adc_resolution']
        self.input_quantize =  config['input_quantize']
        self.sa00_rate = config['sa00_rate']
        self.sa01_rate = config['sa01_rate']
        self.sa10_rate = config['sa10_rate']
        self.sa11_rate = config['sa11_rate']

        self.fault_rate = config['fault_rate']
        # print("fault_rate",self.fault_rate) 
        self.fault_dist = config['fault_dist']
        self.layer_count = layer_count
        self.config_path = config_path
        self.tm= tm
        # self.inject_layer_idx = inject_layer_idx
             
        ################## Crossbar conversion #############################
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.col_size = int(self.crxb_size/(self.quantize_weights/2))

        self.nchout_index = torch.arange(self.out_channels).to(self.device)
        weight_flatten = self.weight.view(self.out_channels, -1)
        #print("weights_flatten size: ", weight_flatten.size())
        #print("weights_ size: ", self.weight.size())
        self.crxb_row, self.crxb_row_pads = self.num_pad(
            weight_flatten.shape[1], self.crxb_size)
        self.crxb_col, self.crxb_col_pads = self.num_pad_col(
            weight_flatten.shape[0], self.crxb_size)
        self.h_out = None
        self.w_out = None
        self.w_pad = (0, self.crxb_col_pads, 0, self.crxb_row_pads)
        self.input_pad = (0, 0, 0, self.crxb_row_pads)
        weight_padded = F.pad(weight_flatten.transpose(0,1), self.w_pad,
                              mode='constant', value=0)
        weight_crxb = weight_padded.reshape(self.crxb_row,  self.crxb_col, self.crxb_size, self.col_size)
        weight_crxb_interleaved = replicate_last_dimension(weight_crxb,(self.quantize_weights//2))
        pos_mask = weight_crxb_interleaved.to(self.device) >0
        neg_mask = weight_crxb_interleaved.to(self.device) <0
        # weight_crxb = weight_padded.view(self.crxb_col, self.col_size,
        #                                  self.crxb_row, self.crxb_size).transpose(1, 2)

        ################# Hardware conversion ##############################
        # weight and input levels
        self.n_lvl = 2 ** self.input_quantize
        self.h_lvl = (self.n_lvl - 2) / 2
        self.n_lvl_w = 2 ** self.quantize_weights
        self.h_lvl_w = (self.n_lvl_w - 2) / 2
        self.n_lvl_adc = 2 ** (self.adc_resolution)
        self.h_lvl_adc = (self.n_lvl_adc - 2) / 2
        self.enable_SAF= config['enable_SAF']
        self.topk_list = topk_list
        # ReRAM cells
        self.thermal_noise_mode = thermal_noise_mode

        self.Gmax = config['gmax']  # max conductance
        self.Gmin = config['gmin']  # min conductance
        self.delta_g = (self.Gmax - self.Gmin) / (2 ** 2-1)  # conductance step
        self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA00=self.Gmin, G_SA01=self.Gmax-2*self.delta_g, G_SA10=self.Gmin+2*self.delta_g, G_SA11=self.Gmax,
                       weight_shape=weight_crxb.shape, p_SA00=self.sa00_rate, p_SA01=self.sa01_rate, p_SA10=self.sa10_rate,p_SA11=self.sa11_rate, 
                       enable_rand=True, enable_SAF=config['enable_SAF'], fault_rate = self.fault_rate, fault_dist = self.fault_dist, msb_only_ec = self.enable_ec_SAF_msb, 
                       device=self.device, config_path=self.config_path, layer_count=self.layer_count, inject_layer_idx = 0)
        self.Gwire = config['gwire']
        self.Gload = config['gload']
        # DAC
        self.Vdd = config['vdd']  # unit: volt
        self.delta_v = self.Vdd / (self.n_lvl - 1)
        self.delta_v_adc = self.Vdd / (self.n_lvl_adc - 1)

        self.scaler_dw = config['scaler_dw']
        self.num_xbs_tile = 96  # number of crossbar tiles
         ################ Stochastic Conductance Noise setup #########################
        # parameters setup
        self.enable_stochastic_noise = config['enable_noise']
        self.freq = config['freq']  # operating frequency
        self.kb = 1.38e-23  # Boltzmann const
        self.temp = config['temp']  # temperature in kelvin
        self.q = 1.6e-19  # electron charge
        # self.thermal_noise_mode = "1"

        self.tau = 0.5  # Probability of RTN
        self.a = 1.662e-7  # RTN fitting parameter
        self.path= path_to_temp
        self.b = 0.0015  # RTN fitting parameter
        self.enable_fec = enable_fec
        self.model_name = model_name
        self.weight_shape = [weight_crxb.shape[0], weight_crxb.shape[1], weight_crxb.shape[2], weight_crxb.shape[3]*self.quantize_weights//2]
        # if self.enable_fec==1:
        speed_up = torch.load(f"./speed_up_{self.model_name}.pt", weights_only = True)
        speed_up_layer = speed_up[self.layer_count]
        hessian_score_path = torch.load(f"./hessian_score_crxb/hessian_score_crxb_{self.model_name}.pt", map_location=self.device,weights_only=True)
        hessian_score = hessian_score_path[self.layer_count]
        # print("size of hessian score, ",hessian_score.size())
        # print("-----------------------------------------",self.layer_count,"------------------------------------")
        # hessian_bool = torch.load(f"./hessian_bool/hessian_bool_{self.model_name}.pt", map_location=self.device,weights_only=True)
        path_to_fault_tolerance_tile = f"./fault_tolerance_tile/fault_tolerance_tile_{self.model_name}.pt"
        fault_tolerance_tile = torch.load(path_to_fault_tolerance_tile, map_location=self.device,weights_only=True)
        fault_tolerance_tile_layer = fault_tolerance_tile[self.layer_count]
        # print("weight protection",self.topk_list[self.layer_count])
        hessian_bool_layer = top_10_percent_boolean(hessian_score,self.topk_list[self.layer_count],self.weight.numel())
        # print("total weights", self.weight.numel(), "total ones", hessian_bool_layer.sum().item(), hessian_bool_layer.float().numel())
        # print("sum of ones before for layer ",self.layer_count,"", hessian_bool_layer.sum().item()/self.weight.numel()*100)
        hessian_score_booled = hessian_score * hessian_bool_layer.float() ##mask of only top-k hessain scores.

        hessian_score_booled = hessian_score_booled.reshape(-1, hessian_score_booled.shape[2], hessian_score_booled.shape[3])
        final_boolean = torch.zeros(hessian_score_booled.shape).to(self.device)
        tile_size = 96
       
        for tile_idx in range(len(fault_tolerance_tile_layer)):
            bool_hessian_tile = hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:]>0 ##only places where the critical weights are marked. 
            # print("totalones",(bool_hessian_tile.sum().item()*self.quantize_weights//2), (128*128*tile_size)*100 ,(bool_hessian_tile.sum().item()*self.quantize_weights//2)/(128*128*tile_size)*100) 
            if bool_hessian_tile.sum().item()*self.quantize_weights//2 <= 128*128*tile_size*self.tm:
                final_boolean[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:] = bool_hessian_tile
                # print("iam here")
            else:
            # print(128*(128/(self.quantize_weights//2))*tile_size, hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:].numel())
                final_boolean[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:] = top_10_percent_boolean(hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:],self.tm,hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:].numel()) # 

        self.boolean_replicated = replicate_last_dimension(final_boolean, self.quantize_weights//2)
        ##################################
        overhead_tiles = []
        new_tm_tile = []
        self.tile_fraction_list = []
        for tile_idx in range(math.ceil(self.boolean_replicated.shape[0]/tile_size)):
            start_idx= tile_idx*tile_size
            end_idx = min((tile_idx+1)*tile_size,self.boolean_replicated.shape[0] )
            # print("Num_ones after interleaving, ",self.final_boolean_replicated[start_idx:end_idx, :,:].sum().item())
            overhead = self.boolean_replicated[start_idx:end_idx, :,:].sum().item()/(128*(128)*tile_size)
            overhead_tiles.append(overhead*speed_up_layer)
            self.tile_fraction_list.append(min(overhead*speed_up_layer, self.tm))
            # print(overhead*speed_up_layer, overhead, speed_up_layer)
            if overhead*speed_up_layer > self.tm:
                new_tm_tile.append(self.tm-((self.tm-(overhead*speed_up_layer))/speed_up_layer))
                # print("Iamhere")
            else:
                new_tm_tile.append(self.tm)
        # print(new_tm_tile) 
            # print(overhead)
        final_boolean = torch.zeros(hessian_score_booled.shape).to(self.device)
        for tile_idx in range(len(fault_tolerance_tile_layer)):
            bool_hessian_tile = hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:]>0 ##only places where the critical weights are marked. 
            final_boolean[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:] = top_10_percent_boolean(hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:],new_tm_tile[tile_idx],hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:].numel()) # 
        
        self.final_boolean_replicated = replicate_last_dimension(final_boolean, self.quantize_weights//2)
        
        
        #modify the last dimension
        self.final_boolean_replicated = self.final_boolean_replicated.reshape(self.weight_shape).to(self.device)
        #############################################################################
        pos_final_boolean_replicated = (self.final_boolean_replicated * pos_mask)
        neg_final_boolean_replicated = (self.final_boolean_replicated * neg_mask)
        
        self.comp_mask_pos = pos_final_boolean_replicated
        self.comp_mask_neg = neg_final_boolean_replicated


    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding

    def num_pad_col(self, source, target):
        #crxb_col_fit = target/(self.quantize_weights/2)
        crxb_index = math.ceil((source * (self.quantize_weights/2)) / target)
        num_padding =int( crxb_index * target / (self.quantize_weights/2) - source)
        return crxb_index, num_padding



    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() / self.h_lvl_w * self.scaler_dw
            self.delta_x = input.abs().max() / self.h_lvl

        input_clip = F.hardtanh(input, min_val=-self.h_lvl * self.delta_x.item(),
                                max_val=self.h_lvl * self.delta_x.item())

        input_quan = quantize_input(
            input_clip, self.delta_x) * self.delta_v  # convert to voltage


        weight_quan = quantize_weight(self.weight, self.delta_w)
        
        # 2. Perform the computation between input voltage and weight conductance
        if self.h_out is None and self.w_out is None:
            self.h_out = int(
                (input.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
            self.w_out = int(
                (input.shape[3] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)

        # 2.1 flatten and unfold the weight and input

        input_unfold = F.unfold(input_quan, kernel_size=self.kernel_size[0],
                                dilation=self.dilation, padding=self.padding,
                                stride=self.stride)
        weight_flatten = weight_quan.view(self.out_channels, -1).transpose(0, 1)

        # 2.2. add paddings
        weight_padded = F.pad(weight_flatten, self.w_pad,
                              mode='constant', value=0)
        input_padded = F.pad(input_unfold, self.input_pad,
                             mode='constant', value=0)

        # 2.3. reshape to crxb size
        input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                                       self.crxb_size, input_padded.shape[2])
        # weight_crxb = weight_padded.view(self.crxb_col, self.col_size,
        #                                  self.crxb_row, self.crxb_size).transpose(1, 2)
        weight_crxb = weight_padded.reshape(self.crxb_row,  self.crxb_col, self.crxb_size, self.col_size)
        # convert the floating point weight into conductance pair values
        #print("crxb_weight",weight_crxb)
        G_crxb, error = self.w2g(weight_crxb)
        if self.enable_fec == 1:
            # quantized_pos_weights = G_crxb[0] * self.final_boolean_replicated.float()
            # quantized_neg_weights = G_crxb[1] * self.final_boolean_replicated.float()
            quantized_pos_weights = (G_crxb[0]-self.Gmin)/self.delta_g * self.final_boolean_replicated.float()
            quantized_neg_weights = (G_crxb[1]-self.Gmin)/self.delta_g * self.final_boolean_replicated.float()
            G_crxb[0] = G_crxb[0] * torch.logical_not(self.final_boolean_replicated)
            G_crxb[1] = G_crxb[1] * torch.logical_not(self.final_boolean_replicated)

        no_of_tiles = math.ceil(G_crxb[0].shape[0] * G_crxb[0].shape[1]/ self.num_xbs_tile)
        ##create a list of temperatures for each tile

        # no_of_tiles = math.ceil(G_crxb[0].shape[0] * G_crxb[0].shape[1]/ self.num_xbs_tile)
        ##create a list of temperatures for each tile

        temp_tile_info = torch.load(self.path,weights_only=True)
        temp_tile_layer = temp_tile_info[self.layer_count]
        original_shape = G_crxb[0].shape
        flattened_G_crxb_0 = G_crxb[0].reshape(-1, G_crxb[0].shape[2], G_crxb[0].shape[3])
        flattened_G_crxb_1 = G_crxb[1].reshape(-1, G_crxb[1].shape[2], G_crxb[1].shape[3])

    
        if self.enable_stochastic_noise:
            
            ##linear decrease 
            m = -0.1  # slope 20 to 15 m=-0.05 -0.15(20 to 5) -0.1(20 to 10)
            b = 50  # interceptb 35(20 to 15) 65(20 to 5) 50(20 to 10)


            for tile_idx in range(no_of_tiles):
                start_idx = tile_idx * self.num_xbs_tile
                end_idx = min((tile_idx + 1) * self.num_xbs_tile, flattened_G_crxb_0.shape[0])

                # Apply temperature modification on the slice of the flattened array
                tile_temp = temp_tile_layer[tile_idx]
                if self.thermal_noise_mode == 1: ##20to15
                    flattened_G_crxb_0[start_idx:end_idx] *= (-0.05 * tile_temp + 35) / 20
                    flattened_G_crxb_1[start_idx:end_idx] *= (-0.05 * tile_temp + 35) / 20
                elif self.thermal_noise_mode == 2: ##20to10
                    flattened_G_crxb_0[start_idx:end_idx] *= (-0.1 * tile_temp + 50) / 20
                    flattened_G_crxb_1[start_idx:end_idx] *= (-0.1 * tile_temp + 50) / 20
                else:##20to5
                    flattened_G_crxb_0[start_idx:end_idx] *= 1 / (1 + 0.01 * (tile_temp - 300))
                    flattened_G_crxb_1[start_idx:end_idx] *= 1 / (1 + 0.01 * (tile_temp - 300))
            # Reshape G_crxb[0] back to its original shape
            G_crxb[0] = flattened_G_crxb_0.reshape(original_shape)
            G_crxb[1] = flattened_G_crxb_1.reshape(original_shape)

            ##clip the conductance values to Gmin and Gmax
            G_crxb[0] = torch.clamp(G_crxb[0], self.Gmin, self.Gmax)
            G_crxb[1] = torch.clamp(G_crxb[1], self.Gmin, self.Gmax)

        G_crxb[0] = ((G_crxb[0] - self.Gmin)/self.delta_g)
        G_crxb[1] = ((G_crxb[1] - self.Gmin)/self.delta_g)

      

        cell_res = 2
        cellRange = 2**cell_res
        # print("G_crxb shape",G_crxb[0].shape)
        xb_pos = reduce_and_reshape_conv(G_crxb[0], self.quantize_weights//cell_res, cellRange, self.kernel_size[0]*self.kernel_size[1]*self.in_channels, self.out_channels)
        xb_neg= reduce_and_reshape_conv(G_crxb[1], self.quantize_weights//cell_res, cellRange, self.kernel_size[0]*self.kernel_size[1]*self.in_channels, self.out_channels)
        a, b, c, d = self.weight.shape
        if self.enable_fec ==1:
            err_pos = reduce_and_reshape_conv(quantized_pos_weights, self.quantize_weights//cell_res, cellRange, self.kernel_size[0]*self.kernel_size[1]*self.in_channels, self.out_channels)
            err_neg = reduce_and_reshape_conv(quantized_neg_weights, self.quantize_weights//cell_res, cellRange, self.kernel_size[0]*self.kernel_size[1]*self.in_channels, self.out_channels)
        # print("xb_pos shape",xb_pos.shape)
        # print("xb_neg shape",xb_neg.shape)
        # print("weight shape",self.weight.shape)
        xb_pos = xb_pos.reshape(a, b, c, d)
        xb_neg = xb_neg.reshape(a, b, c, d)
        if self.enable_fec ==1:
            err_pos = err_pos.reshape(a, b, c, d)
            err_neg = err_neg.reshape(a, b, c, d)
        ##dequantize the conductance values
        dequant_weight = torch.clamp((xb_pos - xb_neg)*self.delta_w, min=-self.h_lvl_w*self.delta_w, max=self.h_lvl_w*self.delta_w)
        if self.enable_fec ==1:
            dequant_error = torch.clamp((err_pos - err_neg)*self.delta_w, min=-self.h_lvl_w*self.delta_w, max=self.h_lvl_w*self.delta_w)
            output_comp = F.conv2d(input, dequant_error, None, self.stride, self.padding, self.dilation, self.groups)
        output = F.conv2d(input, dequant_weight, None, self.stride, self.padding, self.dilation, self.groups)
        
      
        if self.enable_fec == 1:
            output+=output_comp
        
        #print("output",output_add)
      

        if self.bias is not None:
            output += self.bias.unsqueeze(1).unsqueeze(1)


        return output.to(self.device)

    # def _reset_delta(self):
    #     self.delta_in_sum.data[0] = 0
    #     self.delta_out_sum.data[0] = 0
    #     self.counter.data[0] = 0

class crxb_Linear(nn.Linear):
    """
    This is the custom linear layer that takes non-ideal effects of ReRAM crossbar into account. It has three functions.
    1) emulate the DAC at the input of the crossbar and qnantize the input and weight tensors.
    2) map the quantized tensor to the ReRAM crossbar arrays and include non-ideal effects such as noise, ir drop, and
        SAF.
    3) emulate the ADC at the output of he crossbar and convert the current back to digital number
        to the input of next layers

    Args:
        ir_drop(bool): switch that enables the ir drop calculation.
        device(torch.device): device index to select. It’s a no-op if this argument is a negative integer or None.
        gmax(float): maximum conductance of the ReRAM.
        gmin(float): minimun conductance of the ReRAM.
        gwire(float): conductance of the metal wire.
        gload(float): load conductance of the ADC and DAC.
        vdd(float): supply voltage.
        scaler_dw(float): weight quantization scaler to reduce the influence of the ir drop.
        enable_stochastic_noise(bool): switch to enable stochastic_noise.
        freq(float): operating frequency of the ReRAM crossbar.
        temp(float): operating temperature of ReRAM crossbar.
        crxb_size(int): size of the crossbar.
        quantize(int): quantization resolution of the crossbar.
        enable_SAF(bool): switch to enable SAF
        enable_ec_SAF(bool): switch to enable SAF error correction.
    """

    def __init__(self, in_features, out_features, bias=True,device=None,config_path=None, layer_count = 0, path_to_temp = None,thermal_noise_mode=1, topk_list = None, tm=0.1,model_name="resnet18", enable_fec=1):
        super(crxb_Linear, self).__init__(in_features, out_features, bias)
        
        with open(config_path+'/config.json', 'r') as config_file:
            config = json.load(config_file)
        # self.device = config['device']
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ################## Crossbar conversion #############################
        self.crxb_size = config['crxb_size']
        self.enable_ec_SAF = config['enable_ec_SAF']
        self.enable_ec_SAF_msb = config['enable_ec_SAF_msb']
        self.ir_drop = config['ir_drop']
        self.quantize_weights =  config['quantize']
        self.adc_resolution =  config['adc_resolution']
        self.input_quantize =  config['input_quantize']
        self.sa00_rate = config['sa00_rate']
        self.sa01_rate = config['sa01_rate']
        self.sa10_rate = config['sa10_rate']
        self.sa11_rate = config['sa11_rate']
        self.config_path = config_path
        self.layer_count = layer_count
        self.fault_rate = config['fault_rate'] 
        self.fault_dist = config['fault_dist']
        # self.tm = tm
        ################## Crossbar conversion #############################
        self.col_size = int(self.crxb_size/(self.quantize_weights/2))

        self.out_index = torch.arange(out_features).to(self.device)
        self.crxb_row, self.crxb_row_pads = self.num_pad(
            self.weight.shape[1], self.crxb_size)
        self.crxb_col, self.crxb_col_pads = self.num_pad_col(
            self.weight.shape[0], self.crxb_size)
        self.w_pad = (0, self.crxb_col_pads, 0, self.crxb_row_pads)
        self.input_pad = (0, self.crxb_row_pads)
        weight_padded = F.pad(self.weight.transpose(0,1), self.w_pad,
                              mode='constant', value=0)
        # weight_crxb = weight_padded.view(self.crxb_col, self.col_size,
        #                                  self.crxb_row, self.crxb_size).transpose(1, 2)
        weight_crxb = weight_padded.reshape(self.crxb_row,  self.crxb_col, self.crxb_size, self.col_size)
        weight_crxb_interleaved = replicate_last_dimension(weight_crxb,(self.quantize_weights//2))
        pos_mask = weight_crxb_interleaved.to(self.device) >0
        neg_mask = weight_crxb_interleaved.to(self.device) <0

        ################# Hardware conversion ##############################
        # weight and input levels
        self.n_lvl = 2 ** self.input_quantize
        self.h_lvl = (self.n_lvl - 2) / 2
        self.n_lvl_w = 2 ** self.quantize_weights
        self.h_lvl_w = (self.n_lvl_w - 2) / 2
        self.n_lvl_adc = 2 ** (self.adc_resolution)
        self.h_lvl_adc = (self.n_lvl_adc - 2) / 2
        # ReRAM cells
        self.thermal_noise_mode = thermal_noise_mode
       
        self.Gmax = config['gmax']  # min conductance
        self.Gmin = config['gmin'] 
        print(self.Gmax, self.Gmin)
        self.delta_g = (self.Gmax - self.Gmin) / (2 ** 2 -1)  # conductance step
        self.enable_SAF= config['enable_SAF']
        # self.inject_layer_idx = inject_layer_idx
        self.w2g = w2g(self.delta_g,Gmin=self.Gmin, G_SA00=self.Gmin, G_SA01=self.Gmax-2*self.delta_g, G_SA10=self.Gmin+2*self.delta_g, G_SA11=self.Gmax,
                       weight_shape=weight_crxb.shape, p_SA00=self.sa00_rate, p_SA01=self.sa01_rate, p_SA10=self.sa10_rate,p_SA11=self.sa11_rate, 
                       enable_rand=True, enable_SAF=config['enable_SAF'],fault_rate = self.fault_rate, fault_dist = self.fault_dist, msb_only_ec = self.enable_ec_SAF_msb,
                       device=self.device, config_path=self.config_path, layer_count=self.layer_count, inject_layer_idx = 0) 
        self.Gwire = config['gwire']
        self.Gload = config['gload']
        # DAC
        self.scaler_dw = config['scaler_dw']
        self.Vdd = config['vdd']  # unit: volt
        self.delta_v = self.Vdd / (self.n_lvl - 1)
        self.delta_v_adc = self.Vdd / (self.n_lvl_adc - 1)
       
        ################ Stochastic Conductance Noise setup #########################
        # parameters setup
        self.enable_stochastic_noise = config['enable_noise']
        self.freq = config['freq']  # operating frequency
        self.kb = 1.38e-23  # Boltzmann const
        self.temp = config['temp']  # temperature in kelvin
        self.q = 1.6e-19  # electron charge
        self.num_xbs_tile = 96
        self.tau = 0.5  # Probability of RTN
        self.a = 1.662e-7  # RTN fitting parameter
        self.b = 0.0015  # RTN fitting parameter
        self.path = path_to_temp
        self.enable_fec = enable_fec
        self.model_name = model_name
        self.topk_list = topk_list
        self.tm = tm

        speed_up = torch.load(f"./speed_up_{self.model_name}.pt", weights_only = True)
        speed_up_layer = speed_up[self.layer_count]
        self.weight_shape = [weight_crxb.shape[0], weight_crxb.shape[1], weight_crxb.shape[2], weight_crxb.shape[3]*self.quantize_weights//2]
        # if self.enable_fec==1:
        hessian_score_path = torch.load(f"./hessian_score_crxb/hessian_score_crxb_{self.model_name}.pt", map_location=self.device,weights_only=True)
        hessian_score = hessian_score_path[self.layer_count]
        # print("size of hessian score, ",hessian_score.size())
        
        # hessian_bool = torch.load(f"./hessian_bool/hessian_bool_{self.model_name}.pt", map_location=self.device,weights_only=True)
        path_to_fault_tolerance_tile = f"./fault_tolerance_tile/fault_tolerance_tile_{self.model_name}.pt"
        fault_tolerance_tile = torch.load(path_to_fault_tolerance_tile, map_location=self.device,weights_only=True)
        fault_tolerance_tile_layer = fault_tolerance_tile[self.layer_count]
        # print("weight protection",self.topk_list[self.layer_count])
        hessian_bool_layer = top_10_percent_boolean(hessian_score,self.topk_list[self.layer_count],self.weight.numel())
        # print("total weights", self.weight.numel(), "total ones", hessian_bool_layer.sum().item(), hessian_bool_layer.float().numel())
        # print("sum of ones before for layer ",self.layer_count,"", hessian_bool_layer.sum().item()/self.weight.numel()*100)
        hessian_score_booled = hessian_score * hessian_bool_layer.float() ##mask of only top-k hessain scores.

        hessian_score_booled = hessian_score_booled.reshape(-1, hessian_score_booled.shape[2], hessian_score_booled.shape[3])
        final_boolean = torch.zeros(hessian_score_booled.shape).to(self.device)
        tile_size = 96

        for tile_idx in range(len(fault_tolerance_tile_layer)):
            bool_hessian_tile = hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:]>0 ##only places where the critical weights are marked. 
            # print("totalones",(bool_hessian_tile.sum().item()*self.quantize_weights//2), (128*128*tile_size)*100 ,(bool_hessian_tile.sum().item()*self.quantize_weights//2)/(128*128*tile_size)*100) 
            if bool_hessian_tile.sum().item()*self.quantize_weights//2 <= 128*128*tile_size*self.tm:
                final_boolean[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:] = bool_hessian_tile
                # print("iam here")
            else:
            # print(128*(128/(self.quantize_weights//2))*tile_size, hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:].numel())
                final_boolean[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:] = top_10_percent_boolean(hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:],self.tm,hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:].numel()) # 
            ##get the sum 

        self.boolean_replicated = replicate_last_dimension(final_boolean, self.quantize_weights//2)
        ##################################
        overhead_tiles = []
        new_tm_tile = []
        self.tile_fraction_list = []
        for tile_idx in range(math.ceil(self.boolean_replicated.shape[0]/tile_size)):
            start_idx= tile_idx*tile_size
            end_idx = min((tile_idx+1)*tile_size,self.boolean_replicated.shape[0] )
            # print("Num_ones after interleaving, ",self.final_boolean_replicated[start_idx:end_idx, :,:].sum().item())
            overhead = self.boolean_replicated[start_idx:end_idx, :,:].sum().item()/(128*(128)*tile_size)
            overhead_tiles.append(overhead*speed_up_layer)
            self.tile_fraction_list.append(min(overhead*speed_up_layer, self.tm))
            if overhead*speed_up_layer > self.tm:
                new_tm_tile.append(self.tm-((self.tm-(overhead*speed_up_layer))/speed_up_layer))
            else:
                new_tm_tile.append(self.tm)
        # print(new_tm_tile) 
            # print(overhead)
        final_boolean = torch.zeros(hessian_score_booled.shape).to(self.device)
        for tile_idx in range(len(fault_tolerance_tile_layer)):
            bool_hessian_tile = hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:]>0 ##only places where the critical weights are marked. 
            final_boolean[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:] = top_10_percent_boolean(hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:],new_tm_tile[tile_idx],hessian_score_booled[tile_idx*tile_size:(tile_idx+1)*tile_size, :,:].numel()) # 
        
        self.final_boolean_replicated = replicate_last_dimension(final_boolean, self.quantize_weights//2)
        #modify the last dimension
        self.final_boolean_replicated = self.final_boolean_replicated.reshape(self.weight_shape).to(self.device)
        #############################################################################
        ##Update pstate tile_wise 
        pos_final_boolean_replicated = (self.final_boolean_replicated * pos_mask)
        neg_final_boolean_replicated = (self.final_boolean_replicated * neg_mask)
       
        self.comp_mask_pos = pos_final_boolean_replicated
        self.comp_mask_neg = neg_final_boolean_replicated
        
        
    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding
    def num_pad_col(self, source, target):
        crxb_col_fit = target/(self.quantize_weights/2)
        crxb_index = math.ceil((source * (self.quantize_weights/2)) / target)
        num_padding =int( crxb_index * target / (self.quantize_weights/2) - source)
        return crxb_index, num_padding


    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() / self.h_lvl * self.scaler_dw
    
            self.delta_x = input.abs().max() / self.h_lvl

        input_clip = F.hardtanh(input, min_val=-self.h_lvl * self.delta_x.item(),
                                max_val=self.h_lvl * self.delta_x.item())
        input_quan = quantize_input(
            input_clip, self.delta_x) * self.delta_v  # convert to voltage

        weight_quan = quantize_weight(self.weight, self.delta_w).transpose(0, 1)

        # 2. Perform the computation between input voltage and weight conductance
        # 2.1. skip the input unfold and weight flatten for fully-connected layers
        # 2.2. add padding
        weight_padded = F.pad(weight_quan, self.w_pad,
                              mode='constant', value=0)
        input_padded = F.pad(input_quan, self.input_pad,
                             mode='constant', value=0)
        # 2.3. reshape
        input_crxb = input_padded.view(
            input.shape[0], 1, self.crxb_row, self.crxb_size, 1)
        # weight_crxb = weight_padded.view(self.crxb_col, self.col_size,
        #                                  self.crxb_row, self.crxb_size).transpose(1, 2)
        
        weight_crxb = weight_padded.reshape(self.crxb_row,  self.crxb_col, self.crxb_size, self.col_size)
        # convert the floating point weight into conductance pair values
        G_crxb, error = self.w2g(weight_crxb)

      
        if self.enable_fec ==1: ##ERROR VALUES error = actual_value - sa00 == actual value
            # quantized_pos_weights = G_crxb[0] * self.final_boolean_replicated.float()
            # quantized_neg_weights = G_crxb[1] * self.final_boolean_replicated.float()
            quantized_pos_weights = (G_crxb[0]-self.Gmin)/self.delta_g * self.final_boolean_replicated.float()
            quantized_neg_weights = (G_crxb[1]-self.Gmin)/self.delta_g * self.final_boolean_replicated.float()

            G_crxb[0] = G_crxb[0] * torch.logical_not(self.final_boolean_replicated)
            G_crxb[1] = G_crxb[1] * torch.logical_not(self.final_boolean_replicated)

        # error[0] = error[0]*self.comp_mask_pos
        # error[1] = error[1]*self.comp_mask_neg
        
        cell_res = 2
        cellRange = 2**cell_res
        ## Get num_xb_per_tiles. 

        # 2.4. compute matrix multiplication
           # 2.4. compute matrix multiplication followed by reshapes
        no_of_tiles = math.ceil(G_crxb[0].shape[0] * G_crxb[0].shape[1]/ self.num_xbs_tile)
        ##create a list of temperatures for each tile

        temp_tile_info = torch.load(self.path,weights_only=True)
        temp_tile_layer = temp_tile_info[self.layer_count]
        original_shape = G_crxb[0].shape
        flattened_G_crxb_0 = G_crxb[0].reshape(-1, G_crxb[0].shape[2], G_crxb[0].shape[3])
        flattened_G_crxb_1 = G_crxb[1].reshape(-1, G_crxb[1].shape[2], G_crxb[1].shape[3])

    
        # this block is for introducing stochastic noise into ReRAM conductance
        if self.enable_stochastic_noise:
            
            ##linear decrease 
            m = -0.1  # slope 20 to 15 m=-0.05 -0.15(20 to 5) -0.1(20 to 10)
            b = 50  # interceptb 35(20 to 15) 65(20 to 5) 50(20 to 10)
           

            for tile_idx in range(no_of_tiles):
                start_idx = tile_idx * self.num_xbs_tile
                end_idx = min((tile_idx + 1) * self.num_xbs_tile, flattened_G_crxb_0.shape[0])

                # Apply temperature modification on the slice of the flattened array
                tile_temp = temp_tile_layer[tile_idx]
                if self.thermal_noise_mode == 1: ##20to15
                    flattened_G_crxb_0[start_idx:end_idx] *= (-0.05 * tile_temp + 35) / 20
                    flattened_G_crxb_1[start_idx:end_idx] *= (-0.05 * tile_temp + 35) / 20
                elif self.thermal_noise_mode == 2: ##20to10
                    flattened_G_crxb_0[start_idx:end_idx] *= (-0.1 * tile_temp + 50) / 20
                    flattened_G_crxb_1[start_idx:end_idx] *= (-0.1 * tile_temp + 50) / 20
                else:##20to5
                    flattened_G_crxb_0[start_idx:end_idx] *= 1/ (1 + 0.01 * (tile_temp - 300))
                    flattened_G_crxb_1[start_idx:end_idx] *= 1 / (1 + 0.01 * (tile_temp - 300))
            # Reshape G_crxb[0] back to its original shape
            G_crxb[0] = flattened_G_crxb_0.reshape(original_shape)
            G_crxb[1] = flattened_G_crxb_1.reshape(original_shape)
            ##clip the conductance values to Gmin and Gmax
            G_crxb[0] = torch.clamp(G_crxb[0], self.Gmin, self.Gmax)
            G_crxb[1] = torch.clamp(G_crxb[1], self.Gmin, self.Gmax)
        
        G_crxb[0] = ((G_crxb[0] - self.Gmin)/self.delta_g) ##convert to integer values
        G_crxb[1] = ((G_crxb[1] - self.Gmin)/self.delta_g)

        
        xb_pos = reduce_and_reshape_conv(G_crxb[0], self.quantize_weights//cell_res, cellRange, self.in_features, self.out_features)
        xb_neg = reduce_and_reshape_conv(G_crxb[1], self.quantize_weights//cell_res, cellRange, self.in_features, self.out_features)
        if self.enable_fec == 1:
            err_pos = reduce_and_reshape_conv(quantized_pos_weights, self.quantize_weights//cell_res, cellRange, self.in_features, self.out_features)
            err_neg = reduce_and_reshape_conv(quantized_neg_weights, self.quantize_weights//cell_res, cellRange, self.in_features, self.out_features)
            dequant_error = torch.clamp((err_pos - err_neg)*self.delta_w, min=-self.h_lvl_w*self.delta_w, max=self.h_lvl_w*self.delta_w)
            output_comp = F.linear(input, dequant_error, None) ##Compensation computation
        ##dequantize the conductance values
        dequant_weight = torch.clamp((xb_pos - xb_neg)*self.delta_w, min=-self.h_lvl_w*self.delta_w, max=self.h_lvl_w*self.delta_w)
        
        output = F.linear(input, dequant_weight, None)
        

        if self.enable_fec == 1:
           
            output += output_comp 

       

        if self.bias is not None:
            output += self.bias
        return output.to(self.device)
