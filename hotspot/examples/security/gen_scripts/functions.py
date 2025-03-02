import math

def generate_tile_power(crxb_size, cell_res, weight_res, layer_name, layer_size, pe_row, pe_col, array_row, array_col, tile_dict, tile_power, tile_start_idx=0):
    
    if layer_name == "conv":
        # Calculate required number of crossbars (Xbars)
        weight_shape = layer_size  # Assuming weight_shape is a tuple (out_channels, in_channels, kernel_height, kernel_width)
        num_xbs_along_col = math.ceil(weight_shape[0] * (weight_res / cell_res)/crxb_size)
        num_xbs_along_row = math.ceil(weight_shape[1] * weight_shape[2] * weight_shape[3] / crxb_size)
        num_xbs_in_tile_row = pe_row * array_row
        num_xbs_in_tile_col = pe_col * array_col
        num_tile_row = math.ceil(num_xbs_along_row / num_xbs_in_tile_row)
        num_tile_col = math.ceil(num_xbs_along_col / num_xbs_in_tile_col)
        num_tiles = num_tile_row * num_tile_col
        # print("num_tile",num_tiles)
        # Find the utilization rate of each tile
        num_elements_in_xb_in_tile = crxb_size * (crxb_size/(weight_res / cell_res))*pe_col*pe_row*array_row*array_col    
        num_elements_layer = weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3] 
        total_num_elements = num_elements_layer  # Store the total number of elements to calculate the last tile utilization
        tiles_utilization = []
        num_elements_along_row = weight_shape[1] * weight_shape[2] * weight_shape[3]
        num_elements_along_col = weight_shape[0] * (weight_res / cell_res)
        num_elements_in_tile_along_row = pe_row * array_row * crxb_size
        num_elements_in_tile_along_col = pe_col * array_col * crxb_size

        for i in range(num_tile_row):
            for j in range(num_tile_col):
                if num_elements_along_row > 0 and num_elements_along_col > 0:
                    if num_elements_along_row * num_elements_along_col > num_elements_in_tile_along_row * num_elements_in_tile_along_col:
                        tile_utilization = 1.0
                        num_elements_along_col = num_elements_along_col - num_elements_in_tile_along_col
                    else:
                        tile_utilization = (num_elements_along_row * num_elements_along_col) / (num_elements_in_tile_along_row * num_elements_in_tile_along_col)
                        # print('tile_utilization',tile_utilization)
                    tiles_utilization.append(tile_utilization)
            num_elements_along_row = num_elements_along_row - num_elements_in_tile_along_row
            num_elements_along_col = weight_shape[0] * (weight_res / cell_res)
        
        # print("len_tile_util",len(tiles_utilization))
        # Calculate power for each tile based on utilization
        for i, utilization in enumerate(tiles_utilization):
            tile_dict[f"tile{tile_start_idx+i+1}"] = round(utilization * tile_power, 5)

    elif layer_name == "fc":
        # Similar calculations for fully connected layers
        weight_shape = layer_size  # Assuming weight_shape is a tuple (out_features, in_features)
        num_xbs_along_col = math.ceil(weight_shape[0] * weight_res / cell_res/crxb_size)
        num_xbs_along_row = math.ceil(weight_shape[1] / crxb_size)
        num_xbs_in_tile_row = pe_row * array_row
        num_xbs_in_tile_col = pe_col * array_col
        num_tile_row = math.ceil(num_xbs_along_row / num_xbs_in_tile_row)
        num_tile_col = math.ceil(num_xbs_along_col / num_xbs_in_tile_col)
        num_tiles = num_tile_row * num_tile_col
        # num_tiles = num_tile_row * num_tile_col
        # print(num_tiles)
        tiles_utilization = []
        num_elements_along_row = weight_shape[1] 
        num_elements_along_col = weight_shape[0] * (weight_res / cell_res)
        num_elements_in_tile_along_row = pe_row * array_row * crxb_size
        num_elements_in_tile_along_col = pe_col * array_col * crxb_size

        for i in range(num_tile_row):
            for j in range(num_tile_col):
                if num_elements_along_row > 0 and num_elements_along_col > 0:
                    if num_elements_along_row * num_elements_along_col > num_elements_in_tile_along_row * num_elements_in_tile_along_col:
                        tile_utilization = 1.0
                        num_elements_along_col = num_elements_along_col - num_elements_in_tile_along_col
                    else:
                        tile_utilization = (num_elements_along_row * num_elements_along_col) / (num_elements_in_tile_along_row * num_elements_in_tile_along_col)
                        # print('tile_utilization',tile_utilization)
                    tiles_utilization.append(tile_utilization)
            num_elements_along_row = num_elements_along_row - num_elements_in_tile_along_row
            num_elements_along_col = weight_shape[0] * (weight_res / cell_res)

        # Calculate power for each tile based on utilization
        for i, utilization in enumerate(tiles_utilization):
            tile_dict[f"tile{tile_start_idx+i+1}"] = round(utilization * tile_power, 5)

    return num_tiles


def generate_floorplan(Nx, Ny, tx, ty, tsvy, unit_name_tile, unit_name_tsv, file_name, model_name):
    floorplan = []
    tsvx = round(Nx * tx,6)  # Calculate the width of the TSV based on the number of tiles and the width of each tile
    
    # Generate the floorplan
    for y in range(Ny):
        # Calculate the bottom y-coordinate of the current row of tiles and TSV
        bottom_y_tiles = round(y * (ty + tsvy),6)
        bottom_y_tsv = round(bottom_y_tiles + ty,6)
        
        # Add Nx tiles to the floorplan
        for i in range(Nx):
            left_x = round(i * tx,6)  # Calculate the left x-coordinate of the current tile
            unit_name = f"{unit_name_tile}{y*Nx+i+1}"
            floorplan.append(f"{unit_name}\t{tx}\t{ty}\t{left_x}\t{bottom_y_tiles}")
        
        # Add one TSV above the tiles in the current row
        unit_name_tsv_instance = f"{unit_name_tsv}{y+1}"
        floorplan.append(f"{unit_name_tsv_instance}\t{tsvx}\t{tsvy}\t{0.0}\t{bottom_y_tsv}")

    # Save to a text file
    with open(f'{model_name}/{file_name}.flp', 'w') as file:
        file.write("\n".join(floorplan))
        file.write("\n")  # Adds an empty line at the end of the file
    
    return "Floorplan generated and saved to 'floorplan.txt'."


def pad_tile(d, k):
    # Extract suffixes where keys match 'tile{number}' and compute the maximum
    suffixes = [int(key[4:]) for key in d if key.startswith('tile')]
    highest_key = max(suffixes) if suffixes else 0

    # Calculate how many dummy keys are needed
    dummy_keys_needed = (k - (len(d) % k)) % k

    # Add dummy keys continuing from the last key number
    for i in range(1, dummy_keys_needed + 1):
        d[f"tile{highest_key + i}"] = 0

    return d

def write_dict_to_file(d, filename):
    # Open the file for writing
    with open(filename, 'w') as file:
        # Write the keys on the first row separated by tabs
        file.write('\t'.join(d.keys()) + '\n')
        # Write the values on the second row, converting all values to strings and separating by tabs
        file.write('\t'.join(map(str, d.values())) + '\n')
        file.write('\t'.join(map(str, d.values())) + '\n')
        file.write('\t'.join(map(str, d.values())) + '\n')
        file.write('\t'.join(map(str, d.values())) + '\n')
        file.write('\t'.join(map(str, d.values())) + '\n')
        # file.write("\n")  # Adds an empty line at the end of the file

def add_keys_based_on_name(d, base_name,len, tile_power, num_tiles):
    # Count the current number of keys
    current_key_count = len
    
    # Add additional keys based on the base name and the current key count
    for i in range(1, current_key_count + 1):
        new_key = f"{base_name}{i}"  # Construct the new key name
        if i <= num_tiles:
            d[new_key] = tile_power
        else:
            d[new_key] = 0
    

    return d

def create_dict_from_files(file_list, input_dict):
    output_dict = {}
    
    # Iterate through each file
    for file_name in file_list:
        with open(file_name, 'r') as file:
            # Read each line from the file
            for line in file:
                # Get the first word from the line
                first_word = line.split()[0]
                # If the first word is in input_dict, add to output_dict
                if first_word in input_dict:
                    output_dict[first_word] = input_dict[first_word]
    
    return output_dict
