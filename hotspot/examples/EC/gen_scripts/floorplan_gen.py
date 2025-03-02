import argparse

def generate_floorplan(Nx, Ny, tx, ty, tsvy, unit_name_tile, unit_name_tsv):
    floorplan = []
    tsvx = round(Nx * tx,5)  # Calculate the width of the TSV based on the number of tiles and the width of each tile
    
    # Generate the floorplan
    for y in range(Ny):
        # Calculate the bottom y-coordinate of the current row of tiles and TSV
        bottom_y_tiles = y * (ty + tsvy)
        bottom_y_tsv = bottom_y_tiles + ty
        
        # Add Nx tiles to the floorplan
        for i in range(Nx):
            left_x = i * tx  # Calculate the left x-coordinate of the current tile
            unit_name = f"{unit_name_tile}{y*Nx+i+1}"
            floorplan.append(f"{unit_name}\t{tx}\t{ty}\t{left_x}\t{bottom_y_tiles}")
        
        # Add one TSV above the tiles in the current row
        unit_name_tsv_instance = f"{unit_name_tsv}{y+1}"
        floorplan.append(f"{unit_name_tsv_instance}\t{tsvx}\t{tsvy}\t{0.0}\t{bottom_y_tsv}")

    # Save to a text file
    with open('floorplan.txt', 'w') as file:
        file.write("\n".join(floorplan))
    
    return "Floorplan generated and saved to 'floorplan.txt'."
def generate_2d_floorplan(Nx, Ny, tx, ty, unit_name_tile):
    floorplan = []
    
    # Generate the floorplan for 2D tiles only
    for y in range(Ny):
        # Calculate the bottom y-coordinate of the current row of tiles
        bottom_y_tiles = y * ty
        
        # Add Nx tiles to the floorplan in the current row
        for i in range(Nx):
            left_x = i * tx  # Calculate the left x-coordinate of the current tile
            unit_name = f"{unit_name_tile}{y*Nx+i+1}"
            floorplan.append(f"{unit_name}\t{tx}\t{ty}\t{left_x}\t{bottom_y_tiles}")

    # Save to a text file
    with open('2d_floorplan.txt', 'w') as file:
        file.write("\n".join(floorplan))
    
    return "Floorplan generated and saved to '2d_floorplan.txt'."

# # Example usage
# print(generate_2d_floorplan(5, 4, 100, 50, "Tile"))

def main():
    parser = argparse.ArgumentParser(description="Generate a floorplan file for a tile-TSV configuration.")
    parser.add_argument("--Nx", type=int, required=True, help="Number of tiles along the x-axis")
    parser.add_argument("--Ny", type=int, required=True, help="Number of layers along the y-axis")
    parser.add_argument("--tx", type=float, required=True, help="Width of each tile")
    parser.add_argument("--ty", type=float, required=True, help="Height of each tile")
    parser.add_argument("--tsvy", type=float, default=0.0003, help="Height of the TSV")
    parser.add_argument("--unit_name_tile", type=str, default="tile", help="Prefix for tile unit names")
    parser.add_argument("--unit_name_tsv", type=str, default="tsv", help="Prefix for TSV unit names")
    
    args = parser.parse_args()

    # Generate the floorplan using provided arguments
    result = generate_floorplan(args.Nx, args.Ny, args.tx, args.ty, args.tsvy, args.unit_name_tile, args.unit_name_tsv)
    result_2d = generate_2d_floorplan(args.Nx, args.Ny, args.tx, args.ty, args.unit_name_tile)
    print(result)

if __name__ == "__main__":
    main()
