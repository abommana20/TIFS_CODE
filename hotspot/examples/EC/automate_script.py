import subprocess
import argparse
import sys
import re
import matplotlib.pyplot as plt
from functions import *
def parse_args():
    parser = argparse.ArgumentParser(description="Automate running scripts with model configurations.")
    parser.add_argument('--model', type=str, default='resnet18', help='model to use')
    parser.add_argument('--tilepower', type=float, default=0.425, help='tile power') #including tile power plus write power. 
    parser.add_argument('--mempower', type=float, default=0.15, help='memory power')
    parser.add_argument('--crxb_size', type=int, default=128, help='crxb size')
    parser.add_argument('--cell_res', type=int, default=2, help='cell resolution')
    parser.add_argument('--weight_res', type=int, default=8, help='weight resolution')
    parser.add_argument('--pe_row', type=int, default=3, help='pe row')
    parser.add_argument('--pe_col', type=int, default=2, help='pe col')
    parser.add_argument('--array_row', type=int, default=2, help='array row')
    parser.add_argument('--array_col', type=int, default=2, help='array col')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--tx', type=float, default=1.0, help='tile width')
    parser.add_argument('--ty', type=float, default=1.0, help='tile height')
    parser.add_argument('--tsvy', type=float, default=1.0, help='tsv height')
    parser.add_argument('--mem_type', type=str, default='dram', help='memory type')
    return parser.parse_args()


def run_scripts(args):
    # Define the path to the Python generator script
    gen_script_path = './gen_scripts/ptrace_flp_gen.py'
    max_temp = {}
    fc = [0,1,2,3,4,5]
    dram_powers = [0, 0.226, 0.254, 0.447, 0.535, 0.624] ##from Cacti-6.0 (includinf read+leakage+ refresh power)
    sram_powers = [0,0.687,0.8483,1.0135,1.132,1.239]
    # Construct the command with all arguments
    for i in range(len(fc)):
        if args.mem_type == 'sram':
            python_command = [
                'python3', gen_script_path,
                '--tilepower', str(args.tilepower),
                '--mempower', str(sram_powers[i]),
                '--crxb_size', str(args.crxb_size),
                '--cell_res', str(args.cell_res),
                '--weight_res', str(args.weight_res),
                '--pe_row', str(args.pe_row),
                '--pe_col', str(args.pe_col),
                '--array_row', str(args.array_row),
                '--array_col', str(args.array_col),
                '--num_classes', str(args.num_classes),
                '--tx', str(args.tx),
                '--ty', str(args.ty),
                '--tsvy', str(args.tsvy),
                '--model', args.model
            ]
        else:
            python_command = [
                'python3', gen_script_path,
                '--tilepower', str(args.tilepower),
                '--mempower', str(dram_powers[i]),
                '--crxb_size', str(args.crxb_size),
                '--cell_res', str(args.cell_res),
                '--weight_res', str(args.weight_res),
                '--pe_row', str(args.pe_row),
                '--pe_col', str(args.pe_col),
                '--array_row', str(args.array_row),
                '--array_col', str(args.array_col),
                '--num_classes', str(args.num_classes),
                '--tx', str(args.tx),
                '--ty', str(args.ty),
                '--tsvy', str(args.tsvy),
                '--model', args.model
            ]

        try:
            # Execute the Python generator script
            print("Running Python generator script...")
            subprocess.run(python_command, check=True)
            print("Python script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Failed to run the Python script:", e)
            sys.exit(1)

        # Path to the shell script
        shell_script_path = './run.sh'
        shell_command = ['bash', shell_script_path, args.model]

        try:
            # Execute the shell script
            print(f"Running shell script for model {args.model}...")
            subprocess.run(shell_command, check=True)
            print("Shell script executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Failed to run the shell script:", e)
            sys.exit(1)

        # Extract the maximum temperature from the output file
        output_file = f'./{args.model}/outputs/example.steady'
        max_temp[f'{fc[i]}'] = extract_max_value(output_file)
        
        print("All scripts executed successfully.")
        print(f"temp traces are generated in the output directory {args.model}/outputs")
        print(f"Maximum temperature for fc_{fc[i]} is {max_temp[f'{fc[i]}']}")
    #save max_temp
    with open(f'./{args.model}/outputs/max_temp_{args.mem_type}.txt', 'w') as file:
        for key, value in max_temp.items():
            file.write(f"{key}: {value}\n")

    ##conducatance variation. G = (1/R) = (1/Resistivity) = (1/(Resistivity_0*(1+alpha*(T-T_0)))) where G_0=3.9e-08, alpha = 0.03, T_0 = 300
    G0 = 3.9e-08
    alpha = 0.03
    T0 = 300
    G = {}
    G_percentage_drop = {}
    for key, value in max_temp.items():
        G[key] = G0/(1+alpha*(value-T0))
        G_percentage_drop[key] = (G0 - G[key])/G0 * 100
    # print(G)
    # print(G_percentage_drop)
    # print(max_temp)

    #save G
    with open(f'./{args.model}/outputs/conductance_{args.mem_type}.txt', 'w') as file:
        for key, value in G.items():
            file.write(f"{key}: {value}\n")
    #plot G_percentage vs fc
    plot_values(max_temp, save_filename=f'./{args.model}/outputs/max_temp_{args.mem_type}.png')
    plot_values_and_distances(G_percentage_drop, save_filename=f'./{args.model}/outputs/conductance_{args.mem_type}.png')


if __name__ == "__main__":
    args = parse_args()
    run_scripts(args)
