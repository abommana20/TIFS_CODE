#!/usr/bin/env bash

# Usage: ./run.sh <rows> <cols> <mode>
# Example: ./run.sh 16 16 normal
# Example: ./run.sh 16 16 max

# if [ $# -ne 2 ]; then
#     echo "Usage: ./run.sh <rows> <cols> <mode>"
#     echo "Modes available: normal, max"
#     exit 1
# fi

rows=16
cols=16

# Set the appropriate power trace file based on mode

ptrace_file="power.ptrace"
heatmap_output="heatmap_max.png"
transient_output="temp_trace_max.txt"


# Clean up previous simulation files
rm -f *.init
rm -f outputs/*

# Ensure the output directory exists
mkdir -p outputs
mkdir -p results

echo "RUNNING HOTSPOT with grid $rows x $cols using mode $mode"
../../hotspot -c example.config -f reramcore.flp -p $ptrace_file -materials_file example.materials -model_type grid  -grid_rows $rows -grid_cols $cols -grid_steady_file outputs/example.grid.steady -steady_file outputs/example.steady

cp outputs/example.steady example.init

../../hotspot -c example.config -f reramcore.flp -p $ptrace_file -materials_file example.materials -init_file example.init -model_type grid  -o outputs/example.transient -grid_transient_file outputs/example.grid.ttrace

# Process temperature files and generate heatmap
echo "Average temperatures across active regions:"
head -n 2 outputs/example.steady

python3 ../../scripts/split_grid_steady.py outputs/example.grid.steady 4 $rows $cols ./
python3 ../../scripts/grid_thermal_map.py reramcore.flp outputs/example_layer0.grid.steady $rows $cols outputs/$heatmap_output

# Copy relevant outputs to results directory
cp outputs/$heatmap_output ./results/$heatmap_output
cp outputs/example.transient ./results/$transient_output
