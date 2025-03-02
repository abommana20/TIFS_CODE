#! /usr/bin/env bash
if [ $# -ne 2 ]; then
    echo "Usage: ./run.sh main_path path mode"
    exit 1
fi

path=$1
mode=$2

rows=8
cols=8

# Set the appropriate power trace file based on mode

ptrace_file="power.ptrace"
heatmap_output1="${mode}_heatmap_tier1.png"
heatmap_output2="${mode}_heatmap_tier2.png"
# Remove files from previous simulations
rm -f *.init
rm -f outputs/*

# Create outputs directory if it doesn't exist
mkdir -p outputs

# The modeling of a 2-D chip's C4 pad array or a 3-D chip's thermal vias
# requires support for heterogeneous materials within one layer.
# Thanks to Prof. Ayse Coskun's research team in Boston University,
# this feature has been supported in HotSpot since version 6.0.
# To enable this feature in simulation, the command line option
# `-detailed_3D on` must be set to `on`. Currently, heterogeneous layers
# can only be modeled with `-model_type grid` and an LCF file specified
../../hotspot -c example.config -p $ptrace_file -grid_layer_file 3d.lcf -model_type grid -detailed_3D on -grid_steady_file outputs/example.grid.steady -steady_file outputs/example.steady

cp outputs/example.steady example.init

../../hotspot -c example.config -p $ptrace_file -grid_layer_file 3d.lcf -init_file example.init -model_type grid -detailed_3D on -o outputs/example.transient -grid_transient_file outputs/example.grid.ttrace


python3 ../../scripts/split_grid_steady.py outputs/example.grid.steady 4 $rows $cols ./

python3 ../../scripts/grid_thermal_map.py reramcore1.flp outputs/example_layer0.grid.steady $rows $cols outputs/$heatmap_output1

python3 ../../scripts/grid_thermal_map.py reramcore2.flp outputs/example_layer1.grid.steady $rows $cols outputs/$heatmap_output2

# cp -r outputs/ $path
# rm -rf outputs