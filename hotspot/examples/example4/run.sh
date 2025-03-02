#! /usr/bin/env bash


rows=8
cols=8

# Set the appropriate power trace file based on mode

ptrace_file="ev6_3D.ptrace"
heatmap_output1="heatmap_tier1.png"
heatmap_output2="heatmap_tier2.png"
heatmap_output3="heatmap_core.png"
transient_output="temp_trace_max.txt"
# Remove files from previous simulations
rm -f *.init
rm -f outputs/*

# Create outputs directory if it doesn't exist
mkdir outputs

# The modeling of a 2-D chip's C4 pad array or a 3-D chip's thermal vias
# requires support for heterogeneous materials within one layer.
# Thanks to Prof. Ayse Coskun's research team in Boston University,
# this feature has been supported in HotSpot since version 6.0.
# To enable this feature in simulation, the command line option
# `-detailed_3D on` must be set to `on`. Currently, heterogeneous layers
# can only be modeled with `-model_type grid` and an LCF file specified
../../hotspot -c example.config -p $ptrace_file -grid_layer_file ev6_3D.lcf -model_type grid -detailed_3D on -grid_steady_file outputs/example.grid.steady -steady_file outputs/example.steady

cp outputs/example.steady example.init

../../hotspot -c example.config -p ev6_3D.ptrace -grid_layer_file ev6_3D.lcf -init_file example.init -model_type grid -detailed_3D on -o outputs/example.transient -grid_transient_file outputs/example.grid.ttrace


python3 ../../scripts/split_grid_steady.py outputs/example.grid.steady 8 $rows $cols ./

python3 ../../scripts/grid_thermal_map.py ev6_3D_cache_1.flp outputs/example_layer0.grid.steady $rows $cols outputs/$heatmap_output1

python3 ../../scripts/grid_thermal_map.py ev6_3D_cache_2.flp outputs/example_layer2.grid.steady $rows $cols outputs/$heatmap_output2

python3 ../../scripts/grid_thermal_map.py ev6_3D_core_layer.flp outputs/example_layer4.grid.steady $rows $cols outputs/$heatmap_output3