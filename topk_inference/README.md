# Experiment Setup and Running

All executable files are located in the **`benchmark`** directory.  
All configuration files are located in the **`config`** directory.  
All custom **PytorX** files (with crossbar-based layers and mapping) are located in **`benchmark/torx/module`**.

---

## Parameters

> **Note**: The `parser.add_argument` lines below are shown to illustrate which parameters are available and how they are defined.  

parser.add_argument('--model', default='resnet18', type=str, help='model name')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--cuda', type=str, default='cuda:0', help='dataset to use')

parser.add_argument('--thermal_mode', type=int, default=2)
1. Thermal Noise mode: There are three modes of thermal noise injection.
 Example: The on/off ratio drop from 20 to 10 when thermal_mode is set to 2 and another from 20 to 15 when thermal_mode is set to 1.
In our paper, we model 20 to 10.

parser.add_argument('--tm', type=float, default=0.1, help="max fault tolerance in tile")
2. The maximum amount of fault tolerance in each tile (e.g., 10%).

parser.add_argument('--path', type=str, default='.', help='path to save the traces')
3. This is the path where the temperature of each tile (after the attack) is stored.
   Based on these new temperatures, the conductance drift is injected.

parser.add_argument('--trained_model_path', type=str, default='trained_models', help='path to trained model')
4. Path to the trained models.

parser.add_argument('--topk', type=float, default=0.1, help='percentage of protection')
5. The percentage of critical weights that are protected.

parser.add_argument('--enable_fec', type=int, default=0, help='error correction')
6. Enables proposed error correction.
# README

## Overview
This repository implements experiments to evaluate the recovery of accuracy under worst-case adversarial attacks, energy overhead, and system speedup. The provided modules enable temperature drift injection, weight-to-conductance mapping, and energy evaluations using data from NeuroSim and power trace generation. Follow the instructions below to set up and run the experiments correctly.

## Main Files

1. **main_temp.py**  
   - **Purpose:**  
     This script measures the accuracy recovery on worst-case adversarial attacks (refer to Figures 10 and 11) by applying the proposed countermeasure. It generates the results shown in Tables 4, 5, and 6.
   - **Instructions:**  
     - Copy `layer_topk.py` to `layer.py` before running.  
     - Parse the worst temperature traces.  
     - Enable the `enable_fec` signal and observe the recovered accuracy.

2. **main_acc.py**  
   - **Purpose:**  
     Used exclusively for experiments in the `power_trace_gen` directory.
   - **Instructions:**  
     - Ensure that `layer_topk.py` is copied to `layer.py` prior to execution.

3. **main_speedup.py**  
   - **Purpose:**  
     Conducts experiments related to system speed up as discussed in the paper's discussion section.
   - **Instructions:**  
     - Copy `layer_speedup.py` to `layer.py` before running.

## Layer Files
*Located in the `benchmark/torx/module` directory:*

- **layer_speedup.py**
- **layer_topk.py**

**Usage:**  
When running a particular experiment, always copy the corresponding layer file (`layer_topk.py` or `layer_speedup.py`) to `layer.py`, as `layer.py` is the main file invoked by the main scripts.

In these layer files, two classes are implemented:
- **Conv:** A crossbar-based convolution layer.
- **Linear:** A crossbar-based linear layer.

Temperature drift is injected at the tile level based on temperature values from power trace generation files (see lines 410–3430 and 749–770 in the layer files).

**Additional Module:**  
- **w2g.py:**  
  This file maps the weights to conductance values.

## Energy Evaluation
The energy evaluation framework includes the following steps:

1. **Protection Analysis:**  
   The energy overhead is computed based on the protection level in each tile, along with power and latency measurements obtained from NeuroSim.

2. **Calculation of Energy Overhead:**  
   Functions defined in the `energy_eval` directory are used to compute the energy overhead. These functions are called within `main_temp.py` and `main_speedup.py`.

3. **Latency Measurements:**  
   - Latency values are derived from NeuroSim.  
   - The required network CSV files are generated using `csv_generate_func.py` located in the benchmark directory.  
   - Please refer to NeuroSim 1.4 for extracting the latency data.

## Configuration Settings
*The configuration settings are located in `config/uniform`. Do not change these values.*

```json
{
    "batch_size": 1000,
    "test_batch_size": 100,
    "epochs": 20,
    "lr": 0.5,
    "momentum": 0.5,
    "no_cuda": false,
    "seed": 1,
    "log_interval": 10,
    "save_model": false,
    "crxb_size": 128,
    "vdd": 3.3,
    "gwire": 0.375,
    "gload": 0.25,
    "gmax": 0.000333,
    "gmin": 0.00001665,
    "ir_drop": false,
    "scaler_dw": 1,
    "test": false,
    "enable_noise": true,
    "enable_SAF": false,
    "enable_ec_SAF": false,
    "freq": 10000000.0,
    "temp": 300,
    "enable_ec_SAF_msb": false,
    "quantize": 8,
    "adc_resolution": 16,
    "input_quantize": 8,
    "fault_rate": 0.0,
    "fault_dist": "uniform",
    "sa00_rate": 0.25,
    "sa01_rate": 0.25,
    "sa10_rate": 0.25,
    "sa11_rate": 0.25
}
