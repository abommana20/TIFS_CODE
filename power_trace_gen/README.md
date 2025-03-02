# Experiment Setup and Run

This README provides instructions for setting up and running the experiments, including creating floor plans, running simulations, and obtaining accuracy results. All output files (e.g., accuracy results) are stored in the `results` directory.

---

## 1. Experiment Setup

- **Step 1:** Run the create floor plan functions according to the required configurations. (Refer to the _Create Floorplan Files_ section for details.)
- **Step 2:** Run the main file corresponding to each floorplan function.
- **Step 3:** Accuracy results will be stored in the `results` directory.

---

## 2. Parameters

| Parameter      | Value                                      | Description                                                         |
|----------------|--------------------------------------------|---------------------------------------------------------------------|
| `model_names`  | `['vgg8']`                                 | Choose a model name: `densent40`, `vgg8`, `resnet18`, `resnet18`                 |
| `dataset_name` | `'cifar10'`                                | Name of the dataset                                                 |
| `save_path`    | `"./trace_data"`                           | Directory to save the files                                         |
| `attack_names` | `['input_based_attack', 'weight_based_attack', 'weight_input_based_attack']` | Three types of attacks considered                                   |

---
## 3. Create Floorplan Files for Hotspot

Each floorplan creation function generates the necessary power profiles for the specified configurations and attack patterns.

### 3.1. `create_floorplan_special` (2D Configuration)
In this file, we create the floorplan configuration where the attacker tiles surround the victim tenant tiles. You can also vary the number of levels (grid length as used in the paper) to generate the power traces and floorplan file for hotspot simulations.

### 3.2. `create_floorplan_variable` (2D Configuration)
This file creates a floorplan configuration where the number of attacker tiles is varied on the chip relative to the tenant’s tiles. The attacker tiles can range from 10% to 90%, generating the corresponding power traces and floorplan file for hotspot simulations.

### 3.3. `create_floorplan` (2D Configuration)
This is a subset of `create_floorplan_variable` that performs the same function for a limited set of configurations (20, 50, 80, and special).

For the first three files, three attack patterns are explored:  
`attack_names` → `['input_based_attack', 'weight_based_attack', 'weight_input_based_attack']`  
These correspond to the A1, A2, and A3 attack patterns defined in our paper.

### 3.4. `create_flp_2d_bs2` (2D Configuration)
- **Pre-requisite:** Run `extract_tile_power_synthetic.py` before executing this file.  
In this file, we generate traces for the baseline configuration using a floorplan similar to the one defined in file 3.3. The baseline is defined as the scenario where different neural network models from multiple tenants are mapped to the ReRAM CiM accelerator. This file also simulates the A1_2 attack pattern defined in the paper and generates the corresponding power traces.

### 3.5. `create_floor3d` (3D Configuration)
In this file, we generate the power traces for a 3D configuration where the attacker tenant is allocated either the bottom or top tier.

### 3.6. `create_flp_3d_bs2` (3D Configuration)
- **Pre-requisite:** Run `extract_tile_power_synthetic.py` before executing this file.  
Similar to file 3.4, this file generates traces for both the baseline and the A1_2 attack pattern in a 3D configuration.

---

## 4. Main Files

These scripts run the hotspot simulations, extract new temperature profiles based on different attacks, and execute the accuracy framework. All main files invoke the hotspot and accuracy frameworks to generate temperature traces and estimated accuracy corresponding to the power traces generated in **Section 3**.

### 4.1. `main_spe`
- **Parameters:**
  - `configs = [1, 2, 3, 4, 5, 6, 7, 9]` (grid length)
  - `tile_config = 'special'` (fixed for this file)
  - Corresponds to file *3.1*

### 4.2. `main_var`
- **Parameters:**
  - `configs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]` (variation in the number of tiles allocated to the attacker relative to the tenant’s tiles; e.g., 0.1 represents 10%)
  - Corresponds to file *3.2*

### 4.3. `main`
- **Parameters:**
  - `configs = ['20_80', '50_50', '80_20', 'special']`
  - Corresponds to file *3.3*

### 4.4. `main3d`
- **Parameters:**
  - `tile_configs = ["50_50"]` (other options: `'20_80', '50_50', '80_20'`)
  - `flp_configs = ["top_tim", "top_wo_tim", "bottom_tim", "bottom_wo_tim"]`
  - Corresponds to file *3.5*

### 4.5. `main_bs2_2d`
- Creates a baseline file for normal multi-DNN inferencing.
- **Reference:** See the A1_2 attack pattern and baseline experiments.
- **Note:** Uses the same parameters as `main.py`.
- Corresponds to file *3.4*

### 4.6. `main_3d_bs2`
- **Reference:** See the A1_2 attack pattern and baseline experiments.
- **Note:** Uses the same parameters as `main3d.py`.
- Corresponds to file *3.6*

---

## 5. Generate Floorplan Files

- **Script:** `floor_paln_gen.py`
- **Purpose:** Generates the floorplan file required for hotspot simulations.
- **Usage:** This script is invoked within each floorplan creation function.

---

## 6. Extract Tile Information and Related Data

These scripts generate power traces based on the attack and victim models and are invoked within the floorplan creation functions.

1. **`extract_input_weight_traces`**  
   - *Usage:* Called in `main_bs2_2d` and `main_3d_bs2`.

2. **`extract_tile_info`**  
   - *Usage:* Called in all floorplan creation functions.

3. **`extract_tile_power_synthetic`**  
   - *Usage:* Run before executing `create_flp_2d_bs2` and `create_flp_3d_bs2`.

4. **`extract_only_tile_infor.py`**  
   - *Usage:* Extracts only tile information (without power data). **Run this file first.**

---

## 7. Extract Accuracy and Analyze Results

These scripts analyze the simulation results to generate figures and performance metrics.

1. **`plot_results.ipynb`**  
   - *Usage:* Generates Figures 10 and 11 in the paper (showing temperature changes and accuracy drop).

2. **`plot_results_var_surr.ipynb`**  
   - *Usage:* Generates Figure 8 in the paper.

---

## 8. Information You Will Get

1. Power traces for each attack pattern and baseline considered in this work (**Run the floorplan creation functions**).
2. Hotspot-simulated temperatures after the attack and the corresponding accuracy results (**Run the main files**).
3. Plot functions to extract data from each results folder and analyze the outcomes.

---

