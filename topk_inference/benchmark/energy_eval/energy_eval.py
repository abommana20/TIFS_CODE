import torch
import numpy as np

def energy_eval(
    tile_fraction_protection,
    model_name,
    dataset,
    tile_info_path="../tile_info",
    crossbar_info_path="../crossbar_layers",
    latency_path="../latency",
    baseline_energy_path="baseline_energy.pt",  # Path to the already-saved baseline energy dict
    max_storage=0.12,
    error_bits=4
):
    """
    Computes EC overhead for a given model and dataset. 
    Returns the EC overhead (in nJ) and overhead percentage relative to the baseline.

    Parameters:
    -----------
    tile_fraction_protection : list or array-like
        A list of tile-level fractions (0 to 1) indicating how much protection to apply at each tile.
        Must have length equal to sum of tile_info for the model.
    model_name : str
        Name of the model (e.g., "resnet18", "vgg16", etc.)
    dataset : str
        Name of the dataset (e.g., "cifar10")
    tile_info_path : str
        Directory containing the tile_info_{model_name}_{dataset}.pt files
    crossbar_info_path : str
        Directory containing the no_crossbar_layer_{model_name}_{dataset}.pt files
    latency_path : str
        Directory containing the latency_{model_name}_128.pt files
    baseline_energy_path : str
        File containing the baseline energy dictionary, keyed by model_name
    max_storage : float
        The fraction of some 'maximum' memory capacity used in your experiment (the original code uses 0.12).
        If you have multiple scenarios for different max_storages, adapt accordingly.
    error_bits : int
        Error bits used in the model (default is 4).

    Returns:
    --------
    ec_energy : float
        Total EC overhead energy in nJ
    overhead_percentage : float
        Overhead percentage relative to the baseline energy
    """

    # 1. Load the baseline energies (already computed & saved in baseline_energy.pt)
    baseline_energy_dict = torch.load(baseline_energy_path, weights_only=True)
    if model_name not in baseline_energy_dict:
        raise KeyError(f"Model '{model_name}' not found in baseline_energy dictionary.")
    baseline_energy = baseline_energy_dict[model_name]  # in nJ

    # 2. Compute error-storage-based power overhead
    #    (you can adapt these constants as needed, 
    #     but here we follow the original snippet)
    error_storage = max_storage * 128 * 128 * 96 * error_bits / 1024 / 8  # in KB
    power_ec   = 0.21e-3            # W  (i.e., J/ms) from code snippet
    power_dram = error_storage * 20.7 / 64 * 1e-3  # also W
    print(power_dram)

    # 3. Load crossbar info (no_crossbar_layer_{model_name}_{dataset}.pt)
    crossbar_info = torch.load(f"{crossbar_info_path}/no_crossbar_layer_{model_name}_{dataset}.pt", 
                               weights_only=True)
    crosbar_list = list(crossbar_info.values())  # same ordering as in code

    # 4. Load tile info and get total number of tiles
    tile_info = torch.load(f"{tile_info_path}/tile_info_{model_name}_{dataset}.pt", weights_only=True)
    total_tiles = sum(tile_info)
    if len(tile_fraction_protection) != total_tiles:
        raise ValueError(
            f"Length of tile_fraction_protection ({len(tile_fraction_protection)}) "
            f"must match total_tiles ({total_tiles})."
        )

    # 5. Load latency info and convert to list
    latency_info = torch.load(f"{latency_path}/latency_{model_name}_128.pt", weights_only=True)
    latency_list = list(latency_info.values())

    # 6. Compute DRAM overhead per layer, using tile_fraction_protection as needed
    tile_end_idx = 0
    power_cummulative_dram = []
    for layer in range(len(tile_info)):
        tile_start_idx = tile_end_idx
        tile_end_idx  += tile_info[layer]
        tile_fractions = tile_fraction_protection[tile_start_idx:tile_end_idx]
        
        # Original code example did not truly incorporate `tile_fractions` in the DRAM 
        # power calculation, but you can adapt as needed. Here we show the simplest
        # direct approach matching the snippet structure:
        pdram_layer = 0
        for frac in tile_fractions:
            # e.g., incorporate fraction here if you want fraction-based overhead:
            # pdram_layer += power_dram * max_storage * frac
            #
            # The snippet used a constant factor (0.01) for each tile:
            pdram_layer += power_dram * max_storage * frac
        
        power_cummulative_dram.append(pdram_layer)

    # 7. Compute total EC power per layer: crossbar + DRAM overhead
    power_total_ec = []
    for layer in range(len(tile_info)):
        # crossbar_list[layer] is the number of crossbars in that layer
        pec_layer = crosbar_list[layer] * power_ec + power_cummulative_dram[layer]
        power_total_ec.append(pec_layer)

    # 8. Compute total EC energy overhead across all layers
    ec_energy_list = []
    for layer in range(len(tile_info)):
        # Energy = Power * Time
        ec_energy_list.append(power_total_ec[layer] * latency_list[layer])
    ec_energy = sum(ec_energy_list)  # in nJ, if your power is in nJ/ms, be sure units are consistent

    # 9. Compute overhead percentage
    overhead_percentage = (ec_energy / baseline_energy) 

    return ec_energy, overhead_percentage


if __name__ == "__main__":
    # Example usage
    models = ["resnet18", "vgg16", "vgg8","densenet40", "densenet121", "alexnet"]
    dataset = "cifar10"

    # Suppose we want overhead for resnet18 with random tile_fraction_protection:
    # (In a real scenario, you'd define or compute these fractions carefully.)
    for model_name in models:
        tile_info = torch.load("tile_model_info.pt", weights_only=True)

        tile_fraction_protection_example = np.random.rand(tile_info[model_name]).tolist()  # e.g. 1000 tiles, just a placeholder

        ec_energy, overhead_pct = compute_ec_overhead(
            tile_fraction_protection=tile_fraction_protection_example,
            model_name=model_name,
            dataset=dataset,
            tile_info_path="../tile_info",
            crossbar_info_path="../crossbar_layers",
            latency_path="../latency",
            baseline_energy_path="baseline_energy.pt",  # adjust if needed
            max_storage=0.1,
            error_bits=16
        )
        print("model_name", model_name)
        print(f"EC Overhead (nJ): {ec_energy}")
        print(f"Overhead Percentage: {overhead_pct:.2f}%")