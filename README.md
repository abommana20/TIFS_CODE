# TIFS_CODE

# Power Trace and Accuracy Evaluation Framework

This repository contains multiple modules that collectively enable power trace generation, sensitivity analysis, and inference accuracy evaluation under adversarial conditions. Each module (except **Hotspot**) includes its own README file with detailed setup instructions, implementation details, and explanations of the files contained within.

## Repository Structure

1. **Power_trace_gen**
   - **Purpose:** Generates power traces for different attacks and floorplan configurations.
   - **Details:** For installation and usage instructions, please refer to the README within the directory.

2. **Hessian Sensitivity**
   - **Purpose:** Calculates the Hessian scores at each tile and crossbar level.
   - **Details:** Detailed setup and implementation guidelines are provided in the directory’s README file.

3. **Hotspot**
   - **Purpose:** Utilized by **Power_trace_gen** during execution. No independent experiments are required.
   - **Details:** Although this directory contains a README, it is mainly intended for informational support.

4. **Topk_inference**
   - **Purpose:** Implements an accuracy framework to evaluate system performance under various adversarial conditions.
   - **Details:** For complete setup instructions and file descriptions, please see the README provided within this directory.

## Getting Started

To begin working with this project:
- Review the README files in each module (except **Hotspot**) for specific setup instructions.
- Ensure all dependencies and configurations are properly installed as detailed in the individual READMEs.

## References

```bibtex
@inproceedings{He_Lin_Ewetz_Yuan_Fan_2019,
  address={New York, NY, USA},
  series={DAC ’19},
  title={Noise Injection Adaption: End-to-End ReRAM Crossbar Non-ideal Effect Adaption for Neural Network Mapping},
  ISBN={978-1-4503-6725-7},
  author={He, Zhezhi and Lin, Jie and Ewetz, Rickard and Yuan, Jiann-Shiun and Fan, Deliang},
  year={2019},
  month=jun,
  pages={1–6},
  collection={DAC ’19}
}

@ARTICLE{hotspot,
  author={Wei Huang and Ghosh, S. and Velusamy, S. and Sankaranarayanan, K. and Skadron, K. and Stan, M.R.},
  journal={IEEE Transactions on Very Large Scale Integration (VLSI) Systems},
  title={HotSpot: A Compact Thermal Modeling Methodology for Early-Stage VLSI Design},
  year={2006},
  volume={14},
  number={5},
  pages={501-513}
}