# TIFS_CODE

# Power Trace and Accuracy Evaluation Framework

This repository contains multiple modules that collectively enable power trace generation, sensitivity analysis, and inference accuracy evaluation under adversarial conditions. Each module includes its own README file with detailed setup instructions, implementation details, and explanations of the files contained within.

## Repository Structure

1. **Power_trace_gen**
   - **Purpose:** Generates power traces for different attacks and floorplan configurations.
   - **Details:** For installation and usage instructions, please refer to the README within the directory.

2. **Hessian Sensitivity**
   - **Purpose:** Calculates the Hessian scores at each crossbar level.
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

## Experiments Workflow

1. **Model Training**  
   First, train the models as per your experimental requirements. *Copy all trained models directory to topk_inference power_trace_gen hessian_sensitivity*

2. **Hessian Score Generation**  
   Generate the Hessian scores by navigating to the **Hessian Sensitivity** directory and following the provided instructions.

3. **Power Trace Experiments**  
   Proceed to the **Power_trace_gen** directory and run the experiments as detailed in its README file.

4. **Accuracy and Energy Evaluation**  
   Validate the effectiveness of the proposed countermeasure by running the experiments in the **Topk_inference** directory. For further details, please refer to the README within that directory.  
   *Note:* A prerequisite to energy evaluations is the extraction of the latency of each layer from NeuroSim (use latest branch).


## Dependecies 
- Pytorch 
- Numpy
- matplotlib

## References
Please cite the following works along with our paper if use you any part of the code published here. In developing our code we used/adopted part of the code published in the following works. 
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

@article{adahessian, 
title={ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning},
 journal={Proc. AAAI Conf. Artificial Intelligence}, 
 author={Yao, Zhewei and Gholami, Amir and Shen, Sheng and Mustafa, Mustafa and Keutzer, Kurt and Mahoney, Michael}, 
 year={2021} 
}

@inproceedings{yao2020pyhessian,
  title={Pyhessian: Neural networks through the lens of the hessian},
  author={Yao, Zhewei and Gholami, Amir and Keutzer, Kurt and Mahoney, Michael W},
  booktitle={IEEE International Conference on Big Data},
  pages={581--590},
  year={2020},
  organization={IEEE}
}

@INPROCEEDINGS{neurosim,
  author={Peng, Xiaochen and Huang, Shanshi and Luo, Yandong and Sun, Xiaoyu and Yu, Shimeng},
  booktitle={IEEE International Electron Devices Meeting (IEDM)}, 
  title={DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies}, 
  year={2019},
  pages={32.5.1-32.5.4}
  }