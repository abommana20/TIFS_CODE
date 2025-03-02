## Overview
This repository provides the necessary steps to generate and integrate Hessian scores for the models.

## Instructions

1. **Generate Hessian Scores for the Models**  
   Run the main file to first generate the Hessian scores for your models.

2. **Generate Crossbar-Level Hessian Scores**  
   Execute the `hessian_gen_crxb.ipynb` notebook to further process and generate Hessian scores at the crossbar level.

3. **Integrate Generated Scores**  
   Copy the `hessian_score_crxb` directory into the `topk_inference/benchmark/` folder to complete the setup.