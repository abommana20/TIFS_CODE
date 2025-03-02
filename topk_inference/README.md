###Experiment Setup and Running
All Executable files are in benchmark directory
All config files are in config directory
All PytorX files are located in benchmark/torx/module
1. 
2.
3.
4.
5.
6.


##Main files
1. main_temp.py
make sure that layer_topk.py is copied to layer.py 
Using this file we have 
2. main_acc.py 
make sure that layer_topk.py is copied to layer.py 
and this files is only used for the experiments in the directory power_trace_gen
2. main_speedup.py is used for the experiments related to the speed in the discussion section of the paper. 
make sure layer_speedup.py is copied ot layer.py
3.

Parameters:
parser.add_argument('--model', default='resnet18', type=str, help='model name')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--cuda', type=str, default='cuda:0', help='dataset to use')

    
parser.add_argument('--thermal_mode', type=int, default=2)
1. Thermal Noise mode: There are three modes of thermal noise injection. One model 20 to 10 and other 20 to 15 , we model 20 to 10 in our paper. 
parser.add_argument('--tm', type=float, default=0.1, help="max fault tolerance in tile")
2. The max amount of fault tolerance in each tile 10%
parser.add_argument('--path', type=str, default='.', help='path to save the traces')
3. This is the path where the temperature of each tile after the attack is stored. Based on these new temperature after the attack, the condutance drift is injected. 
parser.add_argument('--trained_model_path', type=str, default='trained_models', help='path to trained model')
4. Path to the trained models. 
parser.add_argument('--topk', type=float, default=0.1, help='percentgae of protection')
5. The percentage of critical weights protected.  
parser.add_argument('--enable_fec', type=int, default=0, help='error correction')
6. Enables error correction 



##Layer files (files located benchmark/torx/module)
1. layer_speedup.py
2. layer_topk.py

Whenever we want to run for one particular experiments. 
-->Copy to layer_topk to layer.py as layer.py is the main file which is called in main file. 

In each of these files we have two clases (Conv and Linear) which are crossbar-based layers.
Here we inject the temperature drift at each tile level based on the temperatures obtained from power trace gen files.
lines 410-3430 and lines 749-770
W2G.py file
-->w2g.py maps the weights to conductance. 

##Energy Evaluation (Evaluation setup and Run)
1. Based on the amount of protection in each tile and power and latency measurements obtained from NeuroSim
2. We calculate the energy ovehead-->functions defined energy_eval directory
3. these files are called in main_temp.py and speed_up.py

