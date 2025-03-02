import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

def get_input_traces(dataset_name,model_name, model, save_path, device):
    # Data loader setup
    
    
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std= (0.2470, 0.2435, 0.2616)
            
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, 32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_train)
    elif dataset_name == "cifar100":
        mean =  (0.5071, 0.4867, 0.4408)
        std =  (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, 32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform_train)
    else:
        raise ValueError("Unsupported dataset")
    
    dataloader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=2)
    
    # Initialize list to store inputs for each relevant layer
    # Initialize lists to store inputs and weights for each layer
    inputs_list = []
    weights_list = []

    def save_input(layer_idx):
        # print(f"Saving input for layer {layer_idx}")
        def hook(module, input, output):
            while len(inputs_list) <= layer_idx:
                inputs_list.append([])
            # print(input[0].size())
            
            inputs_list[layer_idx].append(input[0].to('cpu').numpy())
            if len(weights_list) <= layer_idx:
                weights_list.append(module.weight.data.cpu().numpy())
        return hook

    # Attach hooks to all conv and linear layers
    layer_idx = 0
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            layer.register_forward_hook(save_input(layer_idx))
            layer_idx += 1
    
    # Forward pass on the entire dataset
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            model(inputs)
    
    # Calculate average inputs
    # print(len(inputs_list[0]))
    average_inputs = [sum(layer_inputs) / len(layer_inputs) for layer_inputs in inputs_list if layer_inputs]

    # Calculate the average along the batch dimension for each tensor
    averaged_tensors = [np.mean(t, axis=0, keepdims=True) for t in average_inputs]

    # Save average inputs and weights to files
    torch.save(averaged_tensors, f"{save_path}/inputs_{model_name}_{dataset_name}.pt")
    torch.save(weights_list, f"{save_path}/weights_{model_name}_{dataset_name}.pt")

    return average_inputs, weights_list


def get_input_traces_input_attack(model_name, model, save_path, device):
    # Synthetic input setup
    input_size = (3, 32, 32)  # CIFAR10/100 input dimensions
    batch_size = 1
    num_batches = 1  # Define how many batches to simulate

    # Create a synthetic batch with the maximum value (1.0 for all elements)
    synthetic_data = torch.ones((batch_size,) + input_size, dtype=torch.float32)

    # Initialize lists to store inputs and weights for each layer
    inputs_list = []
    # weights_list = []

    def save_input(layer_idx):
        def hook(module, input, output):
            while len(inputs_list) <= layer_idx:
                inputs_list.append([])
            
            inputs_list[layer_idx].append(input[0].to('cpu').numpy())
            # if len(weights_list) <= layer_idx:
            #     weights_list.append(module.weight.data.cpu().numpy())
        return hook

    # Attach hooks to all conv and linear layers
    layer_idx = 0
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            layer.register_forward_hook(save_input(layer_idx))
            layer_idx += 1
    
    # Forward pass on the synthetic data
    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):  # Process multiple batches
            synthetic_data = synthetic_data.to(device)
            model(synthetic_data)

    # Calculate average inputs
    average_inputs = [np.mean(np.stack(layer_inputs), axis=0) for layer_inputs in inputs_list if layer_inputs]

    # Save average inputs and weights to files
    print(f"Saving synthetic inputs for {model_name}...")
    # print(average_inputs)
    torch.save(average_inputs, f"{save_path}/inputs_{model_name}_synthetic.pt")

    return average_inputs
# Example usage
from resnet import *
from densenet import *
from VGG import *
from lenet import *
from alexnet import *
from vgg8 import *

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = ResNet18(num_classes=10)
    ##load the model
    model_name = "resnet18"
    dataset_name = "cifar10"
    save_path = "./trace_data"


    ##create a folder to save the traces
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset_name = "cifar10"
    # model_names = ['resnet18','vgg16','densenet121','densenet40', 'alexnet', 'lenet', 'vgg8']
    model_names = ['vgg8']
    for model_name in model_names:
        if model_name == 'resnet18':
            model = ResNet18(num_classes=10)
            model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
            model.to(device)
        elif model_name == 'vgg16':
            model = VGG('VGG16',num_classes=10).to(device)
            model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
            model.to(device)
        elif model_name == 'densenet121':
            model = densenet121(num_classes=10).to(device)
            model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
            model.to(device)
        elif model_name == 'alexnet':
            model = AlexNet(num_classes=10).to(device)
            model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
            model.to(device)
        elif model_name == 'lenet':
            model = LeNet(num_classes=10).to(device)
            model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
            model.to(device)
        elif model_name == 'vgg8':
            model = VGG8(num_classes=10).to(device)
            model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
            model.to(device)
        elif model_name == 'densenet40':
            model = densenet40(num_classes=10).to(device)
            model.load_state_dict(torch.load(f"trained_models/{model_name}_{dataset_name}.pth"))
            model.to(device)

        model.eval()  # Set the model to evaluation mode
        get_input_traces_input_attack(model_name, model, save_path, device)
        # print(f"Getting input traces for {model_name}...")
        # get_input_traces(dataset_name,model_name, model, save_path, device)