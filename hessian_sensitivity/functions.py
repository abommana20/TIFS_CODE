import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg16
from torch.utils.data import DataLoader
import numpy as np
import random
import math
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
##################################Start of Functions######################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Train and Test Functions
def train(model, train_loader, criterion, optimizer, epoch, device, print_interval=100):
    losses = AverageMeter()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(train_loader, 0):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        losses.update(loss.item(), data.size(0))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # Print average loss and accuracy after completing the batch
        if (batch_idx) % print_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), train_loader.sampler.__len__(),
                        100. * batch_idx / len(train_loader), loss.item()))
    total_avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print('\nTrain set: Avg Loss: {} Accuracy: {}/{} ({:.4f}%)\n'.format(losses.avg,
        correct, train_loader.sampler.__len__(),
        100. * correct / train_loader.sampler.__len__()))
    # print(f'Epoch {epoch}:  Avg Loss: {total_avg_loss}, Train Accuracy: {train_accuracy}%')


def validate(model, test_loader, criterion, epoch, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = 100. * correct / test_loader.sampler.__len__()
    avg_test_loss = running_loss / len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            avg_test_loss, correct, test_loader.sampler.__len__(),
            100. * correct / test_loader.sampler.__len__()))
    # print(f' Test Loss: {avg_test_loss}, Test Accuracy: {test_accuracy}%')
    return avg_test_loss, test_accuracy

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def get_trace_hut(model, loss, n_v,device, layerwise = False, filterwise=True):
    """
    Compute the trace of hessian using Hutchinson's method
    This approach requires computing only the application of the Hessian to a random input vector
    This has the same cost as backpropagating the gradient

    Rademacher vector v is a list of tensors that follows size of parameter tensors.
    Hessian vector product Hv is a tuple of tensors that follows size of parameter tensors.
    Final result trace_vHv is a 3D tensor containing 300 x vHv for each channel in each layer.

    """

    params = []
    conv_fc = []

    for name, module in model.named_modules():
        parent_name, child_name = name.rsplit('.', 1) if '.' in name else (None, name)
        if isinstance(module, nn.Conv2d) or isinstance(module,nn.Linear):
            params.append(module.weight)
            if isinstance(module, nn.Conv2d):
                conv_fc.append(0)
            else:
                conv_fc.append(1)
    # print(len(params))
    # print(len(conv_fc))
    # for name, param in model.named_parameters():
    #     if 'bias' not in name and param.requires_grad:
    #         if 'conv' in name:
    #             # print(name)
    #             params.append(param)
    #             conv_fc.append(0)
    #         if 'linear' in name:
    #             params.append(param)
    #             conv_fc.append(1)

    gradsH = torch.autograd.grad(loss, params, create_graph=True)
    # print(conv_fc)
    trace_vhv = []

    for i in range(n_v):
            # Sampling a random vector from the Rademacher Distribution
            # print(i)
            v = [torch.randint_like(p, high=2, device=device).float() * 2 - 1 for p in params]
            # print(v.shape)
            # Calculate 2nd order gradients in FP32
            # stop_criterion=(i == (n_v - 1))
            Hv = torch.autograd.grad(gradsH, params, grad_outputs=v,only_inputs=True, retain_graph=True)
            # print(len(Hv))
            # for hv_i in Hv:
                # print(hv_i.shape)

            # v = [vi.detach().cpu() for vi in v]
            # Hv = [Hvi.detach().cpu() for Hvi in Hv]

            with torch.no_grad():

                if layerwise:
                    trace_layer = []
                    for Hv_i in range(len(Hv)):
                        trace_layer.append(Hv[Hv_i] *v[Hv_i])
                    trace_vhv.append(trace_layer)


                else:
                    trace_vhv.append(group_product(Hv, v).item())

    ##DO Average
    # print(len(trace_vhv))
    avg_trace =[]
      
    if layerwise:
        average_value = [sum(elements) / len(trace_vhv) for elements in zip(*trace_vhv)]
        avg_trace = (average_value)
    else:
        avg_trace.append(sum(trace_vhv) / len(trace_vhv))

    return avg_trace


def num_pad( source, target):
    crxb_index = math.ceil(source / target)
    num_padding = crxb_index * target - source
    return crxb_index, num_padding

def num_pad_col(source, target, H,B):
    #crxb_col_fit = target/(self.quantize_weights/2)
    crxb_index = math.ceil((source * (H//B)) / target)
    num_padding =int( crxb_index * target / (H//B) - source)
    return crxb_index, num_padding


def quantize(max_weight, num_bits=8):
    qmax = 2**(num_bits - 1) - 1
    # max_val = tensor.abs().max()
    scale = max_weight / qmax
    # quantized = torch.round(tensor / scale)
    return scale

def dequantize(quantized_tensor, scale):
    return quantized_tensor * scale
##################################End of Functions######################