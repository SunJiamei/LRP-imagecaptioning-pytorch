import torch
import torch.nn as nn
import gc
from . import lrp_modules


class SequentialPresetA(object):
    def __init__(self):
        self.lrp_params = {}
        self.lrp_params["alpha"] = 1.
        self.lrp_params["beta"] = 0.
        self.lrp_params["ignore_bias"] = True

def get_lrp_hook(lrp_method, lrp_params=None):
    def lrp_hook(module, relevance_input, relevance_output):  # the function in torch.nn relevance_input is basically the gradient of the input, relevance_output is modified to calculate the new gradient
        lrp_module = lrp_modules.get_lrp_module(module)
        return lrp_module.propagate_relevance(module, relevance_input,
                                              relevance_output,
                                              lrp_method, lrp_params=lrp_params)

    return lrp_hook


def save_input_hook(module, input_, output):
    module.input = input_


class LRPLoss(nn.Module):
    # Dummy loss to provide anchor for LRP recursion
    def forward(self, x):
        return x

    def backward(self, x):
        return x


def add_lrp(model):
    preset = SequentialPresetA()

    # Override default parameters if provided

    lrp_method_curr = 'alpha_beta'
    for module in model.modules():
        # Take only the leaf modules
        num_modules = len(list(module.children()))
        if num_modules == 0:
            if type(module) in[ nn.Linear] :
                lrp_method_curr = 'epsilon'
            if type(module)  in [nn.BatchNorm2d, nn.BatchNorm1d,]:
                lrp_method_curr = 'epsilon'
            if type(module) == nn.ReLU:
                lrp_method_curr = 'identity'
            module.register_forward_hook(save_input_hook)
            module.register_backward_hook(
                get_lrp_hook(lrp_method_curr, lrp_params=preset.lrp_params))
            lrp_method_curr = 'alpha_beta'

    # Add LRP computation as member function for convenience
    model.compute_lrp = lambda sample, **kwargs: compute_lrp(model, sample, **kwargs)


# @profile()
def compute_lrp(model, sample, target=None, return_output=False,
                rectify_logits=False, explain_diff=False):
    # target: list of class labels
    if sample.requires_grad==False:
        sample.requires_grad = True  # We need to compute LRP until input layer
    # sample.register_hook(get_tensor_hook)
    criterion = LRPLoss()
    logits = model(sample)
    # print(logits.shape)
    loss = criterion(logits)
    anchor = target
    # TODO output.backward(torch.Tensor(anchor))

    # TODO Normalize anchor
    # anchor /= anchor.sum(1)
    model.zero_grad()
    sample.retain_grad()
    loss.backward(anchor, retain_graph=True)
    assert sample.grad.sum()!=0
    output = sample.grad.clone().detach()
    model.zero_grad()
    if return_output:
        return output, logits
    else:
        return output


