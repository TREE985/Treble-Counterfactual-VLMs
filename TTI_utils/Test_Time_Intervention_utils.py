import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel
from torch import Tensor
import numpy as np


class Shift_Layer(nn.Module):
    def __init__(self, shift_direction, lam):
        super(Shift_Layer, self).__init__()
        self.shift_direction = shift_direction
        self.lam = lam
    def forward(self, x):
        if self.shift_direction is not None:
            norm = torch.norm(x.float(),dim=-1).unsqueeze(-1)            
            y = 0
            for i in range(len(self.shift_direction)):
                if x.size(1) < 2:
                    lambda_sim = 1.0
                    y += self.lam[i] * lambda_sim * F.normalize(self.shift_direction[i], dim=-1).repeat(1,x.shape[1],1)
                else:
                    lambda_sim = 1.0
                    y += self.lam[i] * lambda_sim * F.normalize(self.shift_direction[i], dim=-1)
            y = y/len(self.shift_direction)
            x = F.normalize(F.normalize(x.float(),dim=-1) +  0.1 * y, dim=-1) * norm
            return x.half()
        else:
            return x

def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)

def find_longest_modulelist(model, path=""):
    longest_path = path
    longest_len = 0
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path
    return longest_path, longest_len

def find_module(block, keywords):
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")

def get_embedding_layer(model: PreTrainedModel):
    keywords = ["emb", "wte"]
    return find_module(model, keywords)

def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path

def get_layers(model: PreTrainedModel):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def get_mlp_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

def test_time_intervention(model: PreTrainedModel, shift_weights: Tensor, alpha: list):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    assert len(shift_weights) == len(layers)
    for i, layer in enumerate(layers):
        original_mlp = find_module(layer, mlp_keywords)
        layer.mlp = nn.Sequential(original_mlp, Shift_Layer(shift_weights[i], alpha))


def remove_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"] 
    for i, layer in enumerate(layers):
        shift_mlp = find_module(layer, mlp_keywords)
        layer.mlp = shift_mlp