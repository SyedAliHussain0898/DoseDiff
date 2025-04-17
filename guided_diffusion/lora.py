import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):  # Changed to nn.Module
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Register as Parameters
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, lora_rank=8, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.lora = LoRALayer(in_features, out_features, lora_rank)
        
    def forward(self, x):
        # Ensure LoRA params are on same device as input
        lora_A = self.lora.lora_A.to(x.device)
        lora_B = self.lora.lora_B.to(x.device)
        
        # Original projection
        result = nn.Linear(x, self.weight, self.bias)
        
        # LoRA projection
        lora_effect = (x @ lora_A.T @ lora_B.T) * (self.lora.alpha / self.lora.rank)
        
        return result + lora_effect
