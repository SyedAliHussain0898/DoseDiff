import torch
import torch.nn as nn
import math

class LoRALayer:
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0
    ):
        self.rank = rank
        self.alpha = alpha
        
        # Actual trainable parameters
        self.lora_A = nn.Parameter(torch.empty((rank, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, rank)))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge_weights(self, base_weight):
        return base_weight + (self.alpha / self.rank) * self.lora_B @ self.lora_A

class LoRALinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora = LoRALayer(
            self.in_features, 
            self.out_features,
            kwargs.get('lora_rank', 8)
        )
        
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = x.float()
        
        # Original projection
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA projection
        lora_effect = (x @ self.lora.lora_A.T @ self.lora.lora_B.T) * (self.lora.alpha / self.lora.rank)
        
        return (result + lora_effect).to(orig_type)
