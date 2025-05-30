import torch
import torch.nn as nn
import torch.nn.functional as F

class W8A16Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self.scales = []
        self.zps = []
        self.x_qs = []


    def quantize_per_channel(self, weight, num_bits=8, symmetric= True):
        # Clear lists to prevent accumulation across calls
        self.scales.clear()
        self.zps.clear()
        self.x_qs.clear()
        
        q_min = -(2**(num_bits - 1))
        q_max = (2**(num_bits - 1)) - 1
        for i in range(weight.shape[0]):
            if symmetric:
                max_val = max(abs(weight[i].min()), abs(weight[i].max())) #Take largest value
                min_val = -max_val #Assume both sides symmetric
            else:
                max_val = weight[i].max()
            scale = (max_val - min_val) / (q_max - q_min) if  max_val != min_val else 1.0
            zp = torch.round((min_val * q_max - max_val * q_min) / (max_val - min_val))
            self.scales.append(scale)
            self.zps.append(zp)
            self.x_qs.append(torch.round(weight[i]/ scale + zp).clamp(q_min, q_max).to(torch.int8))
        print(self.x_qs)
        print(self.zps)
        print(self.scales)
        return torch.stack(self.x_qs, dim=0)

    def quantize(self, weight, num_bits=8, symmetric= True):
        q_min = -(2**(num_bits - 1))
        q_max = (2**(num_bits - 1)) - 1
        if symmetric:
            max_val = max(abs(weight.min()), abs(weight.max())) #Take largest value
            min_val = -max_val #Assume both sides symmetric
        else:
            max_val = weight.max()
            min_val = weight.min()
        scale = (max_val - min_val) / (q_max - q_min) if  max_val != min_val else 1.0
        zp = torch.round((min_val * q_max - max_val * q_min) / (max_val - min_val))
        self.scale = scale
        self.zp = zp
        return torch.round(self.linear.weight/ scale + zp).clamp(q_min, q_max).to(torch.int8)
    
    def forward(self, x):
        self.x_q = self.quantize_per_channel(self.linear.weight)
        print(self.x_q.shape)
        print(torch.stack(self.zps, dim=0).shape)
        print(torch.stack(self.scales, dim=0).shape)
        return self.dequantize().to(torch.float32) @ x.T + self.linear.bias
    
    def dequantize(self):
        assert self.x_q is not None, "Quantized weight is not initialized"
        # Reshape scales and zps to (10, 1) for proper row-wise broadcasting
        scales = torch.stack(self.scales, dim=0).view(-1, 1)
        zps = torch.stack(self.zps, dim=0).view(-1, 1)
        return (self.x_q.float() - zps) * scales
    
if __name__ == "__main__":
    with torch.no_grad():
        linear = nn.Linear(10, 10)
        quantizer = W8A16Linear(linear)
        x = torch.randn(10, 10, dtype=torch.float32)
        print("Quantized: ", quantizer(x))
        print("Original: ", linear(x))
        # Find the average difference between the quantized and original
        diff = quantizer(x) - linear(x)
        print("Average difference: ", diff.abs().mean())
        torch.testing.assert_close(quantizer(x), linear(x), atol=1e-2, rtol=1e-2)