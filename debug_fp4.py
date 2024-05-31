import torch
from src.quantizer_weight import quantizer_weight
import matplotlib.pyplot as plt

m = 64
#X = 2 * torch.rand([m,m])
X = 3*torch.randn([m,m])
x = X.flatten()
b = 4



quantizer = quantizer_weight(x, b=4)
quantizer.G = quantizer.float_grid(quantizer.E, quantizer.M, quantizer.bias, quantizer.c)