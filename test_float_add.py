import torch
torch.set_printoptions(precision=10)
import sys
sys.path.insert(0, "./src")

from utils import float_grid

x = torch.tensor(1).to(torch.float16)
y =  torch.tensor(0.0001).to(torch.float16)

x + y

G = float_grid(E=8, M=10, b=15, special=0)