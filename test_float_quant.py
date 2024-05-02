import torch
import numpy as np
import matplotlib.pyplot as plt
import struct
from utils import cast_to_fp8


X = torch.randn((8,8)).to(torch.float16)
Xf = X.to(torch.float8_e4m3fn)

Xf2 = cast_to_fp8(X)
print(torch.eq(Xf.to(torch.float16), Xf2))

# #E4M3
# b = 7
# E = 4
# M = 3
# max_e = 2**E-1
# max_fn = 2**M-2
# max_fs = 2**M-1

# #E5M2
# b = 15
# E = 5
# M = 2
# max_e = 2**E-2
# max_fn = 2**M-1
# max_fs = 2**M-1

# # Normal grid points
# Gn = []
# for e in range(1, max_e+1):
#     for f in range(0, max_fn+1):
#         fe = np.array([int(x) for x in bin(f)[2:]])
#         c = np.array([2**(-n) for n in range(1, M+1)])[-len(fe):]
#         Gn.append(2**(e-b) * (1 + sum(fe * c)))

# # Sub-normal grid points
# Gs = []
# for f in range(1, max_fs+1):
#     fe = np.array([int(x) for x in bin(f)[2:]])
#     c = np.array([2**(-n) for n in range(1, M+1)])[-len(fe):]
#     Gs.append(2**(1-b) * sum(fe * c))


# G = []
# for k in range(1, 123+1):
#     G.append(2**(k // 2**M) * ((k % (2**M) + (k >= 4) * 1) * 2**(1-M-b)))


# Gn = [2**(k // 2**M) * 2**(-b) * (1 + k % (2**M) * 2**(-M)) for k in range(4, 124)]
# Gs = [2**(k // 2**M) * 2**(-b) * (0 + k % (2**M) * 2**(1-M)) for k in range(1, 4)]

# G = torch.tensor(Gs + Gn)

# v = 2**(torch.floor(torch.log2(torch.abs(X)) + 2**(-b)) - M)
# v[torch.floor(torch.log2(torch.abs(X)) + b) < 1] = 2**(1-M-b)
# Xf2 = v * torch.round(X / v)

# print(torch.eq(Xf.to(torch.float16), Xf2))