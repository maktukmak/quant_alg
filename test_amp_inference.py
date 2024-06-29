
import torch

device= 'cuda'

x = torch.randn((10,10)).to(device)
w = torch.randn((1, 10)).to(device)
b = torch.zeros(1).to(device)

with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
    y = torch.nn.functional.linear(x, w, b)

y2 = x.to(torch.float16) @ w.to(torch.float16).T + b.to(torch.float16)

torch.equal(y, y2)
# There is numerical error due to cublas gemm implementation when bias is present.