import torch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from src.quantizer_weight import quantizer_weight
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

synthetic_data = False
b = 4
if synthetic_data:
    m = 1024
    #X = 2 * torch.rand([m,m])
    X = 3*torch.randn([m,m])
    x = X.flatten()
else:
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()
    x = model.model.layers[0].self_attn.q_proj.weight.detach().flatten().type(torch.float32)
    

start = time.time()
print('Float cast')
module = quantizer_weight(b, qtype='float', format_fp4='e3m0')
module.fit_float_cast(x)
xint = module.quant(x)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print('Time:', time.time() - start)


start = time.time()
print('Float min-max')
module = quantizer_weight(b, qtype='float', format_fp4='e3m0')
module.fit_float_minmax(x)
xint = module.quant(x)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print('Time:', time.time() - start)


start = time.time()
print('Float iterative')
module = quantizer_weight(b, qtype='float', format_fp4='e3m0')
module.fit_float_iterative(x, verbose=False)
xint = module.quant(x)
print('Scale:', module.s)
print('Shift:', module.z)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print('Time:', time.time() - start)

start = time.time()
print('Float normal')
module = quantizer_weight(b, qtype='float', format_fp4='e3m0')
module.fit_float_normal(x)
xint = module.quant(x)
print('Scale:', module.s)
print('Shift:', module.z)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print('Time:', time.time() - start)




