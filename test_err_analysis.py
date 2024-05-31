import torch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from src.quantizer_weight import quantizer_weight
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

synthetic_data = False

if synthetic_data:
    m = 512
    #X = 2 * torch.rand([m,m]) - 1
    X = torch.randn([m,m])
    x = X.flatten()
    b = 4
else:
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()
    x = model.model.layers[0].self_attn.q_proj.weight.detach().flatten().type(torch.float32)
    b = 2


start = time.time()
print('Uniform min-max')
module = quantizer_weight(b, qtype = 'uniform')
module.fit_uniform_minmax(x)
xint = module.quant(x)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print(module.lk)
print('Time:', time.time() - start)

start = time.time()
print('Uniform iterative')
module = quantizer_weight(b, qtype = 'uniform')
module.fit_uniform_iterative(x, verbose=False)
xint = module.quant(x)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print(module.lk)
print('Time:', time.time() - start)

start = time.time()
print('Uniform analytic')
module = quantizer_weight(b, qtype = 'uniform')
module.fit_uniform_normal(x)
xint = module.quant(x)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print(module.lk)
print('Time:', time.time() - start)

start = time.time()
print('Non-uniform iterative')
module = quantizer_weight(b, qtype = 'nonuniform')
module.fit_nonuniform_iterative(x)
xint = module.quant(x)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print(module.lk)
print('Time:', time.time() - start)

start = time.time()
print('Non-uniform quantile')
module = quantizer_weight(b, qtype = 'nonuniform')
module.fit_nonuniform_quantile(x)
xint = module.quant(x, use_threshold=False)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print(module.lk)
print('Time:', time.time() - start)


start = time.time()
print('Non-uniform analytic')
module = quantizer_weight(b, qtype = 'nonuniform')
module.fit_nonuniform_analytic(x)
xint = module.quant(x, use_threshold=False)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print(module.lk)
print('Time:', time.time() - start)

plt.hist(x.numpy(), bins=30)
plt.xticks(module.lk.numpy(), rotation = 45)
plt.show()
plt.savefig('res.jpg')

#print('Scale:', module.s)
#print('Shift:', module.z)



