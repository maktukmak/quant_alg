import torch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from src.quantizer_weight import quantizer_weight
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import seaborn as sns


synthetic_data = False
b = 8
if synthetic_data:
    m = 2048
    #X = 2 * torch.rand([m,m])
    X = torch.randn([m,m])
    x = X.flatten()
else:
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()
    x = model.model.layers[0].self_attn.q_proj.weight.detach().flatten().type(torch.float32)
    


b = 8

start = time.time()
print('Uniform normal')
module = quantizer_weight(x, b)
module.fit_uniform_normal(x)
xint = module.quant(x)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print('Time:', time.time() - start)

ind = (x > module.lk[0]) & (x < module.lk[-1])
plt.xlim(-0.003, 0.003)
sns.histplot((x[ind]-module.lk[xint[ind]]).numpy(), bins=50, stat="probability")
plt.xlabel('Error')
plt.savefig('hist_error_uni8.jpg', bbox_inches='tight')
plt.show()

start = time.time()
print('Float min-max')
module = quantizer_weight(x, b)
module.fit_float_normal(x)
xint = module.quant(x)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print('Time:', time.time() - start)

e = (x-module.lk[xint]).numpy()
max_s = (module.lk[1] - module.lk[0]).numpy()
sns.histplot(e[abs(e)<max_s/2], bins=1000, stat="probability")
plt.xlim(-0.003, 0.003)
plt.xlabel('Error')
plt.savefig('hist_error_float8.jpg', bbox_inches='tight')
plt.show()











b = 4

start = time.time()
print('Uniform normal')
module = quantizer_weight(x, b)
module.fit_uniform_normal(x)
xint = module.quant(x)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print('Time:', time.time() - start)

ind = (x > module.lk[0]) & (x < module.lk[-1])
plt.xlim(-0.03, 0.03)
sns.histplot((x[ind]-module.lk[xint[ind]]).numpy(), bins=50, stat="probability")
plt.xlabel('Error')
plt.savefig('hist_error_uni4.jpg', bbox_inches='tight')
plt.show()

start = time.time()
print('Float normal')
module = quantizer_weight(x, b)
module.fit_float_normal(x)
xint = module.quant(x)
print('MSE:', torch.sqrt(module.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-module.lk[xint]))))
print('Time:', time.time() - start)

e = (x-module.lk[xint]).numpy()
max_s = (module.lk[1] - module.lk[0]).numpy()
sns.histplot(e[abs(e)<max_s/2], bins=100, stat="probability")
plt.xlim(-0.03, 0.03)
plt.xlabel('Error')
plt.savefig('hist_error_float4.jpg', bbox_inches='tight')
plt.show()