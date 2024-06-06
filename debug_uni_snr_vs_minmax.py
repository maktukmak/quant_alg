import torch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from src.quantizer_weight import quantizer_weight
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
import time

synthetic_data = True

if synthetic_data:
    m = 1024
    #X = 2 * torch.rand([m,m]) - 1
    X = torch.randn([m,m])*1
    x = X.flatten()
    b = 8
else:
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()
    x = model.model.layers[0].self_attn.q_proj.weight.detach().flatten().type(torch.float32)
    #x = model.model.layers[0].mlp.gate_proj.weight.detach().flatten().type(torch.float32)

    # model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map="cpu").eval()
    # x = model.model.decoder.layers[0].fc1.weight.detach().flatten().type(torch.float32)

    del model

    b = 8

dist = stats.norm(loc=x.mean(), scale=x.std())
print(stats.kstest(x, dist.cdf))
print(stats.shapiro(x))


d = {}
b_list = list(range(2, 16))

for b in b_list:

    print(b)

    start = time.time()
    print('Uniform min-max')
    module = quantizer_weight(b, qtype = 'uniform')
    module.fit_uniform_minmax(x)
    xint = module.quant(x)

    xc = x[x > module.lk[-1] + module.s/2]
    xcint = module.quant(xc)
    mse_c = torch.sqrt(module.error(xc, xcint))
    print('MSEc:', mse_c)

    mse = torch.sqrt(module.error(x, xint))
    mae = torch.sqrt(torch.mean(torch.abs(x-module.lk[xint])))
    print('MSE:', mse)
    print('MAE:', mae)
    #print(module.lk)
    print('Time:', time.time() - start)
    d[b] = {'minmax': (mse, mae)}




    start = time.time()
    print('Uniform analytic')
    module = quantizer_weight(b, qtype = 'uniform')
    module.fit_uniform_normal(x)
    xint = module.quant(x)

    xc = x[x > module.lk[-1] + module.s/2]
    xcint = module.quant(xc)
    mse_c = torch.sqrt(module.error(xc, xcint))
    print('MSEc:', mse_c)

    mse = torch.sqrt(module.error(x, xint))
    mae = torch.sqrt(torch.mean(torch.abs(x-module.lk[xint])))
    print('MSE:', mse)
    print('MAE:', mae)
    #print(module.lk)
    print('Time:', time.time() - start)
    d[b]['snr'] = (mse, mae)


plt.plot(b_list, [d[b]['minmax'][0] for b in b_list], label='minmax')
plt.plot(b_list, [d[b]['snr'][0] for b in b_list], label='snr')
plt.legend()
plt.yscale('log')
plt.show()