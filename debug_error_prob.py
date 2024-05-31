

import torch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from src.quantizer_weight import quantizer_weight
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import seaborn as sns


synthetic_data = True

if synthetic_data:
    m = 2048
    #X = 2 * torch.rand([m,m])
    X = torch.randn([m,m])
    x = X.flatten()
else:
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()
    x = model.model.layers[0].self_attn.q_proj.weight.detach().flatten().type(torch.float32)
    


b = 4

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
plt.xlim(-0.5, 0.5)
plt.show()

F = module.gauss_cdf(module.xr, 0., x.var()/module.s**2)
p = F[1:] - F[:-1]




torch.concat((-module.vr / 2, module.vr / 2)).sort()[0]

