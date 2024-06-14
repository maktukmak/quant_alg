import torch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from src.quantizer_weight import quantizer_weight
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import seaborn as sns


synthetic_data = False
if synthetic_data:
    m = 2048
    #X = 2 * torch.rand([m,m])-1
    X = torch.randn([m,m])
    x = X.flatten()
else:
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()
    x = model.model.layers[0].self_attn.q_proj.weight.detach().flatten().type(torch.float32)
    

format_fp4='e2m1'
format_fp8='e4m3'
dlim = {8:0.003, 4:0.03}

for b in [8, 4]:

    lim = dlim[b]
    
    print('Uniform normal')
    module = quantizer_weight(b, qtype='uniform')
    xdeq = module.fit_and_quant(x, alg='snr')
    lk = module.compute_quant_levels()

    ind = (x > lk[0]) & (x < lk[-1])
    plt.xlim(-lim, lim)
    sns.histplot((x[ind]-xdeq[ind]).numpy(), bins=50, stat="probability")
    #plt.xscale('symlog', base=2)
    plt.xlabel('Error')
    plt.savefig('hist_error_uni' + str(b) + '.jpg', bbox_inches='tight')
    plt.show()
    plt.close()

    print('Float normal')
    module = quantizer_weight(b, qtype='float', format_fp4=format_fp4)
    xdeq = module.fit_and_quant(x, alg='snr')
    lk = module.compute_quant_levels()

    e = (x-xdeq).numpy()
    max_s = (lk[1] - lk[0]).numpy()
    sns.histplot(e[abs(e)<max_s/2], bins=1000, stat="probability")
    plt.xlim(-lim, lim)
    #plt.xscale('symlog', base=2)
    plt.xlabel('Error')
    plt.savefig('hist_error_float' + str(b) + '.jpg', bbox_inches='tight')
    plt.show()
    plt.close()


