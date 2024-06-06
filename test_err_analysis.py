import torch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from src.quantizer_weight import quantizer_weight
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

synthetic_data = False
b = 4
nblocks = 1

if synthetic_data:
    m = 4096
    X = torch.randn([m,m])
    x = X.flatten()
else:
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()
    x = model.model.layers[0].self_attn.q_proj.weight.detach().flatten().type(torch.float32)
    #x = model.model.layers[0].mlp.gate_proj.weight.detach().flatten().type(torch.float32)
    # model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map="cpu").eval()
    # x = model.model.decoder.layers[0].fc1.weight.detach().flatten().type(torch.float32)
    

methods = [
    ('uniform', 'minmax'),
    #('uniform', 'iterative'),
    ('uniform', 'snr'),
    ('float', 'minmax'),
    #('float', 'iterative'),
    ('float', 'snr'),
    #('nonuniform', 'iterative'),
    ('nonuniform', 'quantile'),
    ('nonuniform', 'snr'),
    ]

for (qtype, alg) in methods:

    start = time.time()
    print((qtype, alg))
    module = quantizer_weight(b, qtype = qtype)
    xdeq = module.fit_and_quant(x, alg = alg, nblocks=nblocks)
    print('RMSE:', torch.sqrt(torch.mean(torch.square(x-xdeq))))
    print('RMAE:', torch.sqrt(torch.mean(torch.abs(x-xdeq))))
    print('Time:', time.time() - start)

