
import torch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from src.quantizer_weight import quantizer_weight
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
import time

model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):

        print(name)
        shape = module.weight.shape
        x = module.weight.detach().flatten().type(torch.float32)


        dist = stats.norm(loc=x.mean(), scale=x.std())
        #print(stats.kstest(x, dist.cdf))

        print(stats.shapiro(x))
