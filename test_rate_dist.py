

from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from scipy import stats
import torch
from src.quantizer_weight import quantizer
torch.set_printoptions(precision=10)
import torch.nn.utils.parametrize as P
import time
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import os


model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")

def process_layer(args):
    name, module = args
    if isinstance(module, torch.nn.Linear) and name!='lm_head':

        print(name)
        
        shape = module.weight.shape
        w = module.weight.detach().flatten()
        quant = quantizer(w, b=4)

        s = time.time()
        quant.fit_nonuniform(w)
        print('Time:', time.time()-s)

        module.levels = quant.lk
        module.weight.data = quant.lk[quant.quant(w).reshape(shape)]
    



for args in model.named_modules():
    process_layer(args)


