

import matplotlib.pyplot as plt
from scipy import stats
import torch
from src.quantizer_weight import quantizer_weight
torch.set_printoptions(precision=10)
import torch.nn.utils.parametrize as P
import time
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import os
from src.utils import QLinear
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import save_file as safe_save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from functools import reduce


# Log layer_name, mse, mae, time 

class quantizer_model():
    def __init__(self, b):
        
        self.exclude_layers = ['lm_head']
        self.fake_quant = True
        self.quantizer = quantizer_weight
        self.b = b

    def square_grad_hook(self, grad):
        return grad.pow(2)
    
    def calibrate(self, model, dataloader):

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name not in self.exclude_layers:
                module.weight.register_hook(self.square_grad_hook)

        model.cuda()
        for data in tqdm(dataloader):
            data = data[0]
            x = data.cuda()
            outputs = model(input_ids=x, labels=x)
            loss = outputs.loss
            loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name not in self.exclude_layers:
                module.weight.data = module.weight.grad
            
                
    def fit_weight(self, model, fisher=None, qtype='float', block_size=1, alg='snr', decompose_outlier=False, format_fp4='e3m0', format_fp8='e4m3', verbose=False):

        def get_module_by_name(module, access_string):
            names = access_string.split(sep='.')
            return reduce(getattr, names, module)

        self.quant = self.quantizer(b=self.b, qtype=qtype, format_fp4=format_fp4, format_fp8=format_fp8)  
        result = {}

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name not in self.exclude_layers:
                if verbose: 
                    print(name)
                shape = module.weight.shape

                if fisher:
                    F = get_module_by_name(fisher, name).weight.detach().flatten()
                    self.quant.f = F.type(torch.float32)

                w = module.weight.detach().flatten().type(torch.float32)

                
                s = time.time()
                wdeq = self.quant.fit_and_quant(w, alg, block_size=block_size, decompose_outlier=decompose_outlier)

                if verbose: 
                    rmse = torch.sqrt(torch.mean(torch.square(w-wdeq)))
                    rmae = torch.sqrt(torch.mean(torch.abs(w-wdeq)))
                    result[name] = (rmse, rmae)
                    print('RMSE:', rmse)
                    print('RMAE:', rmae)
                    print('Time:', time.time()-s)

                if self.fake_quant:
                    module.weight.data = wdeq.reshape(shape).type(module.weight.dtype)

                # else:   
                #     module.levels = torch.nn.Parameter(quant.lk)
                #     module.weight.data = res.reshape(shape)
                #     module.weight.requires_grad = False
                #     module.weight.data = quant.quant(w).reshape(shape).to(torch.int8)
                #     self.recursive_setattr(model, name, self.replace_layer(module))

        return result

    def replace_layer(self, module):
        shape = module.weight.shape
        w = module.weight.detach().flatten()
        quant = self.quantizer(w, b=4)
        
        s = time.time()
        #quant.fit_nonuniform(w)
        print('Time:', time.time()-s)

        new_module = QLinear(module.in_features,
                                module.out_features, 
                                module.bias,
                                levels = quant.lk, 
                                indexes = quant.quant(w).reshape(shape).to(torch.int8))
        return new_module
    
    def recursive_setattr(self, obj, attr, value):
        attr = attr.split('.', 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            self.recursive_setattr(getattr(obj, attr[0]), attr[1], value)




