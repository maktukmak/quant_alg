

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
            
                
    def fit_weight(self, model, fisher):

        def get_module_by_name(module, access_string):
            names = access_string.split(sep='.')
            return reduce(getattr, names, module)
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name not in self.exclude_layers:

                print(name)
                F = get_module_by_name(fisher, name).weight.detach().flatten()

                shape = module.weight.shape
                w = module.weight.detach().flatten()
                quant = self.quantizer(w, b=self.b)
                s = time.time()
                quant.fit_nonuniform(w)
                print('Time:', time.time()-s)

                if self.fake_quant:
                    module.weight.data = quant.lk[quant.quant(w).reshape(shape)]
                else:   
                    module.levels = torch.nn.Parameter(quant.lk)
                    module.weight.data = quant.lk[quant.quant(w).reshape(shape)]
                    module.weight.requires_grad = False
                    module.weight.data = quant.quant(w).reshape(shape).to(torch.int8)
                    self.recursive_setattr(model, name, self.replace_layer(module))

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




