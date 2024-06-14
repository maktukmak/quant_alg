
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantizer_model import quantizer_model
import torch
import argparse
from datasets import load_dataset
import random
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
import time
from src.quantizer_weight import quantizer_weight

from hqq.core.quantize import *
#Quantization settings


model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map="cpu").eval()
layer = model.model.decoder.layers[0].fc1
x = layer.weight.detach().flatten().type(torch.float32)

quant_config = BaseQuantizeConfig(nbits=4, group_size=64 ,view_as_float=False, quant_zero=False)
#Replace your linear layer 
hqq_layer = HQQLinear(layer, #torch.nn.Linear or None 
                      quant_config=quant_config, #quantization configuration
                      compute_dtype=torch.float16, #compute dtype
                      device='cpu', #cuda device
                      initialize=True, #Use False to quantize later
                      del_orig=False #if True, delete the original layer
                      )


print('RMSE:', torch.sqrt(torch.mean(torch.square(layer.weight-hqq_layer.dequantize()))))
print('RMAE:', torch.sqrt(torch.mean(torch.abs(layer.weight-hqq_layer.dequantize()))))


start = time.time()
module = quantizer_weight(b=4, qtype = 'uniform')
xdeq = module.fit_and_quant(x, alg = 'snr', block_size=64, decompose_outlier=False)
print('RMSE:', torch.sqrt(torch.mean(torch.square(x-xdeq))))
print('RMAE:', torch.sqrt(torch.mean(torch.abs(x-xdeq))))
print('Time:', time.time() - start)

