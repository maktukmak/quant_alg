
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantizer_model import quantizer_model
import torch
import argparse
from datasets import load_dataset
import random
import time
import numpy as np


parser = argparse.ArgumentParser(description='Model quantizer')
parser.add_argument("--model_name", type=str, default='facebook/opt-125m')
parser.add_argument("--qtype", type=str, default='nonuniform')
parser.add_argument("--alg", type=str, default='iterative')
parser.add_argument("--b", type=int, default=8, help='Bit number')
parser.add_argument("--block_size", type=int, default=None, help='Bit number')
parser.add_argument("--calib", action='store_true')
parser.add_argument("--decompose_outlier", action='store_true')
parser.add_argument("--cache_path", type=str, default='./models/')
args = parser.parse_args()


format_fp4='e2m1'
format_fp8='e4m3'

print('Quant type:', args.qtype)
print('Algorithm:', args.alg)
print('Bits:', args.b)
print('Block size:', args.block_size)

# model_name = 'lmsys/vicuna-7b-v1.1'
# model_save_name = 'vicuna-7b'
model_save_name = args.model_name.split('/')[-1]

nsamples = 100
seqlen = 2048


model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto").eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

quantizer = quantizer_model(b=args.b)
if args.calib:

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    def encode(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    datasetenc = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')

    dataloader = []
    for _ in range(nsamples):
        i = random.randint(0, datasetenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = datasetenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        dataloader.append((inp, tar))


    quantizer.calibrate(model, dataloader)
    save_name = args.cache_path + "/calib/" + model_save_name
else:
    s = time.time()
    #fisher = AutoModelForCausalLM.from_pretrained(args.cache_path + "/calib/" + model_save_name, torch_dtype="auto").eval()
    result = quantizer.fit_weight(model, fisher=None, qtype=args.qtype, block_size=args.block_size, alg=args.alg, decompose_outlier=args.decompose_outlier, format_fp4=format_fp4, format_fp8=format_fp8)
    save_name = args.cache_path + "/quant/" + model_save_name
    print('Total Time:', time.time()-s)
    print('RMSE:', np.sqrt(np.mean([i[0]**2 for i in result.values()])))
    print('RMAE:', np.sqrt(np.mean([i[1]**2 for i in result.values()])))


model.save_pretrained(save_name, from_pt=True)
tokenizer.save_pretrained(save_name)
print('Processed model saved!')
