
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantizer_model import quantizer_model
import torch
import argparse
from datasets import load_dataset
import random
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
import time

model_id = 'mobiuslabsgmbh/Llama-2-7b-chat-hf_2bitgs8_hqq' 
model     = HQQModelForCausalLM.from_quantized(model_id, adapter='adapter_v0.1.lora')
tokenizer = AutoTokenizer.from_pretrained(model_id)