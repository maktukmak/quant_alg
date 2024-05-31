


from transformers import AutoModelForCausalLM, AutoTokenizer
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
import seaborn as sns





model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()
w = model.model.layers[0].self_attn.q_proj.weight.detach().flatten().type(torch.float32)
b = 4


module = quantizer_weight(w, b)
#module.fit_nonuniform(w)
#module.fit_uniform(w)
#module.fit_minmax(w)
module.fit_uniform_normal(w)
wint = module.quant(w)
print(module.error(w, wint))

plt.hist(w, density=True, bins=30)
plt.xticks(module.lk, rotation = 45)
plt.show()
plt.savefig('res.jpg')

#sns.kdeplot(w)
