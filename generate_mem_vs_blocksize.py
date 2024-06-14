import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()

N = sum([p.numel() for p in model.parameters()])
I = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and name not in ['lm_head']:
        I.append(module.weight.numel())

Bmax = min([torch.sqrt(torch.tensor(i & -i)) for i in I])
I = sum(I)
U = N - I


bo = 16
bq = 4
B = 2**torch.arange(4, torch.log2(Bmax)+1)

g = 2**30
mem_o = (I + U) * bo / (8 * g)
def mem_qn(bq):
    return (I + U) * bq / (8*g) + (2**bq - 1) * I  * bo / (B*g)
def mem_qu(bq):
    return (I + U) * bq / (8*g) + 2 * I  * bo / (B*g)



plt.plot(B, torch.tensor([mem_o]*len(B)), label='FP16', linestyle='--')
plt.plot(B, mem_qn(4), label='4-bit non-uniform', marker='o')
plt.plot(B, mem_qu(4), label='4-bit uniform/float', marker='o')
plt.plot(B, mem_qn(2), label='2-bit non-uniform', marker='o', linestyle='--')
plt.plot(B, mem_qu(2), label='2-bit uniform/float', marker='o', linestyle='--')
plt.xlabel('Block-size')
plt.xscale('log', base=2)
plt.yscale('log')
plt.ylabel('Memory in GB')
plt.legend()
plt.grid()
plt.savefig('mem_vs_blocksize.jpg')
plt.show()
plt.close()