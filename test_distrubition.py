import torch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from src.quantizer_weight import quantizer_weight
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
import time
from sklearn.neighbors import KernelDensity

synthetic_data = False

if synthetic_data:
    m = 512
    #X = 2 * torch.rand([m,m]) - 1
    X = torch.randn([m,m])*2
    x = X.flatten().numpy()
    b = 8
else:
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", device_map="cpu").eval()
    x = model.model.layers[5].self_attn.q_proj.weight.detach().flatten().type(torch.float32).numpy()


eval_points = np.linspace(np.min(x), np.max(x))

kde_sk = KernelDensity(bandwidth=0.01, kernel='gaussian')
kde_sk.fit(x.reshape([-1,1]))
y_sk = np.exp(kde_sk.score_samples(eval_points.reshape(-1,1)))

plt.plot(eval_points, y_sk)