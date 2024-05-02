import torch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

m = 64
X = torch.rand([m,m])
x = X.flatten()
W = torch.randn([m,m])
w = W.flatten()


n_bits = 4
def quant_error_weight(s):
    k = torch.round(w / s)
    k = torch.clip(k, -2**(n_bits-1)-1, 2**(n_bits-1)-1)
    wq = k * s
    return torch.sum((w - wq)**2)


def quant_error_output(s):
    k = torch.round(w / s)
    k = torch.clip(k, -2**(n_bits-1)-1, 2**(n_bits-1)-1)
    wq = (k * s).type(torch.float32)
    return torch.sum((X @ w.reshape(m,m) - X @ wq.reshape(m,m))**2)



x0 = torch.max(torch.abs(x)) / (2**(n_bits-1)-1)
print('Init:', x0)
rng = np.arange(x0-0.3, x0+0.2, 0.001)


res = minimize(quant_error_weight,
               x0, 
               method='Nelder-Mead', 
               tol=1e-6)
print('Weight:', res.x)

l = torch.tensor([quant_error_weight(s) for s in rng])
s = rng[np.argmin(l)]
print('Weight search:', s)
plt.plot(rng, l / max(l), label = 'weight')




res = minimize(quant_error_output,
               x0, 
               method='Nelder-Mead', 
               tol=1e-6)
print('Output:', res.x)

l = torch.tensor([quant_error_output(s) for s in rng])
s = rng[np.argmin([quant_error_output(s) for s in rng])]
print('Output search:', s)
plt.plot(rng, l / max(l), label = 'output')

plt.legend()
plt.show()