

import torch
from src.quantizer_weight import quantizer_weight
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as stats
import math




m = 256
X = torch.randn([m,m])
x = X.flatten()
b = 3


mu = 0
sigma = 1
rng = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(rng, stats.norm.pdf(rng, mu, sigma))


step=0.02

print('Iterative')
quantizer = quantizer_weight(b=b, qtype='nonuniform')
quantizer.fit_nonuniform_iterative(x)
xint = quantizer.quant(x)
plt.scatter(quantizer.lk, 1*step*torch.ones(quantizer.N-1), label='Iterative')
print('MSE:', torch.sqrt(quantizer.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-quantizer.lk[xint]))))

print('Analytic')
quantizer = quantizer_weight(b=b, qtype='nonuniform')
quantizer.fit_nonuniform_analytic(x)
xint = quantizer.quant(x)
plt.scatter(quantizer.lk, 2*step*torch.ones(quantizer.N-1), label='Analytic')
print('MSE:', torch.sqrt(quantizer.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-quantizer.lk[xint]))))


print('Quantile')
quantizer = quantizer_weight(b=b, qtype='nonuniform')
quantizer.fit_nonuniform_quantile(x)
xint = quantizer.quant(x)
plt.scatter(quantizer.lk, 3*step*torch.ones(quantizer.N-1), label='Quantile')
print('MSE:', torch.sqrt(quantizer.error(x, xint)))
print('MAE:', torch.sqrt(torch.mean(torch.abs(x-quantizer.lk[xint]))))

plt.legend()
plt.yticks([])
plt.savefig('nonuni_grids_normal.jpg', bbox_inches='tight')
plt.show()

