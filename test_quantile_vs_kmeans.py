
import torch
from src.quantizer_weight import quantizer_weight

d = 2**10
X = torch.randn((d, d))
x = X.flatten()

print('Iterative')
quant = quantizer_weight(x, b=4)
quant.fit_nonuniform_iterative(x)
print(quant.lk)
print(quant.error(x, quant.quant(x)))

print('Quantile')
quant = quantizer_weight(x, b=4)
quant.fit_nonuniform_quantile(x)
print(quant.lk)
print(quant.error(x, quant.quant(x, use_threshold=True)))


print('Analytic')

pdf = torch.distributions.normal.Normal(0., 1.)
tk = quant.tk
for _ in range(500):

    F = quant.gauss_cdf(tk, 0., 1.)
    P = torch.exp(pdf.log_prob(torch.Tensor(tk)))
    
    lk = -(P[1:] - P[:-1]) / (F[1:] - F[:-1])
    tk[1:-1] = (lk[1:] + lk[0:-1])/2


print(lk)
