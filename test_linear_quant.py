import torch
import matplotlib.pyplot as plt
from src.quantizer_weight import quantizer_weight

d = 2**13
W = torch.randn((d, d))
w = W.flatten()
x = torch.rand(d) + 1 



print('Uniform min-max')
module = quantizer_weight(w, b=4)
module.fit_uniform_minmax(w)
wint = module.quant(w)
wh = module.lk[wint]

e = (w - wh)
Wh = wh.reshape(W.shape)
print('Weight error mean', e.mean())
print('Weight error variance', e.var())
print('Weight error variance-theory', module.s**2/12)

y = W @ x
yh = Wh @ x

ey = y - yh
print('Output error mean', ey.mean())
print('Output error variance', ey.var())
print('Weight error variance-theory', d * (x.var() + x.mean()**2) * module.s**2/12)

plt.hist(ey, bins=30)
plt.show()
