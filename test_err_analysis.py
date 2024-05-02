import torch
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from src.quantizer_weight import quantizer

m = 128
#X = 2 * torch.rand([m,m]) - 1
X = torch.randn([m,m])
x = X.flatten()
b = 4


module = quantizer(x, b)
module.fit_uniform(x)


plt.hist(x, bins=30)
plt.xticks(module.lk, rotation = 45)
plt.show()
plt.savefig('res.jpg')

print('Scale:', module.s)
print('Shift:', module.z)


C = (2 * np.sqrt(6.31)) / (2**(b)-1)
