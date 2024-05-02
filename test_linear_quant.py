import torch
import matplotlib.pyplot as plt

w = torch.randn(10000)
X = torch.randn((10000,10000))


y = X @ w

X @ X.T


plt.hist(y, bins=30)
plt.show()
plt.savefig('res.jpg')