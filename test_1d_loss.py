import torch
import matplotlib.pyplot as plt


x = torch.randn(100)
w = torch.randn(1)

def model(x, w):
    return x * w**2 + 2* x * (w**2-10)
y = model(x,w)

def l_func(x, y, w):
    yp = model(x,w)
    l = torch.sum((yp - y)**2)
    return l


wr = torch.linspace(-10, 10, 1000)
plt.plot(wr, [l_func(x, y, i) for i in wr])
plt.yscale('log')