import torch
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import math

C = 1.
sigma2 = torch.linspace(0.01,0.1,1000)


def q_function(x):
    return 0.5 - 0.5*special.erf(x/np.sqrt(2))

def gauss_cdf( x, m, std):
    return 0.5 * (1 + torch.erf((x - m) / (torch.sqrt(torch.tensor(2.)) * std)))


res = 2 * (1 + C**2/sigma2) * q_function(C / torch.sqrt(sigma2)) 
res += - C * torch.sqrt(torch.tensor(2.)/torch.pi) * torch.exp(-0.5*(C**2)/sigma2) / torch.sqrt(sigma2)



plt.plot(sigma2, res, label='4-bit')
plt.xlabel('Sigma2')
plt.legend()
plt.grid()
plt.ylabel('SNR')
plt.show()