

import matplotlib.pyplot as plt 
import numpy as np


def func(C, N = 4, alpha=1):
    Ec = -(C**3)/(3*alpha) + C**2 / 2 - alpha * C / 2 + alpha**2/6
    Es = C**2 / (3*(N-1)**2) 
    return Ec

C = np.linspace(0, 1, 100)
res = func(C)


plt.plot(C, res)
