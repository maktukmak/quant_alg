
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def q(x):
    return 0.5 - 0.5*special.erf(x/np.sqrt(2))

def g(z, N):
    return 1/(2 * (1 + z) * q(np.sqrt(z)) - np.sqrt(2*z/np.pi) * np.exp(-0.5*z) + z/(3*((N-1)**2)))



z = np.linspace(1,100,10000)

for b in [2, 3, 4, 8]:
    N = 2**b
    gres = g(z, N)
    #plt.plot(g)
    print('{} bit: {}'.format(b, z[np.argmax(gres)]))