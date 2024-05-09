
import torch
torch.set_printoptions(precision=5)
import time
from scipy.spatial import distance
import numpy as np

# Methods
# - Min-max
# - Uniform optimum
# - Nonuniform optimum
# - SqueezeLLM

# - GPTQ with ADMM



class quantizer_weight():
    def __init__(self, x, b, f=None):
        
        self.b = b
        self.N = int(2**b)
        self.k_list = torch.arange(0, self.N-1).to(torch.int) # Integer set (labels)
        if f is None:
            self.f = torch.ones(x.shape)

        #Initialize with min-max
        self.fit_minmax(x)

        self.sigma2k = torch.ones(self.N-1) * self.s
        self.pi = torch.ones(self.N-1) / (self.N-1)
        self.alpha = torch.ones(self.N-1)*2




    def error(self, x, xint):
        err = torch.sum(((x - self.lk[xint])**2) * self.f) / torch.sum(self.f)
        return err
    
    def error_prob(self, x, r):
        
        err = torch.sum(r * torch.log(self.pi))
        err += torch.sum(r * self.ll_normal(x, self.lk, self.sigma2k))
        err += torch.sum((self.alpha-1) * torch.log(self.pi))
        err += -torch.sum(r * torch.log(r+1e-8))

        return err

    def fit_minmax(self, x):

        self.minx = torch.min(x)
        self.maxx = torch.max(x)
        self.s = (self.maxx - self.minx) / (self.N-1)
        self.z = self.minx + self.s/2
        self.lk = self.s * self.k_list + self.z

        xint = self.quant(x)
        err = self.error(x, xint)
        #print(err)

    def quant(self, x):
        xint = torch.argmin((x[:,None] - self.lk[None])**2, axis=1)
        return xint.to(torch.int)
    
    def ll_normal(self, x, m, s2):
        ll = -0.5 * (x[:,None] - m[None])**2 / s2 - 0.5*torch.log(2*torch.pi*s2)
        return ll
    
    def quant_prob(self, x):

        ll = self.ll_normal(x, self.lk, self.sigma2k)

        prob = torch.nn.functional.softmax(ll +  torch.log(self.pi), dim=-1)
        xint = torch.argmax(prob, axis=1)

        return xint, prob

    def fit_nonuniform(self, x, verbose=False):

        x_sorted = torch.sort(x)[0]
        self.lk = x_sorted[((len(x_sorted)-1) // (self.N - 2)) * self.k_list]
        
        err_prev = 1e10
        for _ in range(500):

            xint = self.quant(x)
            err = self.error(x, xint)
            

            if verbose:
                print(err)

            for k in self.k_list:
                ind = (xint == k)
                self.lk[k] = torch.sum(x[ind] * self.f[ind]) / torch.sum(self.f[ind])

            if abs(err_prev - err) / err < 1e-5:
                print('Error', err)
                return
            err_prev = err

        print('NOT CONVERGED !!!')
        print(err)

    def fit_nonuniform_prob(self, x):
        # Not done yet
        for _ in range(100):
            xint, r = self.quant_prob(x)

            err = self.error_prob(x, r)
            print(err)

            rk = r.sum(axis=0)

            self.pi = (rk + self.alpha - 1) / (len(x) + sum(self.alpha) - len(self.alpha))
            self.lk = (r * x[:, None]).sum(axis=0) / rk
            self.sigma2k = (r * ((x[:,None] - self.lk[None])**2)).sum(axis=0) / rk

    
    def fit_uniform(self, x):

        for _ in range(1000):
            xint = self.quant(x)

            err = self.error(x, xint)
            print(err)

            num_s = 0
            denum_s = 0
            num_z = 0
            denum_z = 0
            for k in self.k_list:
                ind = (xint == k)
                xi = x[ind]
                num_s += torch.sum(xi - self.z) * k
                denum_s += k**2  * torch.sum(ind)
                num_z += torch.sum(xi - self.s * k)
                denum_z += torch.sum(ind)

            self.s = num_s / denum_s
            self.z = num_z / denum_z
            
            self.lk = self.s * self.k_list + self.z

if __name__ == "__main__":

    m = 2048
    #X = 2 * torch.rand([m,m]) - 1
    X = 4*torch.randn([m,m])
    x = X.flatten()
    b = 4   

    module = quantizer_weight(x, b)

    print(module.lk)
    module.fit_nonuniform(x)

    print(module.lk)
    print(module.sigma2k)

