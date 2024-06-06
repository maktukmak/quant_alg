
import torch
torch.set_printoptions(precision=5)
import time
from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans
from scipy import special

# Fix self.f

class quantizer_weight():
    def __init__(self, b, f=None, qtype='float', format_fp4='e3m0', format_fp8='e4m3'):

        self.b = b
        self.N = int(2**b)
        self.qtype = qtype

        if self.qtype == 'uniform':
            self.k_list = torch.arange(0, self.N-1).to(torch.int)

        if self.qtype == 'nonuniform':
            self.sk_kmeans = True
            

        if self.qtype == 'float':
            if self.b == 8:
                if format_fp8=='e4m3':
                    self.E=4; self.M=3; self.bias=7; self.c=0
                elif format_fp8=='e5m2':
                    self.E=5; self.M=2; self.bias=15; self.c=3
            elif self.b == 4:
                if format_fp4=='e2m1':
                    self.E=2; self.M=1; self.bias=1; self.c=0
                elif format_fp4=='e3m0':
                    self.E=3; self.M=0; self.bias=1; self.c=0

            self.float_grid(self.E, self.M, self.bias, self.c)

    def error(self, x, xint):
        if not hasattr(self, 'f'):
            self.f = torch.ones(x.shape)
        #err = torch.sum(((x - self.lk[xint])**2) * self.f) / torch.sum(self.f)
        err = torch.sum(((x - self.lk[xint])**2)) / len(x)
        return err
    
    def q_function(self, x):
        return 0.5 - 0.5*special.erf(x/np.sqrt(2))

    def snr_uni(self, z, N):
        return 1/(2 * (1 + z) * self.q_function(np.sqrt(z)) - np.sqrt(2*z/np.pi) * np.exp(-0.5*z) + z/(3*((N-1)**2)))

    def snr_float(self, C, sigma2, xr, vr):
        res = 2 * (1 + C**2 / sigma2) * self.q_function(C / torch.sqrt(sigma2)) 
        res += - C * torch.sqrt(torch.tensor(2.)/torch.pi) * torch.exp(-0.5*(C**2)/sigma2) / torch.sqrt(sigma2)

        F = self.gauss_cdf(xr[None], 0., torch.sqrt(sigma2[:,None]))
        p = F[:,1:] - F[:,:-1]

        res += torch.sum(vr**2 * p / (12*sigma2[:,None]), axis=1)
        return 1/res

    def quant(self, x, use_threshold=False):
        if use_threshold:
            xint = torch.where((x[:,None] > self.tk[:-1]) * (x[:,None] <= self.tk[1:]))[1]
        else:
            xint = torch.argmin((x[:,None] - self.lk[None])**2, axis=1)
        return xint.to(torch.int)

    def cast_to_fp8(self, x):

        v = 2**(torch.floor(torch.log2(torch.abs(x)) + 2**(-self.bias)) - self.M)
        v[torch.floor(torch.log2(torch.abs(x)) + self.bias) < 1] = 2**(1-self.M-self.bias)
        
        x = torch.clamp(x, self.G[0], self.G[-1])
        Xf = v * torch.round(x / v)

        #Xf = torch.sign(X.flatten()) * G[torch.argmin((torch.abs(X.flatten()[:,None]) - G[None]) ** 2, axis = 1)]
        return Xf

    def float_grid(self, E=8, M=10, bias=15, special=0):

        Gn = [2**(k // 2**M) * 2**(-bias) * (1 + k % (2**M) * 2**(-M)) for k in range(2**M, 2**(M+E)-1-special)]
        Gs = [2**(-bias) * (k * 2**(1-M)) for k in range(1, 2**M)]
        self.Gh = torch.tensor(Gs + Gn)
        self.G = torch.concat((-torch.flip(self.Gh, [0]), torch.tensor([0.]), self.Gh))

    def gauss_cdf(self, x, m, std):
        return 0.5 * (1 + torch.erf((x - m) / (torch.sqrt(torch.tensor(2.)) * std)))
    
    def fit_float_cast(self, x):

        self.s = 1
        self.z = 0
        self.lk = self.G * self.s + self.z

    def fit_float_minmax(self, x):

        self.minx = torch.min(x)
        self.maxx = torch.max(x)

        self.s = (self.maxx - self.minx) / (2*torch.max(self.G))
        self.z = self.minx + torch.max(self.G) * self.s
        self.lk = self.G * self.s + self.z

    def fit_float_normal(self, x):

        if not hasattr(self, 'sigma2opt'):
            C = self.G[-1]
            kmax = (2**(self.E + self.M) - 2 - self.c)
            self.R = kmax // 2**self.M + (kmax % 2**self.M > 0) * 1 - 1
            self.R = 2 * self.R - 1

            self.vr = torch.tensor([2 ** (abs(r -1 - self.R//2) + 1 - self.M - self.bias) for r in range(1, self.R+1)])

            #self.vr = torch.tensor([2 ** (r - self.M - self.bias) for r in range(1, self.R+1)])
            #self.vr = torch.concat((torch.flip(self.vr[1:], [0]), self.vr))

            #torch.tensor([2**(self.R//2 - r + 3 - self.bias) for r in range(1,self.R//2+2)])
            #torch.tensor([2**(r - self.R//2  - self.bias) for r in range(self.R//2+2, self.R+2)])
            
            self.xr = torch.tensor([2**(r + 1 - self.bias) for r in range(1,self.R//2+2)])
            self.xr[-1] = C
            self.xr = torch.concat((-torch.flip(self.xr, [0]), self.xr))
            sigma2 = torch.linspace(0.1,100,100000)
            gres = self.snr_float(C, sigma2, self.xr, self.vr)
            self.sigma2opt = sigma2[np.argmax(gres)]

        self.s = torch.sqrt(x.var() / self.sigma2opt)
        self.z = x.mean()
        self.lk = self.G * self.s + self.z

    def fit_float_iterative(self, x, f=None, verbose=False):

        if f is not None:
            self.f = f

        self.fit_float_normal(x)

        denum_z = len(x)

        err_prev = 1e10
        for _ in range(2000):
            xint = self.quant(x)
            xfloat = self.G[xint]

            err = self.error(x, xint)

            if verbose:
               print(err)
            
            num_s = torch.sum((x - self.z) * xfloat)
            denum_s = torch.sum(xfloat**2)
            num_z = torch.sum(x - self.s * xfloat)

            self.s = num_s / denum_s
            self.z = num_z / denum_z
            
            self.lk = self.s * self.G + self.z

            if abs(err_prev - err) / err < 1e-5:
                #print('Loss', err)
                return
            err_prev = err
        print('NOT CONVERGED !!!')
        print(err)

    def fit_uniform_minmax(self, x):

        self.minx = torch.min(x)
        self.maxx = torch.max(x)
        self.s = (self.maxx - self.minx) / (self.N-1)
        self.z = self.minx + self.s/2
        self.lk = self.s * self.k_list + self.z

        #xint = self.quant(x)
        #err = self.error(x, xint)
        #print(err)

    def fit_uniform_normal(self, x):

        if not hasattr(self, 'zeta'):
            z = np.linspace(1,100,10000)
            gres = self.snr_uni(z, self.N)
            self.zeta = z[np.argmax(gres)]

        self.s = (2 * np.sqrt(self.zeta * x.var())) / (self.N-1)
        self.z = - self.s * (self.N/2 - 1) + x.mean()
        self.lk = self.s * self.k_list + self.z
    
    def fit_uniform_iterative(self, x, verbose=False):

        self.fit_uniform_normal(x)

        denum_z = len(x)

        err_prev = 1e10
        for _ in range(2000):
            xint = self.quant(x)

            err = self.error(x, xint)

            if verbose:
               print(err)
            
            num_s = torch.sum((x - self.z) * xint)
            denum_s = torch.sum(xint**2)
            num_z = torch.sum(x - self.s * xint)

            self.s = num_s / denum_s
            self.z = num_z / denum_z
            
            self.lk = self.s * self.k_list + self.z

            if abs(err_prev - err) / err < 1e-5:
                #print('Loss', err)
                return
            err_prev = err
        print('NOT CONVERGED !!!')
        print(err)

    def fit_nonuniform_quantile(self, x):

        x_sorted = torch.sort(x)[0]
        k = torch.arange(0, self.N)
        ind = ((len(x)-1) * k.to(torch.float64) / (self.N-1)).to(torch.int)
        self.tk = x_sorted[ind]
        self.tk[0] = torch.nan_to_num(torch.tensor(-float('inf')))
        self.tk[-1] = torch.nan_to_num(torch.tensor(float('inf')))
        self.lk = torch.zeros(self.N-1)
        for i in k-1:
            self.lk[i] = torch.mean(x_sorted[ind[i]:ind[i+1]])

    def fit_nonuniform_analytic(self, x):

        if not hasattr(self, 'lkopt'):
            self.fit_nonuniform_quantile(x)
            pdf = torch.distributions.normal.Normal(0., 1.)
            tk = self.tk.to(torch.float64)

            for _ in range(500):

                F = self.gauss_cdf(tk, 0., 1.)
                P = torch.exp(pdf.log_prob(torch.Tensor(tk)))
                
                lk = -(P[1:] - P[:-1]) / (F[1:] - F[:-1])
                tk[1:-1] = (lk[1:] + lk[0:-1])/2

            self.lkopt = lk.to(torch.float32)
            self.tkopt = tk.to(torch.float32)

        self.lk = (x.std() * self.lkopt + x.mean())
        self.tk = x.mean()


    def fit_nonuniform_iterative(self, x, verbose=False):

        self.fit_nonuniform_analytic(x)

        if not hasattr(self, 'f'):
            self.f = torch.ones(x.shape)

        if self.sk_kmeans:


            kmeans = KMeans(n_clusters=self.N-1, init=self.lk.reshape(-1,1), max_iter=500, tol=1e-5)
            kmeans.fit(x.reshape(-1,1), self.f, )
            self.lk = torch.tensor(kmeans.cluster_centers_[:,0])

        else:
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
                    #print('Loss', err)
                    return
                err_prev = err

            print('NOT CONVERGED !!!')
            print(err)

    def fit_and_quant(self, x, alg, nblocks=1, outlier_ratio=0.):

        xdeq = torch.clone(x)

        xint = torch.zeros(x.shape, dtype=torch.int)

        block_size = len(x) // nblocks

        for i in range(nblocks):

            xb = x[i*block_size:(i+1)*block_size]

            if self.qtype=='nonuniform':
                if alg == 'iterative':
                    self.fit_nonuniform_iterative(xb)
                elif alg == 'quantile':
                    self.fit_nonuniform_quantile(xb)
                elif alg == 'snr':
                    self.fit_nonuniform_analytic(xb)
            elif self.qtype=='uniform':
                if alg == 'iterative':
                    self.fit_uniform_iterative(xb)
                elif alg == 'minmax':
                    self.fit_uniform_minmax(xb)
                elif alg == 'snr':
                    self.fit_uniform_normal(xb)
            elif self.qtype=='float':
                if alg == 'iterative':
                    self.fit_float_iterative(xb)
                elif alg == 'minmax':
                    self.fit_float_minmax(xb)
                elif alg == 'snr':
                    self.fit_float_normal(xb)

            xint[i*block_size:(i+1)*block_size] = self.quant(xb)
            xdeq[i*block_size:(i+1)*block_size] = self.lk[xint[i*block_size:(i+1)*block_size]]

        return xdeq



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

