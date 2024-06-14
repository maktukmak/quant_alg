
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

        #if self.qtype == 'uniform':
        self.k_list = torch.arange(0, self.N-1).to(torch.int)

        if self.qtype == 'nonuniform':
            self.sk_kmeans = True
            
        self.block_size = None

        if self.qtype == 'float':
            if self.b == 8:
                if format_fp8=='e4m3':
                    self.E=4; self.M=3; self.bias=7; self.c=0
                elif format_fp8=='e5m2':
                    self.E=5; self.M=2; self.bias=15; self.c=3
            elif self.b == 4:
                if format_fp4=='e2m1':
                    self.E=2; self.M=1; self.bias=2; self.c=0
                elif format_fp4=='e3m0':
                    self.E=3; self.M=0; self.bias=3; self.c=0

            self.float_grid(self.E, self.M, self.bias, self.c)

    def error(self, x, xdeq):
        if not hasattr(self, 'f'):
            self.f = torch.ones(x.shape)
        #err = torch.sum(((x - self.lk[xint])**2) * self.f) / torch.sum(self.f)
        err = torch.sum(((x - xdeq)**2)) / len(x)
        return err
    
    def compute_quant_levels(self):
        if self.qtype == 'uniform':
            lk = self.s * self.k_list + self.z
        elif self.qtype == 'nonuniform':
            lk = self.lk
        elif self.qtype == 'float':
            lk = self.G * self.s + self.z
        return lk

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

    def quant_nonuniform(self, x, lk):
        return torch.argmin((x[:,None] - lk)**2, axis=1).to(torch.int)
   
    def dequant_nonuniform(self, xint, lk):
        return lk[torch.arange(len(xint)), xint]

    def quant_uniform(self, x, s, z):
        return torch.clamp(torch.round((x - z) / s), 0, self.N-2).to(torch.int)

    def dequant_uniform(self, xint, s, z):
        return s * xint + z

    def quant_float(self, x, s, z):
        return self.cast_to_fp((x - z)/s) 

    def dequant_float(self, xfloat, s, z):
        return s * xfloat + z

    def cast_to_fp(self, x):

        x = torch.clamp(x, self.G[0], self.G[-1])

        v = 2**(torch.floor(torch.log2(torch.abs(x)) + 2**(-self.bias)) - self.M)
        v[torch.floor(torch.log2(torch.abs(x)) + self.bias) < 1] = 2**(1-self.M-self.bias)
        
        Xf = v * torch.round(x / v)
        
        return Xf

    def float_grid(self, E=8, M=10, bias=15, special=0):

        Gn = [2**(k // 2**M) * 2**(-bias) * (1 + k % (2**M) * 2**(-M)) for k in range(2**M, 2**(M+E)-1-special)]
        Gs = [2**(-bias) * (k * 2**(1-M)) for k in range(1, 2**M)]
        self.Gh = torch.tensor(Gs + Gn)
        self.G = torch.concat((-torch.flip(self.Gh, [0]), torch.tensor([0.]), self.Gh))

    def gauss_cdf(self, x, m, std):
        return 0.5 * (1 + torch.erf((x - m) / (torch.sqrt(torch.tensor(2.)) * std)))
    
    def fit_float_cast(self, x):

        self.s = torch.tensor(1.)
        self.z = torch.tensor(0.)
        #self.lk = self.G * self.s + self.z

    def fit_float_minmax(self, x):
        
        self.minx = torch.min(x.reshape(-1, self.block_size), axis=1)[0]
        self.maxx = torch.max(x.reshape(-1, self.block_size), axis=1)[0]

        self.s = (self.maxx - self.minx) / (2*torch.max(self.G))
        self.z = self.minx + torch.max(self.G) * self.s
        #self.lk = self.G * self.s + self.z

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

        xmean = torch.mean(x.reshape(-1, self.block_size), axis=1)
        xvar = torch.var(x.reshape(-1, self.block_size), axis=1)

        self.s = torch.sqrt(xvar / self.sigma2opt)
        self.z = xmean
        #self.lk = self.G * self.s + self.z

    def fit_float_iterative(self, x, f=None, verbose=False):

        if f is not None:
            self.f = f

        self.fit_float_normal(x)

        denum_z = len(x)
        nblocks = len(x) // self.block_size

        for i in range(nblocks):

            xb = x[i*self.block_size:(i+1)*self.block_size]
            s,z = self.s[i], self.z[i]

            sz_prev = torch.zeros(2)
            for _ in range(2000):
                xfloat = self.quant_float(xb, s, z)
                #xfloat = self.G[xint]

                sz = torch.tensor([s, z])

                num_s = torch.sum((xb - z) * xfloat)
                denum_s = torch.sum(xfloat**2)
                num_z = torch.sum(xb - s * xfloat)

                s = num_s / denum_s
                z = num_z / denum_z

                self.s[i] = s
                self.z[i] = z

                if torch.sum(torch.abs(sz_prev - sz)) / torch.sum(torch.abs(sz)) < 1e-5:
                    #print('Loss', err)
                    return
                sz_prev = sz
            print('NOT CONVERGED !!!')
            print(sz)

    def fit_uniform_minmax(self, x):
        
        self.minx = torch.min(x.reshape(-1, self.block_size), axis=1)[0]
        self.maxx = torch.max(x.reshape(-1, self.block_size), axis=1)[0]

        self.s = (self.maxx - self.minx) / (self.N-1)
        self.z = self.minx + self.s/2
        #self.lk = self.s[:,None] * self.k_list  + self.z[:,None]


    def fit_uniform_normal(self, x):

        if not hasattr(self, 'zeta'):
            z = np.linspace(1,100,10000)
            gres = self.snr_uni(z, self.N)
            self.zeta = z[np.argmax(gres)]

        xmean = torch.mean(x.reshape(-1, self.block_size), axis=1)
        xvar = torch.var(x.reshape(-1, self.block_size), axis=1)

        self.s = (2 * np.sqrt(self.zeta * xvar)) / (self.N-1)
        self.z = - self.s * (self.N/2 - 1) + xmean
        #self.lk = self.s * self.k_list + self.z
    
    def fit_uniform_iterative(self, x, verbose=False):

        self.fit_uniform_normal(x)

        nblocks = len(x) // self.block_size

        for i in range(nblocks):

            xb = x[i*self.block_size:(i+1)*self.block_size]
            s,z = self.s[i], self.z[i]

            denum_z = len(xb)
            sz_prev = torch.zeros(2)
            for _ in range(2000):

                xint = self.quant_uniform(xb, s, z)

                sz = torch.tensor([s, z])

                num_s = torch.sum((xb - z) * xint)
                denum_s = torch.sum(xint**2)
                num_z = torch.sum(xb - s * xint)

                s = num_s / denum_s
                z = num_z / denum_z

                self.s[i] = s
                self.z[i] = z
                
                #self.lk = self.s * self.k_list + self.z

                if torch.sum(torch.abs(sz_prev - sz)) / torch.sum(torch.abs(sz)) < 1e-5:
                    #print('Loss', err)
                    return
                sz_prev = sz
            print('NOT CONVERGED !!!')
            print(sz)

    def fit_nonuniform_quantile(self, x):

        nblocks = len(x) // self.block_size

        x_sorted = torch.sort(x.reshape(-1, self.block_size), axis=1)[0]
        k = torch.arange(0, self.N)
        ind = ((self.block_size-1) * k.to(torch.float64) / (self.N-1)).to(torch.int)
        self.tk = x_sorted[:, ind]
        self.tk[:,0] = torch.nan_to_num(torch.tensor(-float('inf')))
        self.tk[:,-1] = torch.nan_to_num(torch.tensor(float('inf')))
        self.lk = torch.zeros(nblocks, self.N-1)
        for i in k-1:
            self.lk[:,i] = torch.mean(x_sorted[:,ind[i]:ind[i+1]], axis=1)

    def fit_nonuniform_analytic(self, x):

        if not hasattr(self, 'lkopt'):
            tk = torch.quantile(torch.randn(1000), torch.arange(self.N)/(self.N-1))
            tk[0] = torch.nan_to_num(torch.tensor(-float('inf')))
            tk[-1] = torch.nan_to_num(torch.tensor(float('inf')))
            pdf = torch.distributions.normal.Normal(0., 1.)
            tk = tk.to(torch.float64)

            for _ in range(500):

                F = self.gauss_cdf(tk, 0., 1.)
                P = torch.exp(pdf.log_prob(torch.Tensor(tk)))
                
                lk = -(P[1:] - P[:-1]) / (F[1:] - F[:-1])
                tk[1:-1] = (lk[1:] + lk[0:-1])/2

            self.lkopt = lk.to(torch.float32)
            self.tkopt = tk.to(torch.float32)

        xmean = torch.mean(x.reshape(-1, self.block_size), axis=1)
        xstd = torch.std(x.reshape(-1, self.block_size), axis=1)

        self.lk = (xstd[:,None] * self.lkopt + xmean[:,None])
        #self.tk = x.mean()


    def fit_nonuniform_iterative(self, x, verbose=False):

        self.fit_nonuniform_analytic(x)
        #self.fit_uniform_minmax(x)

        if not hasattr(self, 'f'):
            self.f = torch.ones(x.shape)

        nblocks = len(x) // self.block_size

        for i in range(nblocks):

            xb = x[i*self.block_size:(i+1)*self.block_size]
            lk = self.lk[i]

            kmeans = KMeans(n_clusters=self.N-1, init=lk.reshape(-1,1), max_iter=500, tol=1e-5)
            kmeans.fit(xb.reshape(-1,1), self.f, )
            self.lk[i] = torch.tensor(kmeans.cluster_centers_[:,0])

    
    def fit_and_quant(self, x, alg, block_size=None, decompose_outlier=False):



        xdeq = torch.clone(x)

        ind_nonoutlier = torch.arange(len(x), dtype=torch.int)
        if decompose_outlier:
            ind_nonoutlier = torch.where(abs(x-x.mean()) < 3*x.std())[0]
            x = x[ind_nonoutlier]

        if block_size:
            self.block_size=block_size
        else:
            self.block_size=len(x)


        if self.qtype=='nonuniform':
            if alg == 'iterative':
                self.fit_nonuniform_iterative(x)
            elif alg == 'quantile':
                self.fit_nonuniform_quantile(x)
            elif alg == 'snr':
                self.fit_nonuniform_analytic(x)
            xdeq[ind_nonoutlier] = self.dequant_nonuniform(self.quant_nonuniform(x, torch.repeat_interleave(self.lk, self.block_size,axis=0)),
                                                                                torch.repeat_interleave(self.lk, self.block_size,axis=0))

        elif self.qtype=='uniform':
            if alg == 'iterative':
                self.fit_uniform_iterative(x)
            elif alg == 'minmax':
                self.fit_uniform_minmax(x)
            elif alg == 'snr':
                self.fit_uniform_normal(x)
            xdeq[ind_nonoutlier] = self.dequant_uniform(self.quant_uniform(x, torch.repeat_interleave(self.s, self.block_size),
                                                                        torch.repeat_interleave(self.z, self.block_size)),
                                                                         torch.repeat_interleave(self.s, self.block_size),
                                                                           torch.repeat_interleave(self.z, self.block_size))

        elif self.qtype=='float':
            if alg == 'iterative':
                self.fit_float_iterative(x)
            elif alg == 'minmax':
                self.fit_float_minmax(x)
            elif alg == 'snr':
                self.fit_float_normal(x)
            elif alg == 'cast':
                self.fit_float_cast(x)
            xdeq[ind_nonoutlier] = self.dequant_float(self.quant_float(x, torch.repeat_interleave(self.s, self.block_size),
                                                                        torch.repeat_interleave(self.z, self.block_size)),
                                                                         torch.repeat_interleave(self.s, self.block_size),
                                                                           torch.repeat_interleave(self.z, self.block_size))


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

