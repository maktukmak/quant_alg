import torch

X = torch.randn([8,8])
x = X.flatten()


def quant_absmax(x, n_bits):
    s = torch.max(torch.abs(x)) / (2**(n_bits-1)-1)
    k = torch.round(x / s)
    k = torch.clip(k, -2**(n_bits-1)-1, 2**(n_bits-1)-1)
    xq = k * s
    return xq



class quant_prob():
    def __init__(self, n_bits, sigma2=0.1, alpha=2):

        self.sigma2 = torch.tensor(sigma2)
        self.alpha = alpha
        self.n_bits = n_bits

        self.k_list = torch.arange(-2**(n_bits-1)+1, 2**(n_bits-1))

        self.s = torch.max(torch.abs(x)) / (2**(n_bits-1)-1)
        self.pi = torch.ones(2**n_bits-1) / (2**n_bits-1)
        
    def  ll_normal(self, x):
        ll = -(x[:, None] - (self.s * self.k_list)[None])**2 / (2*self.sigma2)
        ll += - 0.5 * torch.log(self.sigma2)
        ll += - 0.5 * torch.log(torch.tensor(2*torch.pi))
        return  ll
    
    def e_step(self, x):
        ll_u = torch.exp(torch.log(self.pi) + self.ll_normal(x)) 
        r = ll_u / torch.sum(ll_u, axis = 1, keepdim=True)
        return r

    def ll_marginal(self, x):

        ll = torch.sum(self.pi * self.ll_normal(x))
        ll += torch.sum((self.alpha-1) * torch.log(self.pi))

        return ll

    def ell(self, x, r):
        ell = torch.sum(r * self.ll_normal(x))
        ell += torch.sum(r * torch.log(self.pi)[None]) 
        ell += torch.sum(r * torch.log(r)) 
        ell += torch.sum((self.alpha-1) * torch.log(self.pi))
        return ell

    def quant(self, x):
        r = self.e_step(x)
        k = torch.argmax(r, axis=1) - 2**(self.n_bits-1) + 1
        return k
    
    def dequant(self, k):
        xq = k * self.s
        return xq

    def fit(self, x, epochs=100):

        for i in range(epochs):
            r = self.e_step(x)

            self.pi = (torch.sum(r, axis=0) + self.alpha - 1) / (len(r) + r.shape[1]*(self.alpha - 1))
            self.s = torch.sum(r * x[:, None] * self.k_list) / torch.sum((self.k_list**2) * r)
            self.sigma2 = torch.sum(r * (x[:, None] - (self.s * self.k_list)[None])**2) / torch.sum(r)

            print(self.ell(x, r))


module = quant_prob(n_bits=4)
module.fit(x)

xq = module.dequant(module.quant(x))
xq_base = quant_absmax(x, n_bits=4)


print('Alg:', torch.mean((x - xq)**2))
print('Base:', torch.mean((x - xq_base)**2))
