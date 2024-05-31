import torch
torch.set_printoptions(precision=5)


class QLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias, levels, indexes):
        super(QLinear, self).__init__(in_features, out_features, bias is not None)
        self.weight = None
        self.levels = torch.nn.Parameter(levels, requires_grad=False)
        self.indexes = torch.nn.Parameter(indexes, requires_grad=False)
        if bias is not  None:
            self.bias = torch.nn.Parameter(bias.data)

    def forward(self, input):
        self.weight = torch.nn.Parameter(self.levels[self.indexes.to(torch.int32)])
        output = super().forward(input)
        self.weight = None
        return output


def float_grid(E=8, M=10, b=15, special=0):

    Gn = [2**(k // 2**M) * 2**(-b) * (1 + k % (2**M) * 2**(-M)) for k in range(2**M, 2**(M+E)-1-special)]
    Gs = [2**(-b) * (k * 2**(1-M)) for k in range(1, 2**M)]
    G = torch.tensor(Gs + Gn)
    return G

def cast_to_fp8(X, dtype='e4m3'):

    if dtype=='e4m3':
        #E4M3
        b = 7
        E = 4
        M = 3
        max_e = 2**E-1
        max_fn = 2**M-2
        max_fs = 2**M-1
        special = 0
    elif dtype=='e5m2':
        #E5M2
        b = 15
        E = 5
        M = 2
        max_e = 2**E-2
        max_fn = 2**M-1
        max_fs = 2**M-1
        special = 3
    else:
        return None

    v = 2**(torch.floor(torch.log2(torch.abs(X)) + 2**(-b)) - M)
    v[torch.floor(torch.log2(torch.abs(X)) + b) < 1] = 2**(1-M-b)
    Xf = v * torch.round(X / v)
    #Xf = torch.sign(X.flatten()) * G[torch.argmin((torch.abs(X.flatten()[:,None]) - G[None]) ** 2, axis = 1)]
    return Xf