
import torch

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.quantizer_weight import quantizer_weight
from src.utils import cast_to_fp8


def test_fp8_cast():
    m = 2048

    module = quantizer_weight(b=8, qtype='float', format_fp8='e4m3')

    X = torch.max(module.G) * torch.randn([m,m])
    x = X.flatten()

    module.fit_float_cast(x)
    xf = module.cast_to_fp8(x).to(torch.float16)

    xt = x.to(torch.float8_e4m3fn).to(torch.float16)
    ind = torch.isfinite(xt) #Torch makes saturated values nan
    assert torch.equal(xf[ind], xt[ind])


def test_fp8_quant():
    
    m = 1024

    module = quantizer_weight(b=8, qtype='float', format_fp8='e4m3')

    X = torch.sqrt(torch.max(module.G)) * torch.randn([m,m])
    x = X.flatten()

    module.fit_float_cast(x)
    xf = module.lk[module.quant(x)].to(torch.float16)

    # Exlcude equal distance to the grid 
    d = (x[:,None] - module.lk[None])**2
    topd = torch.topk(d, k=2, dim=1, largest=False)[0]
    ind = torch.where(torch.eq(topd[:,0], topd[:,1]) == False)[0]

    xt = x.to(torch.float8_e4m3fn).to(torch.float16)
    assert torch.equal(xf[ind], xt[ind])