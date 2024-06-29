
import torch

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.quantizer_weight import quantizer_weight
from src.utils import cast_to_fp8
import pytest


@pytest.mark.parametrize("format_fp16", ['fp'])
def test_fp16_cast(format_fp16):
    m = 4096

    module = quantizer_weight(b=16, qtype='float', format_fp16=format_fp16)

    X = torch.sqrt(torch.max(module.G)) * torch.randn([m,m])
    x = X.flatten()

    # Rounding
    xdeqr = module.fit_and_quant(x, alg='cast').to(torch.float16)

    # Torch native
    if format_fp16 == 'fp':
        xt = x.to(torch.float16)


    cond1 = torch.equal(xdeqr, xt) # Rounding vs torch
    cond3 = torch.all(torch.isin(xdeqr.unique(), module.G)) # Grid points vs torch

    assert cond1 and cond3



@pytest.mark.parametrize("format_fp8", ['e4m3', 'e5m2'])
def test_fp8_cast(format_fp8):
    m = 4096

    module = quantizer_weight(b=8, qtype='float', format_fp8=format_fp8)

    X = torch.sqrt(torch.max(module.G)) * torch.randn([m,m])
    x = X.flatten()

    # Rounding
    xdeqr = module.fit_and_quant(x, alg='cast').to(torch.float16)

    # Distance-based quant
    lk = module.compute_quant_levels()
    xdeqd = lk[torch.argmin((x[:,None] - lk[None])**2, axis=1)]

    # Torch native
    if format_fp8 == 'e4m3':
        xt = x.to(torch.float8_e4m3fn).to(torch.float16)
    else:
        xt = x.to(torch.float8_e5m2).to(torch.float16)

    # Exlcude equal distance to the grid 
    lk = module.G * module.s + module.z
    d = (x[:,None] - lk[None])**2
    topd = torch.topk(d, k=2, dim=1, largest=False)[0]
    inde = torch.eq(topd[:,0], topd[:,1]) == False

    #Exclude saturated values (torch makes nan)
    indn = torch.isfinite(xt) 
    ind = inde * indn

    cond1 = torch.equal(xdeqr[ind], xt[ind]) # Rounding vs torch
    cond2 = torch.equal(xdeqd[ind], xt[ind]) # Clustering vs torch
    cond3 = torch.all(torch.isin(xdeqr[ind].unique(), module.G)) # Grid points vs torch

    assert cond1 and cond2 and cond3

@pytest.mark.parametrize("format_fp4", ['e2m1', 'e3m0'])
def test_fp4_cast(format_fp4):
    
    m = 4096

    module = quantizer_weight(b=4, qtype='float', format_fp8='e2m1')

    X = torch.sqrt(torch.max(module.G)) * torch.randn([m,m])
    x = X.flatten()

    # Rounding
    xdeqr = module.fit_and_quant(x, alg='cast').to(torch.float16)

    # Distance-based
    lk = module.compute_quant_levels()
    xdeqd = lk[torch.argmin((x[:,None] - lk[None])**2, axis=1)]

    cond1 = torch.equal(xdeqr, xdeqd)

    assert cond1