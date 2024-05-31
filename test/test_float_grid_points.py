
import torch
import pytest
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.quantizer_weight import quantizer_weight


@pytest.mark.parametrize("b, format_fp4", [(4, 'e2m1'), (4, 'e3m0'), (8, 'e4m3'), (8, 'e5m2')])
def test_fp4_grid(b, format_fp4):
    m = 2048

    module = quantizer_weight(b=b, qtype='float', format_fp4=format_fp4)

    X = torch.max(module.G) * torch.randn([m,m])
    x = X.flatten()

    module.fit_float_normal(x)

    Gp = torch.hstack([torch.arange(module.xr[r], module.xr[r+1], module.vr[r]) for r in range(len(module.vr))])
    Gp = torch.concat((Gp, torch.tensor([Gp[-1] + module.vr[-1]])))

    assert torch.equal(Gp, module.G)
