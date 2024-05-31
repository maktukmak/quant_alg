import torch
from src.quantizer_weight import quantizer_weight
import matplotlib.pyplot as plt

m = 64
#X = 2 * torch.rand([m,m])
X = 3*torch.randn([m,m])
x = X.flatten()
b = 4


quantizer = quantizer_weight(b=4, qtype='float', format_fp4='e3m0')
quantizer.fit_float_normal(x)
C = quantizer.G[-1]
sigma2 = torch.linspace((C/10)**2,C**2,100000)
res4 = quantizer.snr_float(C, sigma2, quantizer.xr, quantizer.vr)
zeta4 = sigma2[torch.argmax(res4)]



# plt.plot(xr, 1.05*torch.ones(xr.shape), linestyle='--', marker='o', color='b', label='line with marker')
# plt.plot(G, torch.ones(G.shape), linestyle='--', marker='o', color='r', label='line with marker')
# plt.ylim(0.75,1.25)
# plt.grid()
# plt.show()

plt.plot(C / torch.sqrt(sigma2), res4, label='4-bit float (E3M0)')
#plt.plot(torch.sqrt(z), res8, label='8-bit')
# plt.yscale('log')
# plt.xlabel('Sigma2')
# plt.legend()
# plt.grid()
# plt.ylabel('SNR')
# plt.show()



quantizer = quantizer_weight(b=4, qtype='float', format_fp4='e2m1')
quantizer.fit_float_normal(x)
C = quantizer.G[-1]
sigma2 = torch.linspace((C/10)**2,C**2,100000)
res4 = quantizer.snr_float(C, sigma2, quantizer.xr, quantizer.vr)
zeta4 = sigma2[torch.argmax(res4)]
plt.plot(C / torch.sqrt(sigma2), res4, label='4-bit float (E2M1)')


quantizer = quantizer_weight(b=4)

z = torch.linspace(1,100,10000)

res4 = quantizer.snr_uni(z, int(2**4))
res3 = quantizer.snr_uni(z, int(2**3))
res2 = quantizer.snr_uni(z, int(2**2))
res8 = quantizer.snr_uni(z, int(2**8))

plt.plot(torch.sqrt(z), res4, label='4-bit uniform', linestyle='--')
plt.plot(torch.sqrt(z), res3, label='3-bit uniform', linestyle='--')
plt.plot(torch.sqrt(z), res2, label='2-bit uniform', linestyle='--')
#plt.plot(torch.sqrt(z), res8, label='8-bit uniform')

plt.yscale('log')
plt.xlabel('Clipping point')
plt.legend()
plt.grid()
plt.ylabel('SNR')
plt.savefig('snr_vs_clip.jpg', bbox_inches='tight')
plt.show()





