import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.quantizer_model import quantizer_model


class LinearModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(in_size, out_size, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x

batch_size = 16 # Try, for example, 128, 256, 513.
in_size = 10
out_size = 1
num_layers = 3
num_batches = 200
epochs = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

W = torch.randn((out_size, in_size))
data = [torch.randn(batch_size, in_size) for _ in range(num_batches)]
targets = [data[i] @ W.T + torch.randn(batch_size, out_size) for i in range(num_batches)]

model = LinearModel(in_size, out_size).cuda()
loss_fn = torch.nn.MSELoss().cuda()

use_amp = True
opt = torch.optim.SGD(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            output = model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # set_to_none=True here can modestly improve performance

print(model.fc.weight)
print(W)

