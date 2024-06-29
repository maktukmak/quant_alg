import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.quantizer_model import quantizer_model
from src.quantizer_weight import quantizer_weight
torch.manual_seed(0)

mode = 'QAT'   # QAT, PTQ
b = 4
qtype = 'float'
alg = 'minmax'
quantize_act = True
EPOCHS = 300

nsamples = 2000
batch_size = 32
dim = 10

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(10, 10,bias=False)
        self.fc2 = nn.Linear(10, 1,bias=False)
        self.quantizer = quantizer_weight(b, f=None, qtype=qtype, format_fp4='e3m0', format_fp8='e4m3', format_fp16='fp')

    def forward(self, x):
        if quantize_act:
            x = self.quantizer.fit_and_quant(x.flatten(), alg).reshape(x.shape)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        # if quantize_act:
        #     x = self.quantizer.fit_and_quant(x.flatten(), alg).reshape(x.shape)
        x = self.fc2(x)
        return x


model = LinearModel()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


X = torch.randn((nsamples, dim))
W = torch.randn((1,dim))
Y = X @ W.T +  torch.randn((nsamples,1))
my_dataset = TensorDataset(X,Y)
train_size = int(0.8 * len(X))
val_size = len(X) - train_size
data_train, data_val = random_split(my_dataset, [train_size, val_size])
training_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)




quantizer = quantizer_model(b=b)


def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):

        inputs, labels = data

        if mode == 'QAT':
            quantizer.fit_weight(model, 
                                fisher=None, 
                                qtype=qtype, 
                                block_size=None, 
                                alg=alg, 
                                decompose_outlier=None)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return running_loss

def val_eval():
    running_vloss = 0.0
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
    return running_vloss, i

epoch_number = 0
best_vloss = 1_000_000.


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)


    # Quantize before evaluation (QAT)
    if mode == 'QAT':
        quantizer.fit_weight(model, 
                                fisher=None, 
                                qtype=qtype, 
                                block_size=None, 
                                alg=alg, 
                                decompose_outlier=None)

    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    running_vloss, i  = val_eval()
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    epoch_number += 1


quantizer.fit_weight(model, 
                    fisher=None, 
                    qtype=qtype, 
                    block_size=None, 
                    alg=alg, 
                    decompose_outlier=None)

running_vloss, i  = val_eval()
avg_vloss = running_vloss / (i + 1)
print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))


print(W)
print(model.fc2.weight)
