import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.quantizer_model import quantizer_model
from src.quantizer_weight import quantizer_weight
import matplotlib.pyplot as plt
from functools import reduce
from copy import deepcopy
torch.manual_seed(0)

mode = 'QAT'   # QAT, PTQ
b = 4
qtype = 'float'
alg = 'minmax'
prox = True
lamda = 0
quantize_act = False
EPOCHS = 300

nsamples = 2000
batch_size = 32
dim = 10

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(dim, dim,bias=False)
        self.fc2 = nn.Linear(dim, 1,bias=False)
        self.quantizer = quantizer_weight(b, f=None, qtype=qtype, format_fp4='e3m0', format_fp8='e5m2', format_fp16='fp')

    def forward(self, x):
        if quantize_act:
            x = self.quantizer.fit_and_quant(x.flatten(), alg).reshape(x.shape)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        # if quantize_act:
        #     x = self.quantizer.fit_and_quant(x.flatten(), alg).reshape(x.shape)
        x = self.fc2(x)
        return x
    
def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)



model = LinearModel()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001 )


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
            if prox:
                model_unq = deepcopy(model)
            quantizer.fit_weight(model, 
                                fisher=None, 
                                qtype=qtype, 
                                block_size=None, 
                                alg=alg, 
                                decompose_outlier=None)
            if prox:
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        wq = module.weight.data
                        w = get_module_by_name(model_unq, name).weight.data
                        module.weight.data =  (w + lamda * wq) / (1 + lamda)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()



        running_loss += loss.item()

    return running_loss, i

def val_eval(model):
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

tloss_vec = []
vloss_vec = []
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, i = train_one_epoch(epoch_number)
    tloss_vec.append(avg_loss/ (i + 1))

    model.eval()
    # Quantize before evaluation (QAT)
    modelq = deepcopy(model)
    if mode == 'QAT':
        quantizer.fit_weight(modelq, 
                                fisher=None, 
                                qtype=qtype, 
                                block_size=None, 
                                alg=alg, 
                                decompose_outlier=None)


    running_vloss, i  = val_eval(modelq)
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    vloss_vec.append(avg_vloss)

    epoch_number += 1



plt.plot(tloss_vec, label='train error')
#plt.plot(vloss_vec, label='val error')
plt.legend()
plt.show()

quantizer.fit_weight(model, 
                    fisher=None, 
                    qtype=qtype, 
                    block_size=None, 
                    alg=alg, 
                    decompose_outlier=None)

running_vloss, i  = val_eval(model)
avg_vloss = running_vloss / (i + 1)
print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))


print(W)
print(model.fc2.weight)
