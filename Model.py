import torch
from torch import nn
from collections import OrderedDict

class TinySleepNet(nn.Module):
    def __init__(self, seqLength=20):
        super().__init__()
        self.seqLength = seqLength
        self.cnn = nn.Sequential(
            nn.ConstantPad1d((22, 22), 0), # (a) 6*499+50=3044 (b) (3044-3000)/2=22
            nn.Sequential(OrderedDict([
                ('conv1', 
                 nn.Conv1d(in_channels=1, out_channels=128, kernel_size=50, stride=6, bias=False)),
            ])), # feature length = 500
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d((2, 2), 0), # (a) 8*62+8=504 (b) (504-500)/2=2
            nn.MaxPool1d(kernel_size=8, stride=8), # feature length = 63
            nn.Dropout(p=0.5),
            nn.ConstantPad1d((3, 4), 0), # (a) 1*62+8=70 (b) 70-63=7
            nn.Sequential(OrderedDict([
                ('conv2',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])), # feature length = 63
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d((3, 4), 0),
            nn.Sequential(OrderedDict([
                ('conv3',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])), # feature length = 63
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d((3, 4), 0),
            nn.Sequential(OrderedDict([
                ('conv4',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])), # feature length = 63
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d((0, 1), 0), # (a) 4*15+4=64 (b) 64-63=1
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(), # 16*128=2048
            nn.Dropout(p=0.5)
        )
        self.rnn = nn.LSTM(input_size=2048, hidden_size=128, num_layers=1, batch_first=True)
        self.rnnDropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, 5)
    
    def forward(self, input, lstmReset):
        # Input size is (batch size*sequence length, 1, sampling rate*30s)
        x = self.cnn(input)
        x = torch.reshape(x, (-1, self.seqLength, 2048))
        if lstmReset:
            hiddenStates = (
                torch.zeros(1*1, x.shape[0], 128, device=x.device),
                torch.zeros(1*1, x.shape[0], 128, device=x.device),
            )
        else:
            hiddenStates = (
                self.returnStates[0][:, :x.shape[0]],
                self.returnStates[1][:, :x.shape[0]],
            )
        x, self.returnStates = self.rnn(x, hiddenStates)
        self.returnStates = (self.returnStates[0].detach(), self.returnStates[1].detach())
        x = self.rnnDropout(x)
        y = self.fc(torch.reshape(x, (-1, 128)))
        return y