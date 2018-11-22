import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self,params):
        self.params=params
        super().__init__()
        self.conv1=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 3,padding=1)
        self.bn1=torch.nn.BatchNorm1d(params['num_hidden'])
        self.conv2=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 3,padding=1)
        self.bn2=torch.nn.BatchNorm1d(params['num_hidden'])
        self.relu=torch.nn.ReLU()

    def forward(self,x):
        residual=x
        if(self.params['batchnorm']):
            x=self.bn1(x)
        x=self.relu(x)
        x=self.conv1(x)
        if(self.params['batchnorm']):
            x=self.bn2(x)
        x=self.relu(x)
        x=self.conv2(x)
        x+=residual
        return x
class Cnn(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.embed=torch.nn.Embedding(4000,params['num_hidden'])
        self.model = torch.nn.Sequential(
        torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 5,padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 5,padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv1d(params['num_hidden'],4000, 5,padding=2))
    def forward(self,x):
        #print(x.shape)
        embedded=self.embed(x)
        #print(embedded.shape)
        embedded=embedded.permute(0,2,1)
        #print(embedded.shape)
        out=self.model(embedded)
        #print(out.shape)
        return out

class ResNet(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.embed=torch.nn.Embedding(4000,params['num_hidden'])
        self.model = torch.nn.Sequential(
        torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 5,padding=2),
        ResidualBlock(params),
        ResidualBlock(params),
        ResidualBlock(params),
        ResidualBlock(params),
        torch.nn.ReLU(),
        torch.nn.Conv1d(params['num_hidden'],4000, 5,padding=2))
    def forward(self,x):
        embedded=self.embed(x)
        embedded=embedded.permute(0,2,1)
        out=self.model(embedded)
        return out
