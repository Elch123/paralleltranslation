import torch
import torch.nn as nn

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
