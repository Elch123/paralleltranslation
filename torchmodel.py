import torch
import torch.nn as nn

class StandardBlock(nn.Module):
    def __init__(self,params):
        self.params=params
        super().__init__()
        self.conv1=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 3,padding=1)
        self.bn1=torch.nn.BatchNorm1d(params['num_hidden'])
        self.conv2=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 3,padding=1)
        self.bn2=torch.nn.BatchNorm1d(params['num_hidden'])
        self.relu=torch.nn.ReLU()

    def forward(self,x):
        if(self.params['batchnorm']):
            x=self.bn1(x)
        x=self.relu(x)
        x=self.conv1(x)
        if(self.params['batchnorm']):
            x=self.bn2(x)
        x=self.relu(x)
        x=self.conv2(x)
        return x
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

class Embedder(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.embed=torch.nn.Embedding(4000,params['embed_size'])
    def forward(self,x):
        embedded=self.embed(x)
        embedded=embedded.permute(0,2,1)
        return embedded
class PosEncodeLearned(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.encoding=torch.zeros(shape=(1,params['embed_size'],params['max_seqlen']))
    def forward(self,x):
        encoding_slice=encoding[1,:,x.shape[2]]
        print(encoding_slice.shape)
        return encoding_slice+x
class BasicAttention(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.keyconv=torch.nn.Conv1d(params['num_hidden'],params['attnsize'], 1)
        self.queriesconv=torch.nn.Conv1d(params['num_hidden'],params['attnsize'], 1)
        self.softmax=torch.nn.Softmax(dim=-1)
        self.project=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 1)
    def forward(self,x):
        keys=self.keyconv(x)
        #print(keys.shape)
        queries=self.queriesconv(x)
        #print(queries.shape)
        values=torch.matmul(keys.permute(0,2,1),queries)
        #print(values.shape)
        seqlen=x.shape[2]
        attn=self.softmax(values/(seqlen**1/2))
        #print(x.shape)
        #print(attn.shape)
        out=torch.matmul(x,attn)
        out=self.project(out)
        return out
class AttnResBlock(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.attn=BasicAttention(params)
        self.block=ResidualBlock(params)
    def forward(self,x):
        res=x
        x=self.attn(x)
        x+=res
        x=self.block(x)
        return x


class ResNet(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.embed=torch.nn.Embedding(4000,params['num_hidden'])
        upsample=[]
        if(params['upsample']):
            upsample.append(torch.nn.ConvTranspose1d(params['num_hidden'],params['num_hidden'],2,stride=2))
            #upsample.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))
        layers=[torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 3,padding=1),
        ResidualBlock(params),
        ResidualBlock(params),
        ResidualBlock(params)]+upsample+[ResidualBlock(params),
        ResidualBlock(params),
        ResidualBlock(params),
        torch.nn.ReLU(),
        torch.nn.Conv1d(params['num_hidden'],4000, 5,padding=2)]
        self.model = torch.nn.Sequential(*layers)
    def forward(self,x):
        embedded=self.embed(x)
        embedded=embedded.permute(0,2,1)
        out=self.model(embedded)
        return out

class AttnResNet(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.embed=torch.nn.Embedding(4000,params['num_hidden'])
        upsample=[]
        if(params['upsample']):
            upsample.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))
        layers=[torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 3,padding=1),
        AttnResBlock(params),
        AttnResBlock(params),
        AttnResBlock(params)]+upsample+[AttnResBlock(params),
        AttnResBlock(params),
        AttnResBlock(params),
        torch.nn.ReLU(),
        torch.nn.Conv1d(params['num_hidden'],4000, 5,padding=2)]
        self.model = torch.nn.Sequential(*layers)
    def forward(self,x):
        embedded=self.embed(x)
        embedded=embedded.permute(0,2,1)
        out=self.model(embedded)
        return out
