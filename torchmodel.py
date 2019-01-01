import torch
import torch.nn as nn
import math

class StandardBlock(nn.Module):
    def __init__(self,params):
        self.params=params
        super().__init__()
        self.conv1=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], params['filter_width'],padding=params['filter_width']//2)
        self.bn1=torch.nn.BatchNorm1d(params['num_hidden'])
        self.conv2=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], params['filter_width'],padding=params['filter_width']//2)
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
class TransformerConv(nn.Module):
    def __init__(self,params):
        self.params=params
        super().__init__()
        self.conv1=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], params['filter_width'],padding=params['filter_width']//2)
        self.bn1=torch.nn.BatchNorm1d(params['num_hidden'])
        self.conv2=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], params['filter_width'],padding=params['filter_width']//2)
        self.relu=torch.nn.ReLU()
    def forward(self,x):
        add=x
        x=self.conv1(x)
        if(self.params['batchnorm']):
            x=self.bn2(x)
        x=self.relu(x)
        x=self.conv2(x)
        x+=add
        return x
class ResidualBlock(nn.Module):
    def __init__(self,params):
        self.params=params
        super().__init__()
        self.conv1=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], params['filter_width'],padding=params['filter_width']//2)
        self.bn1=torch.nn.BatchNorm1d(params['num_hidden'])
        self.conv2=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], params['filter_width'],padding=params['filter_width']//2)
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
class InceptionBlock(nn.Module):
    def __init__(self,params):
        self.params=params
        super().__init__()
        self.reducea=torch.nn.Conv1d(params['num_hidden'],params['num_hidden']//4, 1)
        self.reduceb=torch.nn.Conv1d(params['num_hidden'],params['num_hidden']//4, 1)
        self.reducec=torch.nn.Conv1d(params['num_hidden'],params['num_hidden']//4, 1)
        self.conva=torch.nn.Conv1d(params['num_hidden']//4,params['num_hidden']//4, 3,padding=1)
        self.convb=torch.nn.Conv1d(params['num_hidden']//4,params['num_hidden']//4, 3,padding=1)
        self.convc=torch.nn.Conv1d(params['num_hidden']//4,params['num_hidden']//4, 3,padding=1)
        self.project=torch.nn.Conv1d(params['num_hidden']//4*3,params['num_hidden'], 1)
        self.activation=torch.nn.ReLU()

    def forward(self,x):
        residual=x
        x=self.activation(x)
        thina=self.activation(self.reducea(x))
        thinb=self.activation(self.reduceb(x))
        thinc=self.activation(self.reducec(x))
        proca=self.activation(self.conva(thinb))
        procb=self.activation(self.convb(thinc))
        procb=self.activation(self.convc(procb))
        combined=torch.cat([thina,proca,procb],dim=1)
        out=self.project(combined)
        out+=residual
        return out
class Cnn(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.embed=torch.nn.Embedding(params['symbols'],params['num_hidden'])
        self.model = torch.nn.Sequential(
        torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 5,padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 5,padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv1d(params['num_hidden'],params['symbols'], 5,padding=2))
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
        self.embed=torch.nn.Embedding(params['symbols'],params['embed_size'])
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
class PosEncoding(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.hparams=params
        pe = torch.zeros(1,params['num_hidden'],params['symbols_in_batch'])
        for p in range(params['symbols_in_batch']):
            for i in range(0, params['num_hidden'], 2):
                pe[0,i,p]=math.sin(p / (10000 ** ((2 * i)/params['num_hidden'])))
                pe[0,i+1,p]=math.cos(p / (10000 ** ((2 * i)/params['num_hidden'])))
        pe/=params['num_hidden']**1/2
        self.register_buffer('pe', pe)
        self.pe=pe.cuda()
    def forward(self,x):
        shape=x.shape
        x+=self.pe[:,:,0:shape[2]]
        return x
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
class MultiHeadAttention(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.hparams=params
        self.keyconv=torch.nn.Conv1d(params['num_hidden'],params['attnsize'], 1)
        self.queriesconv=torch.nn.Conv1d(params['num_hidden'],params['attnsize'], 1)
        self.softmax=torch.nn.Softmax(dim=-1)
        self.projecta=torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], 1)
    def split(self,x,shape):
        x=torch.reshape(x,(shape[0]*self.hparams['heads'],-1,shape[2]))
        return x
    def join(self,x,shape):
        x=torch.reshape(x,(shape[0],-1,shape[2]))
        return x
    def forward(self,x):
        shape=x.shape
        keys=self.keyconv(x)
        keys=self.split(keys,shape)
        queries=self.queriesconv(x)
        queries=self.split(queries,shape)
        values=torch.matmul(keys.permute(0,2,1),queries)
        seqlen=shape[2]
        attn=self.softmax(values/(seqlen**1/2))
        x=self.split(x,shape)
        out=torch.matmul(x,attn)
        out=self.join(out,shape)
        out=self.projecta(out)
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

class AdvancedBlock(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.attn=MultiHeadAttention(params)
        self.block=ResidualBlock(params)
    def forward(self,x):
        res=x
        x=self.attn(x)
        x+=res
        x=self.block(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.attn=MultiHeadAttention(params)
        self.block=TransformerConv(params)
    def forward(self,x):
        res=x
        x=self.attn(x)
        x+=res
        x=self.block(x)
        return x
class ResNet(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.embed=torch.nn.Embedding(params['symbols'],params['num_hidden'])
        upsample=[]
        if(params['upsample']):
            upsample.append(torch.nn.ConvTranspose1d(params['num_hidden'],params['num_hidden'],2,stride=2))
            #upsample.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))
        layers=[torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], params['filter_width'],padding=params['filter_width']//2),
        ResidualBlock(params),
        ResidualBlock(params),
        ResidualBlock(params)]+upsample+[ResidualBlock(params),
        ResidualBlock(params),
        ResidualBlock(params),
        torch.nn.ReLU(),
        torch.nn.Conv1d(params['num_hidden'],params['symbols'], params['filter_width'],padding=params['filter_width']//2)]
        self.model = torch.nn.Sequential(*layers)
    def forward(self,x):
        embedded=self.embed(x)
        embedded=embedded.permute(0,2,1)
        out=self.model(embedded)
        return out

class AttnResNet(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.embed=torch.nn.Embedding(params['symbols'],params['num_hidden'])
        upsample=[]
        if(params['upsample']):
            upsample.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))
        layers=[torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], params['filter_width'],padding=params['filter_width']//2),
        AttnResBlock(params),
        AttnResBlock(params),
        AttnResBlock(params)]+upsample+[AttnResBlock(params),
        AttnResBlock(params),
        AttnResBlock(params),
        torch.nn.ReLU(),
        torch.nn.Conv1d(params['num_hidden'],params['symbols'], params['filter_width'],padding=params['filter_width']//2)]
        self.model = torch.nn.Sequential(*layers)
    def forward(self,x):
        embedded=self.embed(x)
        embedded=embedded.permute(0,2,1)
        out=self.model(embedded)
        return out

class AdvancedNet(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.embed=torch.nn.Embedding(params['symbols'],params['num_hidden'])
        upsample=[]
        layers=[
        torch.nn.Conv1d(params['num_hidden'],params['num_hidden'], params['filter_width'],padding=params['filter_width']//2),
        AdvancedBlock(params),
        AdvancedBlock(params),
        AdvancedBlock(params),
        AdvancedBlock(params),
        AdvancedBlock(params),
        torch.nn.ReLU(),
        torch.nn.Conv1d(params['num_hidden'],params['symbols'], params['filter_width'],padding=params['filter_width']//2)]
        self.model = torch.nn.Sequential(*layers)
    def forward(self,x):
        embedded=self.embed(x)
        embedded=embedded.permute(0,2,1)
        out=self.model(embedded)
        return out

class Transformer(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.embed=torch.nn.Embedding(params['symbols'],params['num_hidden'])
        params['filter_width']=1
        upsample=[]
        layers=[
        PosEncoding(params),
        TransformerBlock(params),
        TransformerBlock(params),
        TransformerBlock(params),
        TransformerBlock(params),
        TransformerBlock(params),
        TransformerBlock(params),
        torch.nn.ReLU(),
        torch.nn.Conv1d(params['num_hidden'],params['symbols'], params['filter_width'],padding=params['filter_width']//2)]
        self.model = torch.nn.Sequential(*layers)
    def forward(self,x):
        embedded=self.embed(x)
        embedded=embedded.permute(0,2,1)
        out=self.model(embedded)
        return out
