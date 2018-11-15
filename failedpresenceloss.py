from __future__ import print_function
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn,Block
import numpy as np
num_examples=10000
seqlen=10
symbols=10
batch_size=16
num_hidden=32
epochs=100
smoothing=.95
model_ctx=mx.cpu()
data=np.random.randint(symbols,size=(num_examples,seqlen),dtype=np.int32)
labels=np.zeros(shape=(num_examples,seqlen),dtype=np.int32)
for i in range(len(data)):
    labels[i]=np.random.permutation(data[i])
train_data=gluon.data.DataLoader(gluon.data.ArrayDataset(data,labels),batch_size=batch_size,shuffle=True)
class Permuter(gluon.Block):
    def __init__(self,**kwargs):
        super(Permuter,self).__init__(**kwargs)
        with self.name_scope():
            self.net=gluon.nn.Sequential()
            with self.net.name_scope():
                self.net.add(gluon.nn.Embedding(symbols,num_hidden))
                self.net.add(gluon.nn.Dense(num_hidden,activation='relu'))
                self.net.add(gluon.nn.Dense(num_hidden,activation='relu'))
                #self.net.add(gluon.nn.Dense(num_hidden,activation='relu'))
                #self.net.add(gluon.nn.Dense(num_hidden,activation='relu'))
                self.net.add(gluon.nn.Dense(symbols*seqlen))
    def forward(self,x):
        netout=self.net(x)
        out=nd.reshape(netout,(batch_size,seqlen,symbols))
        return out
def scoreseq(candidate,target,num_symbols):
    #this function returns how many identical symbols appear in these two sequences. Equvilently, the size of the subset of them that are a permutation of each other.
    dist=[0]*num_symbols
    for i in target.asnumpy():
        dist[i]+=1
    score=0
    for i in candidate.asnumpy().astype(np.int32):
        if(dist[i]>0):
            dist[i]-=1
            score+=1
    return score

net=Permuter()
net.collect_params().initialize(mx.init.Normal(sigma=.1),ctx=model_ctx)
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':.05})
loss_fn=gluon.loss.KLDivLoss(from_logits=False)
for e in range(epochs):
    #This loss function produces the one already done if it is the best, or else an even weighting over all improvements
    smoothed_loss=5
    for i,(data,labels) in enumerate(train_data):
        print(i)
        data=data.as_in_context(model_ctx)
        labels=labels.as_in_context(model_ctx)
        with autograd.record():
            output=net(data)
        preds=output.argmax(axis=2)
        target=nd.zeros(shape=(batch_size,seqlen,symbols))
        for p,pred in enumerate(preds):
            #print(p)
            label=labels[p]
            scorenow=scoreseq(pred,label,symbols)
            for j in range(seqlen):
                for k in range(symbols):
                    store=pred[j]
                    pred[j]=k
                    s=scoreseq(pred,label,symbols)
                    if(s>scorenow):
                        target[p,j,k]=1
                    pred[j]=store
        for (index,example) in enumerate(target):
            numbetter=nd.sum(example,axis=-1)
            for (j,better) in enumerate(numbetter):
                if(better==0):
                    target[index,j,preds[index,j]]=1
                    numbetter[j]=1
            target[index]/=numbetter
        with autograd.record():
            loss=loss_fn(output,target)
            if(i%100==0):
                print("data")
                print(data)
                print("output")
                print(output.argmax(axis=2))
        loss.backward()
        trainer.step(data.shape[0])
        smoothed_loss=smoothed_loss*smoothing+(1-smoothing)*nd.mean(loss).asscalar()
        print(nd.mean(loss).asscalar())
        #print(smoothed_loss)
