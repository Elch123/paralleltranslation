from __future__ import print_function
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn,Block
import random
import numpy as np
import sys
num_examples=1000000
seqlen=10
symbols=10
batch_size=16
num_hidden=256
epochs=1000
smoothing=.99
scoreavg=0
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
                self.net.add(gluon.nn.Dense(num_hidden,activation='relu'))
                self.net.add(gluon.nn.Dense(num_hidden,activation='relu'))
                self.net.add(gluon.nn.Dense(symbols*seqlen))
    def forward(self,x):
        netout=self.net(x)
        out=nd.reshape(netout,(batch_size,seqlen,symbols))
        return out
def scoreseq(candidate,target,num_symbols):
    #this function returns how many identical symbols appear in these two sequences. Equvilently, the size of the largest subset of them that are a permutation of each other.
    dist=[0]*num_symbols
    for i in target.asnumpy().astype(np.int32):
        dist[i]+=1
    score=0
    for i in candidate.asnumpy().astype(np.int32):
        if(dist[i]>0):
            dist[i]-=1
            score+=1
    return score
def makescores(candidate,target,num_symbols):
    currentscore=scoreseq(candidate,target,num_symbols)
    global scoreavg
    scoreavg=scoreavg*smoothing+(1-smoothing)*currentscore
    scores=nd.zeros(shape=(len(candidate),num_symbols))
    for i in range(len(candidate)):
        symbol=candidate[i].copy()
        for j in range(num_symbols):
            candidate[i]=j
            s=scoreseq(candidate,target,num_symbols)
            scores[i,j]=s
        candidate[i]=symbol

    #print(copy==candidate)
    return scores
def make_batch_scores(output,labels,num_symbols):
    preds=output.argmax(axis=2)
    scoreholder=nd.zeros_like(output)
    global scoreavg
    print("average score "+str(scoreavg))
    for i in range(len(preds)):
        scoreholder[i]=makescores(preds[i],labels[i],num_symbols)
    return scoreholder
#print(scoreseq(nd.array([0,1,1,0,2]),nd.array([0,1,2,3,1,1,1,1,1]),4))
#print(makescores(nd.array([0,1,1,0,2]),nd.array([0,1,2,3,1]),4))
#sys.exit()
net=Permuter()
mx.random.seed(1000)
#net.collect_params().initialize(mx.init.Normal(sigma=.1),ctx=model_ctx)
net.collect_params().initialize(mx.init.Xavier(),ctx=model_ctx)
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':5}) #5 for MSE
gluonloss=gluon.loss.L2Loss()
smoothed_loss="null"
for e in range(epochs):
    #This loss function produces the one already done if it is the best, or else the closest improvement

    for i,(data,labels) in enumerate(train_data):
        data=data.as_in_context(model_ctx)
        labels=labels.as_in_context(model_ctx)
        with autograd.record():
            output=net(data)
        scores=make_batch_scores(output,labels,symbols)
        avgscore=nd.mean(scores,axis=(1,2),keepdims=True)
        #print(avgscore)
        scores-=avgscore
        scores/=nd.norm(scores)
        with autograd.record():
            loss=gluonloss(output,scores)
            if(i%100==0):
                print(i)
                print("data")
                print(data)
                print("output")
                print(output.argmax(axis=2))
        loss.backward()
        trainer.step(data.shape[0])
        if(smoothed_loss=="null"):
            smoothed_loss=nd.mean(loss).asscalar()
        else:
            smoothed_loss=smoothed_loss*smoothing+(1-smoothing)*nd.mean(loss).asscalar()
        #print(nd.mean(loss).asscalar())
        print(smoothed_loss)
# use sentencepiece to tokenize WMT datasets
