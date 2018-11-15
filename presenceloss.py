from __future__ import print_function
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn,Block
import random
import numpy as np
import sys
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
    scores=np.zeros(shape=(len(candidate),num_symbols))
    print(currentscore)
    for i in range(len(candidate)):
        symbol=candidate[i].copy()
        for j in range(num_symbols):
            candidate[i]=j
            s=scoreseq(candidate,target,num_symbols)
            scores[i,j]=s
        candidate[i]=symbol
    #print(copy==candidate)
    return scores
def findimprovements(scorematrix):
    l=len(scorematrix)
    bettersymbols=[[] for i in range(l)]
    currentscore=scorematrix[0,0]
    #copy=candidate.copy()
    print(currentscore)
    for i in range(l):
        for j in range(len(scorematrix[0])):
            if(scorematrix[i,j]>currentscore):
                bettersymbols[i].append(j)
    #print(copy==candidate)
    return bettersymbols
def maketarget(output,candidate,target,num_symbols):
    scores=makescores(candidate,target,num_symbols)
    improvements=findimprovements(scores)
    mask=nd.ones(shape=(len(candidate)))
    for i in range(len(improvements)):
        if(len(improvements[i])==0):
            improvements[i].append(int(candidate[i].asscalar()))
            mask[i]=0
    target=[0]*len(candidate)
    for i in range(len(improvements)):
        strongestoutput=-1
        for improvement in improvements[i]:
            activation=output[i,improvement]
            if(strongestoutput<activation):
                strongestoutput=activation
                target[i]=improvement
    target=nd.array(target)
    return (target,mask)
#print(scoreseq(nd.array([0,1,1,0,2]),nd.array([0,1,2,3,1,1,1,1,1]),4))
#print(makescores(nd.array([0,1,1,0,2]),nd.array([0,1,2,3,1]),4))
#sys.exit()
net=Permuter()
mx.random.seed(1000)
net.collect_params().initialize(mx.init.Normal(sigma=.2),ctx=model_ctx)
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':.1})
loss_fn=gluon.loss.SoftmaxCrossEntropyLoss()
for e in range(epochs):
    #This loss function produces the one already done if it is the best, or else the closest improvement
    smoothed_loss=5
    for i,(data,labels) in enumerate(train_data):
        data=data.as_in_context(model_ctx)
        labels=labels.as_in_context(model_ctx)
        with autograd.record():
            output=net(data)
        target=nd.zeros_like(labels)
        mask=nd.zeros_like(labels)
        preds=output.argmax(axis=2)
        #print(preds)
        for p in range(len(preds)):
            print("#######################################################")
            print(labels[p])
            print(preds[p])

            (target[p],mask[p])=maketarget(output[p],preds[p],labels[p],symbols)
            print(target[p])
            pass
        with autograd.record():
            #print(target.shape)
            #print(mask.shape)
            mask=mask.astype(np.float32)
            #print(output.shape)
            #loss=loss_fn(output,target,nd.expand_dims(mask,axis=-1))
            loss=loss_fn(output,target)
            if(i%100==0):
                print(i)
                print("data")
                print(data)
                print("output")
                print(output.argmax(axis=2))
        loss.backward()
        trainer.step(data.shape[0])
        smoothed_loss=smoothed_loss*smoothing+(1-smoothing)*nd.mean(loss).asscalar()
        #print(nd.mean(loss).asscalar())
        #print(smoothed_loss)
