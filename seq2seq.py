import sentencepiece as spm
import numpy as np
import pickle
import makeendeprocessors
from makeendeprocessors import decode
from makeendeprocessors import decodearray
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn,Block
from makebatches import makebatch
(enprocessor,deprocessor)=makeendeprocessors.load()
#hyperparameters
symbols_in_batch=500
num_hidden=320
epochs=100
smoothing=.999
model_ctx=mx.cpu()
scoreavg=0
symbols=4000
net_verbose=False
#model defanition
class Permuter(gluon.Block):
    def __init__(self,**kwargs):
        super(Permuter,self).__init__(**kwargs)
        with self.name_scope():
            self.embed=gluon.nn.Embedding(symbols,num_hidden)
            self.net=gluon.nn.Sequential()
            with self.net.name_scope():
                self.net.add(gluon.nn.Conv1D(num_hidden,3,activation='relu',padding=(1,1)))
                self.net.add(gluon.nn.Conv1D(num_hidden,3,activation='relu',padding=(1,1)))
                self.net.add(gluon.nn.Conv1D(num_hidden,3,activation='relu',padding=(1,1)))
                self.net.add(gluon.nn.Conv1D(4000,1,activation='relu'))
    def forward(self,x): #ND requires the channel axis to be second, not last, so I need to transpose axes to make that possible.
        if(net_verbose):
            print("before net shape")
            print(x.shape)
        embedded=self.embed(x)
        if(net_verbose):
            print("embedded shape")
            print(embedded.shape)
        embedded=nd.transpose(embedded,(0,2,1))
        if(net_verbose):
            print("transposed shape")
            print(embedded.shape)
        #x=nd.expand_dims(x,axis=1)
        netout=self.net(embedded)
        if(net_verbose):
            print("after net shape")
            print(netout.shape)
        out=nd.transpose(netout,(0,2,1))
        if(net_verbose):
            print("out shape ")
            print(out.shape)
        return out

net=Permuter()
mx.random.seed(np.random.randint(0,1000000))
#net.collect_params().initialize(mx.init.Normal(sigma=.1),ctx=model_ctx)
net.collect_params().initialize(mx.init.Xavier(),ctx=model_ctx)
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':1}) #5 for MSE
gluonloss=gluon.loss.L2Loss()
smoothed_loss="null"
#scoring functions
def scoreseq(candidate,target,num_symbols):
    #this function returns how many identical symbols appear in these two sequences. Equvilently, the size of the largest subset of them that are a permutation of each other.
    #OR 1 gram clipped recall
    dist=[0]*num_symbols
    for i in target.asnumpy().astype(np.int32):
        if(i>0):
            dist[i]+=1
    score=0
    for i in candidate.asnumpy().astype(np.int32):
        if(dist[i]>0):
            dist[i]-=1
            score+=1
    return score
def countup(seq,num_symbols): # counts how many of each symbol exist in the sequence (A histogram)
    dist=[0]*num_symbols
    for i in seq.asnumpy().astype(np.int32):
        if(i>0): #don't reward padding or unknown symbols
            dist[i]+=1
    return dist
def makescores(candidate,target,num_symbols): # this function provides scores to optimize recall,
    currentscore=scoreseq(candidate,target,num_symbols)
    global scoreavg
    scoreavg=scoreavg*smoothing+(1-smoothing)*currentscore
    scores=nd.zeros(shape=(len(candidate),num_symbols))
    distcandidate=countup(candidate,num_symbols)
    disttarget=countup(target,num_symbols)
    for i in range(num_symbols):
        if(disttarget[i]>distcandidate[i]):
            scores[:,i]+=1 # add one to all rows where it would be beneficial to have another of that symbol
    for i in range(len(candidate)):
        npcandidate=candidate.asnumpy().astype(np.int32)
        #print(npcandidate)
        if(disttarget[npcandidate[i]]>=distcandidate[npcandidate[i]]):
            #print("less")
            scores[i,:]=0 #dont change where changing it would hurt scores
            scores[i,npcandidate[i]]=1
    return scores
def make_batch_scores(output,labels,num_symbols):
    preds=output.argmax(axis=2)
    if(net_verbose):
        print(preds)
    scoreholder=nd.zeros_like(output)
    global scoreavg
    print("average score "+str(scoreavg))
    for i in range(len(preds)):
        scoreholder[i]=makescores(preds[i],labels[i],num_symbols)
    return scoreholder
#
for e in range(epochs):
    #This loss function produces the one already done if it is the best, or else the closest improvement
    count=0
    while(True):
        batch=nd.array(makebatch(symbols_in_batch)).as_in_context(model_ctx)
        data=batch[1]
        labels=batch[0]
        #data=nd.expand_dims(data,axis=-1)
        #labels=nd.expand_dims(labels,axis=-1)
        with autograd.record():
            output=net(data)
        scores=make_batch_scores(output,labels,symbols)
        avgscore=nd.mean(scores,axis=(1,2),keepdims=True)
        #print(avgscore)
        scores-=avgscore
        scores/=nd.norm(scores)
        scores*=100 # just scale it up a bit so losses are in a reasonable range
        with autograd.record():
            loss=gluonloss(output,scores)
            count+=1
            if(count%10==0 ):
                print(count)
                print("data")
                print(data)
                print("output")
                print(output.argmax(axis=2).asnumpy().astype(np.int32))
                print(decodearray(enprocessor,output.argmax(axis=2).asnumpy().astype(np.int32)))
        loss.backward()
        trainer.step(data.shape[0])
        if(smoothed_loss=="null"):
            smoothed_loss=nd.mean(loss).asscalar()
        else:
            smoothed_loss=smoothed_loss*smoothing+(1-smoothing)*nd.mean(loss).asscalar()
        #print(nd.mean(loss).asscalar())
        print(smoothed_loss)
