import sentencepiece as spm
import numpy as np
import pickle
import makeendeprocessors
from makeendeprocessors import decode
from makeendeprocessors import decodearray
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn,Block
from makebatches import Batch_maker
from model import Cnn,RNNmodel
import math
(enprocessor,deprocessor)=makeendeprocessors.load()
#export MXNET_GPU_MEM_POOL_RESERVE=10
#for debugging export MXNET_ENGINE_TYPE=NaiveEngine
#hyperparameters
params={
'symbols_in_batch':1000,
'num_hidden':400,
'epochs':100,
'symbols':4000,
'net_verbose':False,
}
smoothing=.5
blank_weight=.00
data_ctx=mx.cpu()
model_ctx=mx.gpu()
scoreavg=0
traindata=Batch_maker("traindeen.pickle")
validdata=Batch_maker("traindeen.pickle")
def alignment_score(label,tensor):
    return tensor[label]
def align(output,label): # This uses the Needleman Wunsch optimal alignment algrithm to align the outputs of the model with the labels
    indel_penalty=-.05
    l_label=len(label)+1
    l_output=len(output)+1
    output=output.asnumpy()
    labelcopy=nd.zeros(shape=(len(output)))
    tracelabel=label.asnumpy().astype(np.int32)
    array=np.zeros(shape=(l_label,l_output)) # One larger than actual sequence to have the edge cases of all inserition/all deletion.
    trace=np.zeros(shape=(l_label,l_output),dtype=np.int32) #Find where
    for i in range(l_label):
        array[i,0]= i*indel_penalty
        trace[i,0]=1
    for i in range(l_output): #Fill out the edges
        array[0,i]= i*indel_penalty
        trace[0,i]=2
    trace[0,0]=0
    for i in range(1,l_label):
        for j in range(1,l_output):
            inslabel=array[i-1,j]+indel_penalty
            dellabel=array[i,j-1]+indel_penalty
            usebonus=output[j-1,tracelabel[i-1]]
            #This is somewhat hackish, but the labels and output are offset by one in this array,
            #print(uselabel)
            #so I use output[j-1] and label[i-1] to get the correct indicies for the output and label sequence to fill the whole array.
            uselabel=array[i-1,j-1]+usebonus
            s=(uselabel,inslabel,dellabel)
            array[i,j]=max(s)
            trace[i,j]=s.index(max(s))
    #Read off the optimal alignment in reverse
    k=l_label-1
    m=l_output-1
    path=[]
    #print(array)
    #print(trace)
    #print(trace.shape)
    while(k>=0 and m>=0):
        #print(k,m)
        path.append((k,m))
        if(trace[k,m]==0): #Keep label
            k-=1
            m-=1
            labelcopy[m]=label[k]
        #Use elif statements to only trigger one of the cases, not multiple ones before the path is updated.
        elif(trace[k,m]==1): #Insert label
            k-=1
        elif(trace[k,m]==2): #Delete label
            m-=1
    #print(label.asnumpy().astype(np.int32))
    #print(labelcopy.asnumpy().astype(np.int32))
    path=path[::-1]
    #print(path)
    return labelcopy
def alignbatch(outputs,labels): # Run alignment algorithm on each tensor in the batch
    aligned=nd.zeros_like(labels)
    for i in range(len(aligned)):
        aligned[i]=align(outputs[i],labels[i])
    return aligned
    pass
"""genea=nd.array([[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,1,0]])
geneb=nd.array([3,1,2,2,1,4,1])
print(genea)
print(geneb)
a=align(genea,geneb)
print(a)
print(exiting)"""
#model defanition
#net=RNNmodel(params)
net=Cnn(params)
mx.random.seed(np.random.randint(0,1000000))
net.collect_params().initialize(mx.init.Xavier(),ctx=model_ctx)
net.hybridize()
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':.01,'momentum': .9}) #5 for MSE
loss_fn=gluon.loss.SoftmaxCrossEntropyLoss()
gluonloss=gluon.loss.L2Loss()
smoothed_loss="null"
def printvalid():
    batch=nd.array(validdata.makebatch(params['symbols_in_batch'])).as_in_context(model_ctx)
    data=batch[1]
    labels=batch[0]
    output=net(data)
    print(output.argmax(axis=2).asnumpy().astype(np.int32))
    print(decodearray(enprocessor,output.argmax(axis=2).asnumpy().astype(np.int32)))
    print("\n")
    print(decodearray(enprocessor,labels.asnumpy().astype(np.int32)))
for e in range(params['epochs']):
    #This loss function produces the one already done if it is the best, or else the closest improvement
    count=0
    while(True):
        #print(dir(trainer))
        batch=traindata.makebatch(params['symbols_in_batch'])
        print(batch.shape)
        batch=nd.array(batch).as_in_context(model_ctx)
        data=batch[1]
        labels=batch[0]
        with autograd.record():
            output=net(data)
        cpuoutput=output.as_in_context(data_ctx)
        cpulabels=labels.as_in_context(data_ctx)
        alignedlabels=alignbatch(cpuoutput,cpulabels).as_in_context(model_ctx)

        with autograd.record():
            #print("#######################")
            #print(alignedlabels.asnumpy().astype(np.int32))
            mask=nd.sign(alignedlabels)#*(1-blank_weight)+blank_weight
            mask=mask.reshape((mask.shape[0],mask.shape[1],1))
            #print(mask.shape)
            #print(output.shape)
            loss=loss_fn(output,alignedlabels,mask)
            count+=1
            if(count%10==0 ):
                printvalid()
        loss.backward()
        trainer.step(data.shape[0])
        if(smoothed_loss=="null"):
            smoothed_loss=nd.mean(loss).asscalar()
        else:
            smoothed_loss=smoothed_loss*smoothing+(1-smoothing)*nd.mean(loss).asscalar()
        #print(nd.mean(loss).asscalar())
        print(smoothed_loss)
