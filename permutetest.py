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
net=Permuter()
net.collect_params().initialize(mx.init.Normal(sigma=.1),ctx=model_ctx)
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':.05})
loss_fn=gluon.loss.SoftmaxCrossEntropyLoss()
for e in range(epochs):
    smoothed_loss=5
    for i,(data,label) in enumerate(train_data):
        data=data.as_in_context(model_ctx)
        label=label.as_in_context(model_ctx)
        with autograd.record():
            output=net(data)
            loss=loss_fn(output,label)
            if(i%100==0):
                print("data")
                print(data)
                print("output")
                print(output.asnumpy().argmax(axis=2))
        loss.backward()
        trainer.step(data.shape[0])
        smoothed_loss=smoothed_loss*smoothing+(1-smoothing)*nd.mean(loss).asscalar()
        #print(smoothed_loss)
