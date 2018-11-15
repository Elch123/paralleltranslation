import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn,Block,rnn



class Cnn_block(gluon.HybridBlock):
    def __init__(self,params,**kwargs):
        super().__init__(**kwargs)
        self.paramsdict=params
        with self.name_scope():
            self.net=gluon.nn.HybridSequential()
            with self.net.name_scope():
                self.net.add(gluon.nn.Conv1D(params['num_hidden'],3,activation='relu',padding=(1,1)))
                self.net.add(gluon.nn.Conv1D(params['num_hidden'],3,activation='relu',padding=(1,1)))
                self.net.add(gluon.nn.Conv1D(params['num_hidden'],3,activation='relu',padding=(1,1)))
                self.net.add(gluon.nn.Conv1D(params['symbols'],1,activation='relu'))
                print("cnn called")
    def hybrid_forward(self,F,x): #ND requires the channel axis to be second, not last, so I need to transpose axes to make that possible.
        net_verbose=self.paramsdict['net_verbose']
        if(net_verbose):
            print("before net shape")
            print(x.shape)
        embedded=x
        embedded=F.transpose(embedded,(0,2,1))
        if(net_verbose):
            print("transposed shape")
            print(embedded.shape)
        #x=nd.expand_dims(x,axis=1)
        #print(embedded)
        netout=self.net(embedded)
        #print(netout)
        if(net_verbose):
            print("after net shape")
            print(netout.shape)
        out=F.transpose(netout,(0,2,1))
        if(net_verbose):
            print("out shape ")
            print(out.shape)
        return out

class Final_layer(gluon.HybridBlock):
    def __init__(self,params,**kwargs):
        super().__init__(**kwargs)
        self.paramsdict=params
        with self.name_scope():
            self.net=gluon.nn.Sequential()
            with self.net.name_scope():
                self.net.add(gluon.nn.Conv1D(params['symbols'],1,activation='relu'))
                print("cnn called")
    def forward(self,x): #ND requires the channel axis to be second, not last, so I need to transpose axes to make that possible.
        net_verbose=self.paramsdict['net_verbose']
        if(net_verbose):
            print("before net shape")
            print(x.shape)
        embedded=x
        embedded=nd.transpose(embedded,(0,2,1))
        if(net_verbose):
            print("transposed shape")
            print(embedded.shape)
        #x=nd.expand_dims(x,axis=1)
        #print(embedded)
        netout=self.net(embedded)
        #print(netout)
        if(net_verbose):
            print("after net shape")
            print(netout.shape)
        out=nd.transpose(netout,(0,2,1))
        if(net_verbose):
            print("out shape ")
            print(out.shape)
        return out

class Cnn(gluon.HybridBlock):
    def __init__(self,params,**kwargs):
        super().__init__(**kwargs)
        self.paramsdict=params
        with self.name_scope():
            self.embed=gluon.nn.Embedding(params['symbols'],params['num_hidden'])
            self.net=Cnn_block(params)
    def hybrid_forward(self,F,x): #ND requires the channel axis to be second, not last, so I need to transpose axes to make that possible.
        net_verbose=self.paramsdict['net_verbose']
        if(net_verbose):
            print("before net shape")
            print(x.shape)
        embedded=self.embed(x)
        out=self.net(embedded)
        if(net_verbose):
            print("out shape ")
            print(out.shape)
        return out


class RNNmodel(gluon.Block):
    def __init__(self,params,**kwargs):
        super().__init__(**kwargs)
        self.paramsdict=params
        with self.name_scope():
            self.embed=gluon.nn.Embedding(params['symbols'],params['num_hidden'])
            self.net=gluon.nn.Sequential()
            with self.net.name_scope():
                self.net.add(rnn.LSTM(params['num_hidden'], num_layers=2,layout='NTC')) #
                self.net.add(Final_layer(params))
                #self.net.add(gluon.nn.Conv1D(4000,1,activation='relu'))
                #
            """self.convpart=gluon.nn.Sequential()
            with self.convpart.name_scope():
                self.net.add(gluon.nn.Conv1D(4000,1,activation='relu'))"""
    def forward(self,x): #ND requires the channel axis to be second, not last, so I need to transpose axes to make that possible.
        net_verbose=self.paramsdict['net_verbose']
        if(net_verbose):
            print("before net shape")
            print(x.shape)
        out=self.net(self.embed(x))
        #out=self.embed(x)
        if(net_verbose):
            print("after net shape")
            print(out.shape)
        return out
