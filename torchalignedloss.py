import sentencepiece as spm
import numpy as np
import pickle
import makeendeprocessors
from makeendeprocessors import decode
from makeendeprocessors import decodearray
import torch
from torchmodel import Cnn
from makebatches import Batch_maker
(enprocessor,deprocessor)=makeendeprocessors.load()
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
scoreavg=0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
traindata=Batch_maker("traindeen.pickle")
validdata=Batch_maker("traindeen.pickle")
def alignment_score(label,tensor):
    return tensor[label]
def align(output,label): # This uses the Needleman Wunsch optimal alignment algrithm to align the outputs of the model with the labels
    indel_penalty=-.05
    l_label=len(label)+1
    l_output=len(output)+1
    labelcopy=np.zeros(shape=(len(output)))
    tracelabel=label
    array=np.zeros(shape=(l_label,l_output)) # One larger than actual sequence to handle the edge cases of all inserition/all deletion.
    trace=np.zeros(shape=(l_label,l_output),dtype=np.int32) #Record path the algorithm is taking
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
            trace[i,j]=s.index(max(s)) #0 is keep, 1 in insert, 2 is delete
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
    aligned=np.zeros_like(labels)
    for i in range(len(aligned)):
        aligned[i]=align(outputs[i],labels[i])
    return aligned
    pass
"""genea=np.array([[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,1,0]])
geneb=np.array([3,1,2,2,1,4,1])
print(genea)
print(geneb)
a=align(genea,geneb)
print(a)
print(exiting)"""
#model defanition
#net=RNNmodel(params)
net=Cnn(params)
net.to(device)
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1, momentum=0.9)
smoothed_loss="null"

def maketorchbatch(gen):
    batch=gen.makebatch(params['symbols_in_batch'])
    #print(batch.shape)
    return torch.from_numpy(batch).long()
def tonumpy(tensor):
    return tensor.detach().cpu().numpy()
def printvalid():
    batch=maketorchbatch(validdata).to(device)#.as_in_context(model_ctx)
    data=batch[1]
    labels=batch[0]
    output=net(data).permute(0,2,1).detach().cpu().numpy() #get the output as a channels last numpy array.
    output=np.argmax(output,axis=2) #argmax it to get the real words
    print(output)
    print(decodearray(enprocessor,output))
    print("\n")
    print(decodearray(enprocessor,labels.detach().cpu().numpy()))
for e in range(params['epochs']):
    #This loss function produces the one already done if it is the best, or else the closest improvement
    count=0
    while(True):
        #print(dir(trainer))
        batch=maketorchbatch(traindata).to(device)
        data=batch[1]
        labels=batch[0]
        output=net(data)
        cpuoutput=output.permute(0,2,1).detach().cpu().numpy()#convert the tensor to have the channel dimantion last for the alignment stage
        cpulabels=labels.detach().cpu().numpy()
        alignedlabels=alignbatch(cpuoutput,cpulabels)
        alignedlabels=torch.from_numpy(alignedlabels).long().to(device) #reconvert it to a torch tensor
        loss=loss_fn(output,alignedlabels)
        #loss=loss_fn(output,labels)
        #print(loss)
        if(count%10==0 ):
            printvalid()
        net.zero_grad()
        loss.backward()
        optimizer.step()
        count+=1
        if(smoothed_loss=="null"):
            smoothed_loss=loss.item()
        else:
            smoothed_loss=smoothed_loss*smoothing+(1-smoothing)*loss.item()
        #print(nd.mean(loss).asscalar())
        print(smoothed_loss)
