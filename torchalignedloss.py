import sentencepiece as spm
import numpy as np
import pickle
import makeendeprocessors
from makeendeprocessors import decode
from makeendeprocessors import decodearray
import torch
from torchmodel import Cnn,ResNet,AttnResNet
from makebatches import Batch_maker
import atexit
import json
(enprocessor,deprocessor)=makeendeprocessors.load()
#hyperparameters
params={
'symbols_in_batch':1000,
'num_hidden':600,
'attnsize':200,
'epochs':100,
'max_seqlen':1000,
'embed_size':200,
'symbols':4000,
'net_verbose':False,
'batchnorm':False, #True causes training failure, gradient clipping might fix this
'upsample':False
}
indel_penalty=-.01
smoothing=.5
blank_weight=0
scoreavg=0
mse_scale=1e-3
validlosses=[]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
traindata=Batch_maker("traindeen.pickle")
validdata=Batch_maker("traindeen.pickle")
def on_exit():
    with open("validhistory",'w') as f:
        json.dump(validlosses,f)
atexit.register(on_exit)
def alignment_score(label,tensor):
    if(label==0):
        return -tensor[label]*indel_penalty #don't reward extra blank labels more than the indel indel_penalty, to cancel that out
        #This will prevent the creation of many blamks
    return tensor[label]
def align(output,label): # This uses the Needleman Wunsch optimal alignment algrithm to align the outputs of the model with the labels
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
    while(k>=0 and m>=0):
        #print(k,m)
        path.append((k,m))
        if(trace[k,m]==0): #Keep label
            k-=1
            m-=1
            labelcopy[m]=label[k] #Use elif statements to only trigger one of the cases, not multiple ones before the path is updated.
        elif(trace[k,m]==1): #Insert label
            k-=1
        elif(trace[k,m]==2): #Delete label
            m-=1
    path=path[::-1]
    return labelcopy
def alignbatch(outputs,labels): # Run alignment algorithm on each tensor in the batch
    upsamplemult=1
    if(params['upsample']):
        upsamplemult=2
    aligned=np.zeros(shape=(labels.shape[0],labels.shape[1]*upsamplemult))
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
net=AttnResNet(params)
net.to(device)
mask=torch.ones((4000,))
mask[0]=blank_weight
ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean',weight=mask.to(device))#
mse_loss_fn = torch.nn.MSELoss(reduction='mean')
l_loss_fn = torch.nn.MSELoss(reduction='mean') #L1Loss
softmax=torch.nn.Softmax(dim=2)
optimizer = torch.optim.SGD(net.parameters(), lr=3e-2, momentum=0.9,nesterov=True,weight_decay=1e-5)
smoothed_loss="null"
def maketorchbatch(gen):
    batch=gen.makebatch(params['symbols_in_batch'])
    return torch.from_numpy(batch).long()
def tonumpy(tensor):
    return tensor.detach().cpu().numpy()
def printvalid():
    batch=maketorchbatch(validdata).to(device)
    data=batch[1]
    labels=batch[0]
    output=net(data)
    cpuoutput=tonumpy(output.permute(0,2,1)) #get the output as a channels last numpy array.
    cpulabels=tonumpy(labels)
    alignedlabels=alignbatch(cpuoutput,cpulabels)
    alignedlabels=torch.from_numpy(alignedlabels).long().to(device) #reconvert it to a torch tensor
    loss=ce_loss_fn(output,alignedlabels)
    cpuoutput=np.argmax(cpuoutput,axis=2) #argmax it to get the real words
    #print(cpuoutput)
    print(decodearray(enprocessor,cpuoutput))
    print("\n")
    print(decodearray(enprocessor,labels.detach().cpu().numpy()))
    print("validation loss "+str(loss.item())+"\n")
    validlosses.append(loss.item())
for e in range(params['epochs']):
    #This loss function produces the one already done if it is the best, or else the closest improvement
    count=0
    while(True):
        batch=maketorchbatch(traindata).to(device)
        data=batch[1]
        print(data.shape)
        labels=batch[0]
        output=net(data)

        cpuoutput=tonumpy(softmax(output.permute(0,2,1)))#convert the tensor to have the channel dimantion last for the alignment stage

        #Take the softmax to constarin output between 0 and 1
        cpulabels=tonumpy(labels)
        alignedlabels=alignbatch(cpuoutput,cpulabels)
        alignedlabels=torch.from_numpy(alignedlabels).long().to(device) #reconvert it to a torch tensor
        labelcount=torch.sum(torch.sign(labels),dim=1) #number of non zero output in label
        outcount=torch.sum(torch.sign(torch.argmax(output,dim=1)),dim=1) #number of non zero output in tensor
        diffcount=outcount-labelcount
        #print(labelcount)
        cpulabelcount=tonumpy(labelcount)
        batchsize=len(labelcount)
        #print(outcount)
        outmask=torch.sign(torch.argmax(output,dim=1))
        nonzerooutput=torch.sum(output.softmax(dim=-2)[:,1:,:],dim=1)*outmask.float()
        outputnonzero=torch.sum(nonzerooutput,dim=1)
        diffcount=diffcount.reshape(batchsize).float()/(labelcount+2).float()
        #print(direction)
        l_targets=outputnonzero-diffcount
        l_targets=l_targets.detach()
        #lloss=l_loss_fn(outputnonzero,labelcount.float()*5-4*outcount.float())*.01
        lloss=l_loss_fn(outputnonzero,l_targets)*.01
        print("reg loss "+str(lloss.item()))
        celoss=ce_loss_fn(output,alignedlabels)
        print("ce loss "+str(celoss.item())+"\n")
        loss=lloss+celoss
        if(count%10==0 ):
            printvalid()
        net.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(net.parameters(), 1)
        optimizer.step()
        count+=1
        if(smoothed_loss=="null"):
            smoothed_loss=loss.item()
        else:
            smoothed_loss=smoothed_loss*smoothing+(1-smoothing)*loss.item()
        #print(nd.mean(loss).asscalar())
        #print(smoothed_loss)
