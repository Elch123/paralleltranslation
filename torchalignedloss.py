import sentencepiece as spm
import numpy as np
import pickle
import makeendeprocessors
from makeendeprocessors import decode
from makeendeprocessors import decodearray
import torch
from torchmodel import Cnn,ResNet,AttnResNet,AdvancedNet,Transformer
from makebatches import Batch_maker
import json
from tensorboardX import SummaryWriter
from hparams import params
import argparse
import os
import sys
import select
import matplotlib.pyplot as plt
(enprocessor,deprocessor)=makeendeprocessors.load()
#hyperparameters
indel_penalty=-.050
ins_penalty=-.000
del_penalty=-.400
spread_penalty=-.030
smoothing=.5
scoreavg=0
mse_scale=0.5
max_clamp=4.0
lr=1e-2
validlosses=[]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)
traindata=Batch_maker("traindeen.pickle")
validdata=Batch_maker("traindeen.pickle")
parser = argparse.ArgumentParser(description='Save file location')
parser.add_argument('savename',help="the name of the savefile, which will be stored in this dir",default=None)
args=parser.parse_args()
print(args.savename)
writer = SummaryWriter('saves/torchalignedloss_'+args.savename)
def heard_c_then_enter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            text = sys.stdin.readline()
            if 'c' in text:
                return True
    return False
def filepath(name):
    return "saves/"+name+".tar"
def save(model,optimizer,count,filename):
    torch.save({'count':count,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},filename)
def alignment_score(label,tensor):
    if(label==0):
        return -tensor[label]*indel_penalty #don't reward extra blank labels more than the indel indel_penalty, to cancel that out
        #This will prevent the creation of many blamks
    return tensor[label]
def nearest_insert_direction(index,inserts):
    #0 for left, 1 for right
    left=index
    right=index
    while(True):
        if(left>0):
            left-=1
            if(inserts[left]==1):
                return 0
        else:
            return 1
        if(right<len(inserts)-1):
            right+=1
            if(inserts[right]==1):
                return 1
        else:
            return 0
def reassign(label,inserts,deletes):
    #reassign multilabel deletions
    for i in range(1,len(label)):
        j=i
        while(deletes[j]==1):
            j+=1
        if(i-j>=2):
            for k in range(i,j):
                deletes[k]=0#Eliminate longer streches of deletion so they won't affect the single align stage
            label[j-1]=label[i-1]
            label[i]=label[j] #swap opposite label to allow the model to learn to build up n-grams. This might work, or be really bad
    #reassign single label deletions
    """for i in range(len(label)):
        if(deletes[i]==1):
            direction=nearest_insert_direction(i,inserts)
            if(direction==0):
                label[i]=label[i-1]
            if(direction==1):
                label[i]=label[i+1]"""
    return label
def unzeropad(array):
    l=len(array)-1
    while(array[l]==0 and l>1):
        l-=1
    return array[0:l+1]
def align(output,label,reassign_blank=False): # This uses the Needleman Wunsch optimal alignment algrithm to align the outputs of the model with the labels
    label=unzeropad(label)
    l_label=len(label)+1
    l_output=len(output)+1
    tracelabel=label
    array=np.zeros(shape=(l_label,l_output)) # One larger than actual sequence to handle the edge cases of all inserition/all deletion.
    trace=np.zeros(shape=(l_label,l_output),dtype=np.int32) #Record path the algorithm is taking
    for i in range(l_label):
        array[i,0]= i*del_penalty
        trace[i,0]=1
    for i in range(l_output): #Fill out the edges
        array[0,i]= i*ins_penalty
        trace[0,i]=2
    trace[0,0]=0
    for i in range(1,l_label):
        for j in range(1,l_output):
            dellabel=array[i-1,j]+del_penalty
            inslabel=array[i,j-1]+ins_penalty
            usebonus=output[j-1,tracelabel[i-1]]
            #This is somewhat hackish, but the labels and output are offset by one in this array,
            #so I use output[j-1] and label[i-1] to get the correct indicies for the output and label sequence to fill the whole array.
            uselabel=array[i-1,j-1]+usebonus
            if(trace[i-1][j-1]==0):
                uselabel+=spread_penalty
            s=(uselabel,inslabel,dellabel)
            array[i,j]=max(s)
            trace[i,j]=s.index(max(s)) #0 is keep, 1 in insert, 2 is delete
    #Read off the optimal alignment in reverse
    k=l_label-1
    #plt.imshow(array)
    #plt.show()
    m=l_output-1
    path=[]
    labelcopy=np.zeros(shape=(len(output)))
    inserts=np.zeros(shape=(len(output)+1))
    deletes=np.zeros(shape=(len(output)+1))
    while(k>=0 and m>=0):
        #print(k,m)
        path.append((k,m))
        if(trace[k,m]==0): #Keep label
            k-=1
            m-=1
            labelcopy[m]=label[k] #Use elif statements to only trigger one of the cases, not multiple ones before the path is updated.
        elif(trace[k,m]==1): #Insert label
            k-=1
            inserts[m]=1
        elif(trace[k,m]==2): #Delete label
            m-=1
            deletes[m]=1
    if(reassign_blank):
        labelcopy=reassign(labelcopy,inserts,deletes)
    path=path[::-1]
    return (labelcopy,array[l_label-1][l_output-1])
def alignbatch(outputs,labels,reassign_blank=False): # Run alignment algorithm on each tensor in the batch
    upsamplemult=1
    if(params['upsample']):
        upsamplemult=2
    aligned=np.zeros(shape=(labels.shape[0],labels.shape[1]*upsamplemult),dtype=np.int32)
    totalscore=0
    for i in range(len(aligned)):
        a=align(outputs[i],labels[i],reassign_blank)
        totalscore+=a[1]
        aligned[i]=a[0]
    totalscore/=len(aligned)
    totalscore/=len(aligned[0])
    return (aligned,totalscore)
    pass
def maketorchbatch(gen):
    batch=gen.makebatch(params['symbols_in_batch'])
    print(batch.shape)
    return torch.from_numpy(batch).long()
def tonumpy(tensor):
    return tensor.detach().cpu().numpy()
def printvalid(count):
    batch=maketorchbatch(validdata).to(device)
    data=batch[1]
    labels=batch[0]
    output=net(data)
    cpuoutput=tonumpy(softmax(output.permute(0,2,1))) #get the output as a channels last numpy array.
    cpulabels=tonumpy(labels)
    alignedlabels=alignbatch(cpuoutput,cpulabels)
    writer.add_scalar('Valid/Alignment', alignedlabels[1], count)
    alignedlabels=alignedlabels[0]
    alignedlabels=torch.from_numpy(alignedlabels).long().to(device) #reconvert it to a torch tensor
    loss=ce_loss_fn(output,alignedlabels)
    cpuoutput=np.argmax(cpuoutput,axis=2) #argmax it to get the real words
    #print(cpuoutput)
    print(decodearray(enprocessor,cpuoutput))
    print("\n")
    print(decodearray(enprocessor,labels.detach().cpu().numpy()))
    print("validation loss "+str(loss.item())+"\n")
    writer.add_scalar('Valid/Loss', loss.item(), count)
"""genea=np.array([[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,1,0]])
geneb=np.array([3,1,2,2,1,4,1])
print(genea)
print(geneb)
a=align(genea,geneb)
print(a)
print(exiting)"""
count=0
net=Transformer(params)#AttnResNet
net.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,nesterov=True,weight_decay=params['weight_decay'])
mask=torch.ones((params['symbols'],))
mask[0]=0
ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean',weight=mask.to(device))#
l_loss_fn = torch.nn.MSELoss(reduction='mean') #L1Loss
softmax=torch.nn.Softmax(dim=2)
loadpath=filepath(args.savename)
if(os.path.isfile(loadpath)):
    checkpoint=torch.load(loadpath)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.train()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    count=checkpoint['count']
else:
    print("load failure")

for e in range(params['epochs']):
    #This loss function produces the one already done if it is the best, or else the closest improvement

    while(True):
        print(count)
        batch=maketorchbatch(traindata)
        #print(decode(enprocessor,batch[0][0]))
        #print(decode(deprocessor,batch[1][0]))
        s=batch[0].shape
        batch=batch.to(device)
        mask=np.random.random_sample((s[0],s[1]))>=.03
        mask=torch.from_numpy(mask.astype(np.float32)).float().to(device)
        mask=mask.unsqueeze(1)
        data=batch[1]
        labels=batch[0]
        output=net(data)
        #print(output.shape)
        #print(mask.shape)
        output*=mask
        cpuoutput=tonumpy(softmax(output.permute(0,2,1)))#convert the tensor to have the channel dimantion last for the alignment stage
        #Take the softmax to constarin output between 0 and 1
        cpulabels=tonumpy(labels)
        alignedlabels=alignbatch(cpuoutput,cpulabels)
        print(decode(enprocessor,labels[0]))
        print(decode(enprocessor,alignedlabels[0][0]))
        writer.add_scalar('Train/Alignment', alignedlabels[1], count)
        alignedlabels=alignedlabels[0]
        alignedlabels=torch.from_numpy(alignedlabels).long().to(device) #reconvert it to a torch tensor
        #This conditional loss to constrin the number of generaed tokens is fairly complicated, but works. Perhaps find a simpler way?
        labelcount=torch.sum(torch.sign(labels),dim=1) #number of non zero output in label
        outcount=torch.sum(torch.sign(torch.argmax(output,dim=1)),dim=1) #number of non zero output in tensor
        diffcount=outcount-labelcount
        nonzerooutput=torch.sum(output.softmax(dim=-2)[:,1:,:],dim=1)
        outputnonzero=torch.sum(nonzerooutput,dim=1)
        diffcount=diffcount.float()/(labelcount+3).float()
        diffcount=torch.clamp(diffcount,-100.0,max_clamp)#limit how strongly this loss constrains the output token number.
        l_targets=outputnonzero-diffcount
        l_targets=l_targets.detach()
        lloss=l_loss_fn(outputnonzero,l_targets)*mse_scale
        #End conditional loss. Probably sould put in it's own function
        print("reg loss "+str(lloss.item()))
        celoss=ce_loss_fn(output,alignedlabels)
        print("ce loss "+str(celoss.item())+"\n")
        loss=lloss+celoss
        writer.add_scalar('Train/Loss', loss.item(), count)
        if(count%10==0 ):
            printvalid(count)
        if(count%500==0):
            save(net,optimizer,count,loadpath)
        net.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        count+=1
        if(heard_c_then_enter()):
            break
    break
torch.cuda.synchronize()
del net
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.synchronize()
