import sentencepiece as spm
import numpy as np
import pickle
import makeendeprocessors
#load the files and apply BPE encoding, this could be swapped for tokenization

def loadtobpe(processor,filepath,start,end):
    lines=[]
    histogram=[0]*4000
    with open(filepath,newline="\n") as f:
        f=f.readlines()
        maxlen=len(f)
        for (count,line) in enumerate(f):
            if(count/maxlen>start and count/maxlen<end ): #
                data=processor.EncodeAsIds(line)
                for d in data:
                    #print(d)
                    histogram[d]+=1
                lines.append(np.array(data))
                if(count%1000==0):
                    print(count)
    return (lines,histogram)
(enprocessor,deprocessor)=makeendeprocessors.load()
def makeslice(start,end,savefile):
    enbpe=loadtobpe(enprocessor,"ende/news-commentary-v11.de-en.en",start,end)
    debpe=loadtobpe(deprocessor,"ende/news-commentary-v11.de-en.de",start,end)
    enhist=enbpe[1]
    dehist=debpe[1]
    print(len(enbpe[0]))
    print(len(debpe[0]))
    def sortbymaxlen(sentencepair):
        return max(len(sentencepair[0]),len(sentencepair[1]))
    #sort the sentece arrays by length
    enbpe,debpe=zip(*sorted(zip(enbpe[0],debpe[0]),key=sortbymaxlen))
    text=(enbpe,debpe,enhist,dehist)
    with open(savefile,'wb') as pairedtext:
        pickle.dump(text,pairedtext)
makeslice(0,.9,"traindeen.pickle")
makeslice(.9,1,"validationdeen.pickle")
#print([enprocessor.DecodeIds(sentence) for sentence in enbpe[0:10]])
#print([deprocessor.DecodeIds(sentence) for sentence in debpe[0:10]])