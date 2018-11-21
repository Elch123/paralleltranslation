import sentencepiece as spm
import numpy as np
import pickle
import makeendeprocessors
from makeendeprocessors import decode
from makeendeprocessors import decodearray
(enprocessor,deprocessor)=makeendeprocessors.load()

def maxlen(langa,langb):
    return max(len(langa),len(langb))
#print(text[0][500])
#print(decode(enprocessor,text[0][500]))
#print(decodearray(enprocessor,text[0][6000:6010]))

"""b=makebatch(500)
print(b)
print(decodearray(deprocessor,b[1]))"""
class Batch_maker():
    def __init__(self,filename):
        with open(filename,'rb') as pairedtext:
            self.text=pickle.load(pairedtext)
    def maxlen(self,langa,langb):
        return max(len(langa),len(langb))
    def makebatch(self,maxsymbols):
        numstrings=len(self.text[0])
        topi=np.random.randint(numstrings)
        strlen=max(maxlen(self.text[0][topi],self.text[1][topi]),4)
        numback=max(maxsymbols//strlen,1)
        numback=numback+min(0,topi-numback)#clip number of elements going back if it is less than zero, to not overrun start of array.
        fronti=topi-numback
        batch=np.zeros(shape=(2,numback,strlen),dtype=np.int32)
        for i in range(numback):
            seta=self.text[0][fronti+i]
            batch[0][i][0:len(seta)]=seta
            setb=self.text[1][fronti+i]
            batch[1][i][0:len(setb)]=setb
        return batch
