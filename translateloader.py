import sentencepiece as spm

def loadtobpe(modelpath,filepath):
    processor=spm.SentencePieceProcessor()
    processor.Load(modelpath)
    lines=[]
    with open(filepath) as f:
        for line in f:
            data=processor.EncodeAsIds(line)
            lines.append(data)
    return lines
engbpe=loadtobpe("ende/english.model","ende/news-commentary-v11.de-en.en")
print(engbpe[0:10])
