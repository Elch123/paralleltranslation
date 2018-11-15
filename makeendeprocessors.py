import sentencepiece as spm
def makeprocessor(modelpath):
    processor=spm.SentencePieceProcessor()
    processor.Load(modelpath)
    return processor
def load():
    return (makeprocessor("ende/english.model"),makeprocessor("ende/german.model"))
def decode(processor,sentence):
    return processor.DecodeIds(sentence.tolist())
def decodearray(processor,sentences):
    text=[processor.DecodeIds(sentence.tolist()) for sentence in sentences]
    return text
