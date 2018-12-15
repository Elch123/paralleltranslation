import sentencepiece as spm
from hparams import params
import subprocess
print("change symbols argument in hparams to change vocabulay size")
spm.SentencePieceTrainer.Train('--input=ende/news-commentary-v11.de-en.de --model_prefix=ende/german --vocab_size='+str(params['symbols']))
spm.SentencePieceTrainer.Train('--input=ende/news-commentary-v11.de-en.en --model_prefix=ende/english --vocab_size='+str(params['symbols']))
subprocess.call("python3","translateconverter.py")
