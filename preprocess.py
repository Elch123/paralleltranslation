import sentencepiece as spm
from hparams import params
import subprocess
print("change symbols argument in hparams to change vocabulay size")
spm.SentencePieceTrainer.Train('--input=ende/text.de --model_prefix=ende/german --vocab_size='+str(params['symbols']))
spm.SentencePieceTrainer.Train('--input=ende/text.en --model_prefix=ende/english --vocab_size='+str(params['symbols']))
subprocess.call("python3","translateconverter.py", capture_output=False)
