import os
import urllib.request
import re
import torch
from torch.utils.data import Dataset, DataLoader
from importlib.metadata import version
import tiktoken

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
#print("Total number of character:", len(raw_text))
#print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(preprocessed[30:80])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
vocab = {token:integer for integer,token in enumerate(all_words)}
# print('Some entries (tokens and their ids) in the voculary')
# for i, item in enumerate(list(vocab.items())[-5:]):
  #  print(item)
# print(f'Vocabulary size {vocab_size}')

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}
print((vocab.items()))
