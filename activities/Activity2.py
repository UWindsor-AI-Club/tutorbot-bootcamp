import re
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab # str key int value
        self.int_to_str = { i:s for s,i in vocab.items()} # int key str value
    
    def encode(self, text): # to int
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        #print(preprocessed)
        a = []
        for item in preprocessed:
            if item in self.str_to_int: # vocab dictionary
                a.append(item)
            else:
                a.append("<|unk|>")
            
        preprocessed = a

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids): # to string
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text



text1 = "What is the opposite word of hot?"
text = text1 +" <|endoftext|> "
print("\n", text + "\n")

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) # include these tokens in the vocab
tokenVocab = {token:integer for integer,token in enumerate(all_tokens)}

print("Vocab: ")
for i, item in enumerate(list(tokenVocab.items())[:15]):
  print(item)

print("\n Using Simple Tokenizer V2: \n")
tokenizer1 = SimpleTokenizerV2(tokenVocab)


encodedText = tokenizer1.encode(text)
print("Ids: ")
print(encodedText)
decodedText = tokenizer1.decode(encodedText)
print("Text: ")
print(decodedText)


print("\nUsing GPT2 Tokenizer: \n")

tokenizer2 = tiktoken.get_encoding("gpt2")
print("Ids:")
encodedText = tokenizer2.encode(text, allowed_special={'<|endoftext|>'})
print(encodedText)
print("Text:")
decodedText = tokenizer2.decode(encodedText)
print(decodedText)