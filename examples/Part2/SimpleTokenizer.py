import re
from torch.utils.data import Dataset, DataLoader
from importlib.metadata import version

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


    
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) # include these tokens in the vocab
tokenVocab = {token:integer for integer,token in enumerate(all_tokens)}

tokenizer = SimpleTokenizerV2(tokenVocab)


""""

print("str to int:")
for i, item in enumerate(list(tokenVocab.items())):
  print(item)

print("str to int:")
for i, item in enumerate(list(tokenizer.str_to_int.items())):
  print(item)

for i, item in enumerate(list(tokenizer.str_to_int.items())[:15]):
  print(item)



text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(tokenizer.encode(text)) # print ids of words
print(tokenizer.decode(tokenizer.encode(text))) # print words

"""

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

context_size = 4
enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[50:]

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
