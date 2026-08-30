import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# read it in to inspect it
with open('shakesphere.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))

## converting the chars into integers  in stoi 
##  converting the int to chars again 
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}


## doing the conversion for large texts and back
encode = lambda input : [stoi[ch] for ch in input]
decode = lambda input : ''.join([itos[i] for i in input])


#creating the large tensor with the whole data
data = torch.tensor(encode(text))


## doing the train and val split
data_length = len(data)
n = int( 0.9*data_length)

train_data= data[:n]
val_data= data[n:]
vocab_size = len(chars)
batch_size = 4
block_size = 8
max_steps = 50000
step_interval = 500
eval_iters = 200 
n_embd = 32 
num_heads = 4 

lr = 1e-3

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        ## defing the 3 layer q, k , v 
        self.query = nn.Linear(n_embd,head_size,bias=False) #(n_embd,head_size)
        self.key = nn.Linear(n_embd,head_size,bias=False)   #(n_embd,head_size)
        self.value = nn.Linear(n_embd,head_size,bias=False) #(n_embd,head_size)

        ## defining it as buffer it is a constant not a trainable parameter 
        self.register_buffer('trill', torch.tril(torch.ones(block_size,block_size)))


    def  forward(self,x):

        B,T,C = x.shape   #(C== n_embd)
        q = self.query(x)  # (B,T,head_size)
        k  = self.key(x)   # (B,T,head_size)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        #(B,T,T) = (B,T,Head_szie) @ (B, Head_size, T)
        ## and setting the variance is 1
        # we normalize the wei 
        wei  = wei.masked_fill(self.trill[:T,:T]==0,float('-inf'))
        wei  = F.softmax(wei,dim=-1)
        # getting the value from value layer
        v = self.value(x) # (B,T,head_size)

        # matrix mul
        out = wei @ v  #(B,T,T) @ # (B,T,head_size)

        return out # (B,T, Head_Size) 

       





##defining the initial model 

class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) #(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd) #(block_size,#n_embd)
        ### the self-attention mechanism
        self.sa_head =MultiHeadAttention(num_heads,n_embd//num_heads) #(n_embd,#nembd)
        ##### 
        self.lm_head = nn.Linear(n_embd,vocab_size) #(n_embd,vocab_size)
       
    def forward(self,idx,targets=None):

        ## doing the embedding using the embedding dimension
        B,T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb =  self.position_embedding_table(torch.arange(T))
        x = token_emb + pos_emb  #(B,T,n_embd)
        ## passing the x through the self attention 
        x = self.sa_head(x)
        ## getting the logits from the final layer
        logits = self.lm_head(x)

        if targets is None:
             return logits,None
        else:
            B,T,C = logits.shape           ## here C == n_embd
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
            return logits,loss

   # stopped at step 4 this method do this first thing in the morning
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)      # here (B,T,vocab_size)
            logits = logits[:, -1, :]              # (B,vocab_size)
            probs = F.softmax(logits, dim=-1)      # (B,vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        
m = BigramLanguageModel(vocab_size)

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

#defining the optimizer
optimizer = torch.optim.Adam(m.parameters(),lr)


for step in range(max_steps):

    if step % step_interval ==0:
        losses = estimate_loss()
        print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")


    xb,yb = get_batch('train')
    # forward propagation of the network
    logits,loss = m(xb,yb)
    # setting the gradients to zero before doing the back propagation
    optimizer.zero_grad(set_to_none=True)
    # backpropagation
    loss.backward()
    # updating the weights using the optimizer adam
    optimizer.step()

# printing the final loss
print(loss)





idx = torch.zeros((1,1), dtype=torch.long)                     # start from the newline token
print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))