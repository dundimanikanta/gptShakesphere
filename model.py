import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
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

lr = 1e-3

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) #(65,n_embd)
        self.position_embedding_table = nn.Embedding(n_embd,n_embd) #(n_embd,#n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size) #(n_embd,65)

    def forward(self,idx,targets=None):

        ## doing the embedding using the embedding dimension
        B,T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb =  self.token_embedding_table(torch.arange(T))
        x = token_emb + pos_emb 

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