import torch
import torch.nn as nn
from torch.nn import functional as F

# h-params
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading

def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data

  # batch_size num of random offsets (valid indices (0, len(data)-block_size))
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])

  return x, y

xb, yb = get_batch('train')


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
          X, Y = get_batch(split)
          logits, loss = model(X, Y)
          losses[k] = loss.item()
      out[split] = losses.mean()
  model.train()
  return out

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):

    #idx and targets are both (B,T) tensor of integers
    logits = self.token_embedding_table(idx) # (B,T,C)

    if targets is None:
      loss = None
    else:
      #reshape logits into (B*T,C)
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)

      # negative log likelihood (cross entropy)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is size (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      # get the predictions
      logits, loss = self(idx)
      # focus only on the last time step (-1 for last idx)
      logits = logits[:, -1, :] # becomes (B,C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B,C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
    return idx


model = BigramLanguageModel(vocab_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):

  if iter % eval_interval == 0:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
  #sample a batch of data
  xb, yb = get_batch('train')

  # evaluate loss
  logits, loss = m(xb, yb )
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


class Head(nn.Moldule):
   """One head of self-attention"""

   def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias=False)
      self.query  = nn.