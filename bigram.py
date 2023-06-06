import os

import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
max_iters = 5_001
eval_interval = 500
learning_rate = 1e-3
min_lr = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_layers = 6
n_heads = 6
dropout = 0.2
torch.manual_seed(1337)
# import wget
# wget.download(url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()

# extract all unique characters and sort them
chars = sorted(list(set(text)))
# the amount of unique character is the lenght of the previous list
vocab_size = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda input_text: [char_to_int[s] for s in list(input_text)]
decode = lambda list_of_ints: "".join([int_to_char[i] for i in list_of_ints])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
train_split = 0.9 # 90 / 10 split
n = int(train_split * len(data))  
train_data = data[:n]
val_data = data[n:]
BEST_WEIGHTS = "./weights/best_model.pth"

def get_batch(split):
    # generate a small batch of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, 16)
        q = self.query(x)  # (B, T, 16)
        v = self.value(x)  # (B, T, 16)

        # compute attention scores
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, 16) @ (B, 16, T) --> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads in parallel
    """

    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    a simple feed forward network
    """

    def __init__(self, n_embed):
        super().__init__()
        self.ffw = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffw(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        # each token directly reads off the logits for
        # the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_heads)
                                      for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd, vocab_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        #self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # B,T, EMBED_C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # T,C
        x = token_emb + pos_emb  # word embedding + positional embeddings
        x = self.blocks(x)  # multi-head attention
        x = self.ln_f(x)  # layer_norm
        logits = self.lm_head(x)  # B,T, VOCAB_SIZE

        if targets is None:
            loss = None
        else:
            # (Batch=batch_size, Time = block_size, Channel = vocab_size)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)  # could also use targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is  (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions => self here is the model
            # =>same as doing: logits, loss = m(xb, yb)
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # focus only on the last time step (that of the T timension)
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# Create GPT model
model = BigramLanguageModel().to(device)

# Create Pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=min_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=min_lr)

pretrained=False
if pretrained:
	checkpoint = torch.load(BEST_WEIGHTS)
	model.load_state_dict(checkpoint['model_state'])
	optimizer.load_state_dict(checkpoint['optim_state'])
	epoch = checkpoint['epoch']
	train_loss = checkpoint['train_loss']
	val_loss = checkpoint['val_loss']

best_model_loss = 1e4  # arbitrary large loss
os.makedirs("weights", exist_ok=True)

with open("evaluation_results.txt", 'a') as file:
    file.writelines(f"train_test_split={(train_split, round(1 - train_split,2))},{learning_rate=}, {dropout=}, "
                    f"scheduler={str(type(scheduler)).split('.')[-1][:-2]}, "
                    f"optimizer={str(type(optimizer)).split('.')[-1][:-2]} \n"
                    f"------------------------------------------------------------- \n")

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, validation loss: {losses['val']:.4f}")
        with open("evaluation_results.txt", 'a') as file:
            file.writelines(f"step {iter}: train loss {losses['train']:.4f}, validation loss: {losses['val']:.4f} \n")
        if losses['val'] < best_model_loss:
            checkpoint = {
                "epoch": iter,
                "train_loss": losses['train'],
                "val_loss": losses['val'],
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join("weights", "best_model.pth"))
            best_model_loss = losses['val']

    # sample a new batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
