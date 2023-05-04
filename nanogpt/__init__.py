# %%

from einops import einsum, pack, unpack
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch
import itertools
import random
import string
from typing import Callable, Final
from pathlib import Path
import math
import statistics
from torch import Tensor
from torch import vmap, jit
import urllib.request
import torch
from typing import Literal
import bidict
from tqdm import trange
from torch import nn
from torch import Tensor as TT
import torch.nn.functional as F
from jaxtyping import Float, Integer

random.seed(1_337)
torch.manual_seed(1_337)  # follow karpathy's seed for repro
# TODO: add multi-head
# TODO: vmap
# HACK: jaxtyping doesn't like longtensor, so reassign to plain tensor
LT = TT
BATCH_SIZE: Final[int] = 16  # v important that constants be in SCREAMING_SNAKE_CASE
CTX_LEN: Final[int] = 8
EMBED_DIM: Final[int] = 32
# time aka ctx_len aka seq
B, T, C = BATCH_SIZE, CTX_LEN, EMBED_DIM
HEAD_SIZE: Final[int] = EMBED_DIM
N_HEADS: Final[int] = 8
INPUT_FILE = Path("input.txt")
data = urllib.request.urlretrieve(
    url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    filename=INPUT_FILE,
)
raw_data = INPUT_FILE.read_text()

chars = sorted(set(raw_data))
text = "".join(chars)


# %% Tokenization


# encoder: str -> [int]
# decoder: [int] -> str
# init codebook/vocab

VOCAB: Final[bidict.bidict[int, str]] = bidict.bidict(enumerate(chars))
VOCAB_SIZE: Final[int] = len(VOCAB)


# %%
def encode(s: str) -> list[int]:
    return [VOCAB.inv[c] for c in s]


# %%
def decode(l: list[int] | list[LT]) -> str:
    if isinstance(l, LT) or isinstance(l[0], LT):
        return "".join([VOCAB[i.item()] for i in l])
    else:
        return "".join([VOCAB[i] for i in l])


# %%
s = "".join(random.choices(string.ascii_lowercase, k=10))
assert decode(encode(s)) == s
assert encode("abc") == [39, 40, 41]
assert decode([41, 40, 39]) == "cba"

# %% Encode dataset and store in a torch tensor
data = torch.tensor(encode(raw_data), dtype=torch.long)  #


# %% Split into train and test sets

split = int(0.9 * len(data))
train, val = data[:split], data[split:]

# %%
x, y = train[:CTX_LEN], train[1 : CTX_LEN + 1]

# We use different
for t in range(CTX_LEN):
    input, target = x[: t + 1], y[t]  # y[t] = x[t+1]


# %%
def get_batch(mode: Literal["train", "val"]) -> tuple[TT, TT]:
    data = train if mode == "train" else val
    random_idxs = torch.randint(high=len(data) - CTX_LEN, size=(BATCH_SIZE,))
    inputs: Integer[LT, "b ctx_len"] = torch.stack(
        [data[i : i + CTX_LEN] for i in random_idxs]
    )
    outputs: Integer[LT, "b ctx_len"] = torch.stack(
        [data[i + 1 : i + 1 + CTX_LEN] for i in random_idxs]
    )
    return inputs, outputs


xb, yb = get_batch("train")
for b, t in itertools.product(range(BATCH_SIZE), range(CTX_LEN)):
    ctx, tgt = xb[b, : t + 1], yb[b, t]

# %%
# Feedforward module


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


# %%
# bigram(xb, yb)
class Head(nn.Module):
    def __init__(self, head_size: int = HEAD_SIZE, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

    def forward(self, x):  #: Float[TT, 'b t t2']
        def mask_out(x):
            #    : Float[TT, "b t t"]->Float[TT, "b t t"]
            _, T, _ = x.shape
            return x.masked_fill(x.tril() == 0, float("-inf")).softmax(dim=-1)

        k, q, v = self.key(x), self.query(x), self.value(x)
        weights = einsum(k, q, "b t c, b t2 c -> b t t2")  #: Float[TT, "b t t"]
        masked = mask_out(weights)
        out = masked @ v
        return out


class MultiHead(nn.Module):
    def __init__(
        self,
        n_heads: int = N_HEADS,
        head_size: int = HEAD_SIZE,
        embed_dim: int = EMBED_DIM,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(self.n_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x) -> TT:
        # TODO replace cat with pack
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: TT) -> TT:
        return self.fn(x) + x


class Block(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, n_heads: int = N_HEADS):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(MultiHead(n_heads, head_size=embed_dim // n_heads)),
            ResBlock(FeedForward(embed_dim)),
        )

    def forward(self, x):
        return self.net(x)


class Bigram(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        n_heads: int = N_HEADS,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pos_emb = nn.Embedding(CTX_LEN, embed_dim)
        self.lang_head = nn.Linear(embed_dim, vocab_size)
        self.blocks = nn.Sequential(*[Block(embed_dim, n_heads) for _ in range(3)])
        # self.s_attn_heads = MultiHead(self.n_heads, embed_dim // self.n_heads)
        # self.feed_forward = FeedForward(embed_dim)
        # this can represent a bigram model since the 2d matrix gives "probability of col given row"

    def forward(
        self, idxs: Integer[LT, "b t"], targets: Integer[LT, "b t"] | None = None
    ) -> tuple[TT, TT | None]:
        """
        idxs: batch of indexes to represent a sentence.
        """
        B, T = idxs.shape
        x = self.tok_emb(idxs) + self.pos_emb(torch.arange(T))
        x = self.blocks(x)
        logits: Float[TT, "b t vocab"] = self.lang_head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(
                rearrange(logits, "b t vocab -> (b t) vocab"),
                rearrange(targets, "b t -> (b t)"),
            )
        return logits, loss

    def generate(
        self, idxs: Integer[LT, "b t"], max_new_toks: int
    ) -> Integer[LT, "b t+max_new_toks"]:
        for i in range(max_new_toks):
            logits: Integer[LT, "b t embed"]
            # crop idxs to avoid messing with positional embedding
            logits, loss = self(idxs[:, -CTX_LEN:])
            logits: Integer[LT, "b embed"] = logits[:, -1, :]
            probs: Integer[LT, "b embed"] = logits.softmax(dim=-1)
            next_idx: Integer[LT, "b 1"] = probs.multinomial(num_samples=1)
            idxs: Integer[LT, "b i+1"] = torch.cat([idxs, next_idx], dim=1)

        return idxs


bigram = Bigram()
bigram.generate(idxs=torch.zeros((1, 1)).long(), max_new_toks=10)
# %%
optimizer = torch.optim.AdamW(
    bigram.parameters(), lr=1e-3
)  # TODO test out 1e-2,1e-3, 3e-4


# training: call on batches
for epoch in trange(10_000):
    xb, yb = get_batch("train")

    logits, loss = bigram(idxs=xb, targets=yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(loss.item())

# %%
print(decode(bigram.generate(idxs=torch.zeros((1, 1)).long(), max_new_toks=100)[0]))


# %%
def estimate_loss(model: nn.Module):
    metrics = {"train": [], "val": []}

    with torch.no_grad():
        for split in ("train", "val"):
            for iter in range(10_000):
                xb, yb = get_batch(split)
                logits, loss = model(xb, yb)
                metrics[split].append(loss.item())
            metrics[split] = statistics.mean(metrics[split])
    return metrics


# %%
estimate_loss(bigram)
