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
from torch import nn
from torch import Tensor as T, LongTensor as LT
import torch.nn.functional as F
from jaxtyping import Float, Integer

random.seed(1_337)
torch.manual_seed(1_337)

# HACK: jaxtyping doesn't like longtensor, so reassign to plain tensor
LT = T
BATCH_SIZE: int = 32
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
EMBED_DIM: Final[int] = 32


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

CTX_LEN: int = 8  # v important that constants be in SCREAMING_SNAKE_CASE

split = int(0.9 * len(data))
train, val = data[:split], data[split:]

# %%
x, y = train[:CTX_LEN], train[1 : CTX_LEN + 1]

# We use different
for t in range(CTX_LEN):
    input, target = x[: t + 1], y[t]  # y[t] = x[t+1]
    print(input, target)
    # print(decode(input), decode(target))


# %%
def get_batch(mode: Literal["train", "val"]) -> tuple[Tensor, Tensor]:
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


# %% bigram language model
class Bigram(nn.Module):
    def __init__(
        self, vocab_size: int = VOCAB_SIZE, embed_dim: int = EMBED_DIM
    ) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.pos_emb = nn.Embedding(CTX_LEN, embed_dim)
        self.lang_head = nn.Linear(embed_dim, vocab_size)

        self.model: Callable[[Integer[LT, "b seq"]], Float[T, "b seq"]] = nn.Sequential(
            self.token_embeddings,
            Rearrange("b seq embed -> b embed seq"),
            self.lang_head,
        )

        # this can represent a bigram model since the 2d matrix gives "probability of col given row"

    def forward(
        self, idxs: Integer[LT, "b seq"], targets: Integer[LT, "b seq"] | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """
        idxs: batch of indexes to represent a sentence.
        """
        logits: Float[T, "b seq"] = self.model(idxs)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(
        self, idxs: Integer[LT, "b seq"], max_new_toks: int
    ) -> Integer[LT, "b seq+max_new_toks"]:
        for i in range(max_new_toks):
            logits: Integer[LT, "b embed seq"]
            logits, loss = self(idxs)
            logits: Integer[LT, "b embed"] = logits[:, :, -1]
            probs: Integer[LT, "b embed"] = logits.softmax(dim=-1)
            next_idx: Integer[LT, "b 1"] = probs.multinomial(num_samples=1)
            idxs: Integer[LT, "b i+1"] = torch.cat([idxs, next_idx], dim=1)

        return idxs


bigram = Bigram(vocab_size=len(chars))
# %% shape experiment
# bigram(torch.tensor(1)).shape


# %%
assert next(bigram.token_embeddings.parameters()).shape == (
    len(chars),
    len(chars),
)
# %%
bigram(xb, yb)

bigram.generate(idxs=torch.zeros((1, 1)).long(), max_new_toks=10)
# %%
optimizer = torch.optim.AdamW(
    bigram.parameters(), lr=1e-3
)  # TODO test out 1e-2,1e-3, 3e-4


# training: call on batches
for epoch in range(10_000):
    xb, yb = get_batch("train")

    logits, loss = bigram(idxs=xb, targets=yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(loss.item())

# %%
print(decode(bigram.generate(idxs=torch.zeros((1, 1)).long(), max_new_toks=100)[0]))


# %%
def estimate_loss(model):
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
# %%
# the mathematical trick in self-attention

torch.manual_seed(1_337)  # follow karpathy's seed for repro
B, T, C = 8, 3, 32
HEAD_SIZE: int = 16
x = torch.randn(B, T, C)
x_bow = torch.zeros(B, T, C)

for b in range(B):
    for t in range(T):
        x_bow[b, t] = x[b, : t + 1].mean(dim=0)
print(x_bow.shape)
mask = torch.full((T, T), float("-inf")).triu(diagonal=1)
assert (mask @ x_bow).shape == (B, T, C)

# %%
# version 4: self attention

key, query = nn.Linear(C, HEAD_SIZE, bias=False), nn.Linear(C, HEAD_SIZE, bias=False)
k, q = key(x), query(x)  # B T H,  B T H
weights = einsum(k, q, "b seq h, b seq2 h -> b seq seq2")  #: Float[T, "b seq seq"]


def mask_out(x) :
#    : Float[T, "b t t"]->Float[T, "b t t"]
    _, T, _ = x.shape
    return x.masked_fill(x.tril() == 0, float("-inf")).softmax(dim=-1)


masked = mask_out(weights)
assert torch.allclose(masked[0].sum(),torch.tensor(masked[0].shape[-1]).float())
# %%
