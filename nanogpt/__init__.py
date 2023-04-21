# %%
from einops import einsum, pack, unpack
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import itertools
import random
import string
from typing import Final
from pathlib import Path
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

random.seed(1337)
torch.manual_seed(1337)

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
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=vocab_size
        )
        # this can represent a bigram model since the 2d matrix gives "probability of col given row"

    def forward(
        self, idxs: Integer[LT, "b seq"], targets: Integer[LT, "b seq"] | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """
        idxs: batch of indexes to represent a sentence.
        """
        logits: Float[T, "b embed seq"] = rearrange(
            self.token_embeddings(idxs), "b seq embed -> b embed seq"
        )
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
for epoch in range(1000):
    xb, yb = get_batch("train")

    logits, loss = bigram(idxs=xb, targets=yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(loss.item())

# %%
print(decode(bigram.generate(idxs=torch.zeros((1, 1)).long(), max_new_toks=100)[0]))

# %%
