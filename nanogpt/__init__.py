# %%
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
import torch.nn.functional as F

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
def decode(l: list[int] | list[torch.LongTensor]) -> str:
    if isinstance(l, torch.LongTensor) or isinstance(l[0], torch.LongTensor):
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
# random.seed(48)

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
random.seed(1337)
torch.manual_seed(1337)

BATCH_SIZE: int = 4


# %%
def get_batch(mode: Literal["train", "val"]) -> tuple[Tensor, Tensor]:
    data = train if mode == "train" else val
    random_idxs = torch.randint(high=len(data) - CTX_LEN, size=(BATCH_SIZE,))
    inputs = torch.stack([data[i : i + CTX_LEN] for i in random_idxs])
    outputs = torch.stack([data[i + 1 : i + 1 + CTX_LEN] for i in random_idxs])
    return inputs, outputs


xb, yb = get_batch("train")
for b, t in itertools.product(range(BATCH_SIZE), range(CTX_LEN)):
    ctx, tgt = xb[b, : t + 1], yb[b, t]


# %% bigram language model
class Bigram(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, vocab_size)
        print(self.token_embeddings)

    def forward(self, args: Tensor) -> Tensor:
        ...


bigram = Bigram(vocab_size=len(chars))
# %%
