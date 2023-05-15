# %%
import itertools
import math
import random
import statistics
import string
import urllib.request
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Callable, Final, Literal, Sequence

import bidict
import torch
import torch.nn.functional as F
import tyro
from einops import einsum, pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange, Reduce
from jaxtyping import Float, Integer
from torch import Tensor
from torch import Tensor as TT
from torch import jit, nn, vmap
from tqdm import trange

random.seed(1_337)
torch.manual_seed(1_337)  # follow karpathy's seed for repro
# TODO: add multi-head
# TODO: vmap
# HACK: jaxtyping doesn't like longtensor, so reassign to plain tensor
LT = TT


@dataclass
class Args:
    # v important that constants be in SCREAMING_SNAKE_CASE
    BATCH_SIZE: Final[int] = 32
    CTX_LEN: Final[int] = 12

    HEAD_SIZE: Final[int] = 64
    N_HEADS: Final[int] = 8
    EMBED_DIM: Final[int] = HEAD_SIZE * N_HEADS

    N_LAYERS: Final[int] = 3
    N_ITERS: Final[int] = 5_000
    DROP_FRAC: Final[float] = 0.0


args = Args()

# time aka ctx_len aka seq
B, T, C = args.BATCH_SIZE, args.CTX_LEN, args.EMBED_DIM


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
x, y = train[: args.CTX_LEN], train[1 : args.CTX_LEN + 1]

# We use different
for t in range(args.CTX_LEN):
    input, target = x[: t + 1], y[t]  # y[t] = x[t+1]


# %%
def get_batch(mode: Literal["train", "val"]) -> tuple[TT, TT]:
    data = train if mode == "train" else val
    random_idxs = torch.randint(high=len(data) - args.CTX_LEN, size=(args.BATCH_SIZE,))
    inputs: Integer[LT, "b ctx_len"] = torch.stack(
        [data[i : i + args.CTX_LEN] for i in random_idxs]
    )
    outputs: Integer[LT, "b ctx_len"] = torch.stack(
        [data[i + 1 : i + 1 + args.CTX_LEN] for i in random_idxs]
    )
    return inputs, outputs


xb, yb = get_batch("train")
for b, t in itertools.product(range(args.BATCH_SIZE), range(args.CTX_LEN)):
    ctx, tgt = xb[b, : t + 1], yb[b, t]

# %%
# Feedforward module


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int = args.EMBED_DIM):
        super().__init__()
        # 4 * multiplier is from original attention paper, where d_model = 512 and d_ff = 2048
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(args.DROP_FRAC),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# %%
# gpt(xb, yb)
class MultiHead(nn.Module):
    def __init__(
        self,
        head_size: int = args.HEAD_SIZE,
        embed_dim: int = args.EMBED_DIM,
        n_heads: int = args.N_HEADS,
    ):
        super().__init__()
        self.head_size = embed_dim // n_heads
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.key = nn.Linear(embed_dim, embed_dim)  # B C
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(args.DROP_FRAC)
        self.register_buffer(
            "mask",
            rearrange(torch.ones(args.CTX_LEN, args.CTX_LEN), "t t2 -> 1 1 t t2"),
        )

    def forward(self, x: Tensor) -> Tensor:  #: Float[TT, 'b t t2']
        def mask_out(x: Tensor) -> Tensor:
            #    : Float[TT, "b nh t t"]->Float[TT, "b nh t t"]
            B, nh, T, C = x.shape
            return x.masked_fill(self.mask[..., :T, :T], float("-inf")).softmax(dim=-1)

        k, q, v = self.key(x), self.query(x), self.value(x)
        nh, c = self.n_heads, self.embed_dim // self.n_heads

        weights: Tensor = einsum(
            rearrange(k, "b t (nh c) -> b nh t c", nh=nh),
            rearrange(q, "b t (nh c) -> b nh t c", nh=nh),
            "b nh t c, b nh t2 c -> b nh t t2",
        )  #: Float[TT, "b nh t t"]

        masked = mask_out(weights)
        masked = self.dropout(masked)

        out: Tensor = einsum(
            masked,
            rearrange(v, "b t (nh c) -> b nh t c", nh=nh),
            "b nh t t, b nh t c -> b nh t c",
        )
        out = rearrange(out, "b nh t c -> b t (nh c)")

        out = self.proj(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, dims: int | Sequence[int]) -> None:
        super().__init__()
        self.mean_coeff, self.bias_coeff = nn.Parameter(torch.ones(dims)), nn.Parameter(
            torch.zeros(dims)
        )

    def forward(self, x: Tensor, eps: Number = 1e-5) -> Tensor:
        # why keepdim?
        # Unlike Andrej, we take mean over last (channel) dim
        return (
            self.mean_coeff
            * (x - x.mean(-1, keepdim=True))
            / torch.sqrt(x.var(-1, keepdim=True) + eps)
        ) + self.bias_coeff


class ResBlock(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: TT) -> TT:
        return self.fn(x) + x


class Block(nn.Module):
    def __init__(self, embed_dim: int = args.EMBED_DIM, n_heads: int = args.N_HEADS):
        super().__init__()

        self.sa = MultiHead(n_heads=n_heads, head_size=embed_dim // n_heads)
        self.ffwd = FeedForward(embed_dim)
        self.ln1, self.ln2 = LayerNorm(embed_dim), LayerNorm(embed_dim)
        # equivalent:
        # self.model = nn.Sequential(
        #     ResBlock(fn=nn.Sequential(self.ln1, self.sa)),
        #     ResBlock(nn.Sequential(self.ln2, self.ffwd)),
        # )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = args.EMBED_DIM,
        n_heads: int = args.N_HEADS,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pos_emb = nn.Embedding(args.CTX_LEN, embed_dim)
        self.lang_head = nn.Linear(embed_dim, vocab_size)
        self.blocks = nn.Sequential(
            *[Block(embed_dim, n_heads) for _ in range(args.N_LAYERS)]
        )
        self.final_ln = LayerNorm(embed_dim)

    def forward(
        self, idxs: Integer[LT, "b t"], targets: Integer[LT, "b t"] | None = None
    ) -> tuple[TT, TT | None]:
        """
        idxs: batch of indexes to represent a sentence.
        """
        B, T = idxs.shape
        x = self.tok_emb(idxs) + self.pos_emb(torch.arange(T))

        x = self.blocks(x)
        x = self.final_ln(x)
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
            logits, loss = self(idxs[:, -args.CTX_LEN :])
            logits: Integer[LT, "b embed"] = logits[:, -1, :]
            probs: Integer[LT, "b embed"] = logits.softmax(dim=-1)

            next_idx: Integer[LT, "b 1"] = probs.multinomial(num_samples=1)
            idxs: Integer[LT, "b i+1"] = torch.cat([idxs, next_idx], dim=1)

        return idxs


gpt = GPT()
gpt.generate(idxs=torch.zeros((args.BATCH_SIZE, args.CTX_LEN)).long(), max_new_toks=10)
# %%
optimizer = torch.optim.AdamW(
    gpt.parameters(), lr=1e-3
)  # TODO test out 1e-2,1e-3, 3e-4


# training: call on batches
for epoch in trange(10_000):
    xb, yb = get_batch("train")

    logits, loss = gpt(idxs=xb, targets=yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(loss.item())

# %%
print(
    decode(
        gpt.generate(
            idxs=torch.zeros((args.BATCH_SIZE, args.CTX_LEN)).long(), max_new_toks=100
        )[0]
    )
)


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
estimate_loss(gpt)

# %%
