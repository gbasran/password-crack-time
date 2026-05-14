"""
PassBERT-lite: a small char level autoregressive transformer that learns
the distribution of human chosen passwords and emits candidate plaintexts
in approximate descending probability order.

runs as a regular python script or pasted into a colab notebook. either
way you get the same model and the same `neural_candidates.txt` at the end.

scope (deliberately small):
- char level, vocab ~50 chars on the seclists 10k (more on rockyou)
- 4 transformer blocks, 4 heads, d_model 128 -> ~810k params
- trains in ~10-20 min on a colab t4 (much slower on cpu, ok for testing)
- generates 100k candidates by nucleus sampling + dedup
- output: candidates.txt ranked by joint log probability

why neural vs markov: markov is k-gram limited, cant model long range
structure like "capital letter at position 0 implies digits at the end"
because the context window is only 2-3 chars. a transformer with self
attention sees the WHOLE prefix at every step, so it picks up patterns
like "if i started with a name, end with 4 digits" that markov misses.
melicher 2016 / pasquini 2021 show neural models crack ~25-50% more
hashes than markov at the same budget.
"""

# on colab, torch is preinstalled. locally:  pip install torch
import os
import math
import time
import random
import urllib.request
import hashlib
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# auto detect device. on colab, runtime > change runtime type > gpu gets
# you a t4 (or better with pro). on a mac m1/m2 use 'mps'
DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE)


# ---- hyperparameters ----
# tunable. the defaults target a colab t4 (16gb vram) and ~20 min total
# runtime. bump CONTEXT, D_MODEL, N_LAYERS for stronger results at the
# cost of train time
SEED        = 42
CONTEXT     = 32     # max password length we model
D_MODEL     = 128    # embedding + hidden dim
N_HEADS     = 4
N_LAYERS    = 4
DROPOUT     = 0.1
BATCH_SIZE  = 256
EPOCHS      = 3
LR          = 3e-4
N_TRAIN     = 200_000  # subset size, colab can comfortably do 1M+
N_CANDIDATES_TO_EMIT = 100_000

torch.manual_seed(SEED); random.seed(SEED)


# ---- data ----
# we use the seclists 10k for portability. for real results swap in a
# bigger corpus. urls that work as drop in replacements:
#
#   rockyou-75 (top 75% of rockyou, ~10M):
#     https://github.com/zacheller/rockyou/raw/master/rockyou.txt.tar.gz
#   haveibeenpwned v8 (1B unique, hashed, need the cleartext form)
#
# colab users: upload your own wordlist.txt with the file panel on left

WORDLIST_URL = ("https://raw.githubusercontent.com/danielmiessler/SecLists/"
                "refs/heads/master/Passwords/Common-Credentials/"
                "10k-most-common.txt")
WORDLIST_PATH = "data/seclist_10k.txt"

def load_corpus(path=WORDLIST_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(WORDLIST_URL, path)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [w.strip() for w in f if 1 <= len(w.strip()) <= CONTEXT - 2]

corpus = load_corpus()
print(f"loaded {len(corpus):,} passwords")


# ---- tokenizer ----
# char level vocab: every distinct character we see in the corpus, plus
# three special tokens (pad, start, end). building the vocab from the data
# instead of hardcoding ascii printable keeps the embedding matrix small
# and means non ascii passwords (spanish accents, russian cyrillic, emoji)
# work without code changes

PAD, BOS, EOS = "\x00", "\x02", "\x03"
charset = sorted({c for p in corpus for c in p})
itos = [PAD, BOS, EOS] + charset
stoi = {c: i for i, c in enumerate(itos)}
VOCAB = len(itos)
print(f"vocab size = {VOCAB}")

def encode(s, max_len=CONTEXT):
    """str -> tensor of token ids, padded/truncated to max_len."""
    ids = [stoi[BOS]] + [stoi[c] for c in s[:max_len-2]] + [stoi[EOS]]
    ids = ids + [stoi[PAD]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def decode(ids):
    """tensor of token ids -> str, stripping bos/eos/pad."""
    chars = []
    for i in ids.tolist():
        if i == stoi[EOS]: break
        if i not in (stoi[BOS], stoi[PAD]):
            chars.append(itos[i])
    return "".join(chars)


# ---- dataset ----
class PasswordDataset(Dataset):
    def __init__(self, passwords):
        self.data = [encode(p) for p in passwords]
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        seq = self.data[i]
        # standard autoregressive shift: input is positions 0..n-1, target
        # is positions 1..n. the model learns P(c_t | c_<t)
        return seq[:-1], seq[1:]

# hold out 5% for a quick perplexity / loss sanity check
random.shuffle(corpus)
n_train = min(N_TRAIN, int(len(corpus) * 0.95))
train_set = PasswordDataset(corpus[:n_train])
val_set   = PasswordDataset(corpus[n_train:n_train + 500])
train_dl  = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dl    = DataLoader(val_set,   batch_size=BATCH_SIZE)


# ---- model ----
# minimal gpt style decoder. each block: causal multi head self attention,
# feed forward, two residual connections, two layernorms (pre norm).
# causal mask is the triangular -inf matrix that stops position t from
# attending to positions > t (so the model cant peek at the answer).
#
# attention + block accept an optional `kv_cache` arg. when None (training,
# full forward) they behave as a normal causal transformer. when a tuple
# (past_k, past_v) is provided (inference), they treat the input as just
# the new token and concat its k/v onto the cache. avoids the O(T^2) work
# of recomputing the whole prefix at every step. roughly a 20x speedup on
# attention for our 32-char context.

class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads, ctx, dropout):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n_heads = n_heads; self.d_head = d // n_heads
        self.dropout = dropout

    def forward(self, x, kv_cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        # (B, T, D) -> (B, n_heads, T, d_head) for multi head
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if kv_cache is not None:
            # incremental decode: concat the new tokens k/v onto the cache
            past_k, past_v = kv_cache
            if past_k.numel() > 0:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
            # T_q=1 attends to ALL of T_k (past + self). no causal mask needed
            # because by construction we only ever cache tokens that came before
            is_causal = False
        else:
            new_cache = None
            is_causal = True

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out), new_cache

class Block(nn.Module):
    def __init__(self, d, n_heads, ctx, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads, ctx, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.drop = nn.Dropout(dropout)
    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + self.drop(attn_out)                  # pre norm residual
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, new_cache

class PasswordTransformer(nn.Module):
    def __init__(self, vocab, ctx, d=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, ctx, dropout)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.ctx = ctx

    def forward(self, idx):
        # training path: full forward, blocks return (x, cache) but we
        # discard the cache since we dont need it for the loss
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x, _ = blk(x)
        return self.head(self.ln_f(x))              # (B, T, vocab)

    @torch.no_grad()
    def generate(self, n_samples, max_len, bos_id, eos_id, bos_mask_id,
                 top_p=0.85, temperature=0.9, use_amp=True):
        """KV-cached, optionally fp16-autocast batched generation.

        n_samples: how many sequences to generate IN PARALLEL (the batch).
        returns (token_ids tensor of shape (n_samples, len), log_probs tensor).

        the speedup vs naive sampling comes from three things:
        1. KV cache: each step only computes attention for the NEW token
        2. autocast fp16: matmuls run on tensor cores, ~2-4x on T4
        3. one device sync at end of generation, not per step
        """
        self.eval()
        device = next(self.parameters()).device
        cur = torch.full((n_samples, 1), bos_id, dtype=torch.long, device=device)
        log_probs_total = torch.zeros(n_samples, device=device)
        done = torch.zeros(n_samples, dtype=torch.bool, device=device)
        caches = [None] * len(self.blocks)

        amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                   if use_amp and device.type == "cuda"
                   else torch.amp.autocast(device_type="cpu", enabled=False))

        with amp_ctx:
            for step in range(max_len - 1):
                # input is just the new token (BOS on step 0, sampled char after)
                x_tok = self.tok_emb(cur[:, -1:])                       # (B, 1, D)
                pos_idx = torch.tensor([step], device=device)
                x = x_tok + self.pos_emb(pos_idx)[None, :, :]

                for i, blk in enumerate(self.blocks):
                    x, caches[i] = blk(x, kv_cache=caches[i])

                # softmax + sampling math in fp32 for numerical stability
                logits = self.head(self.ln_f(x))[:, -1, :].float() / temperature
                logits[:, bos_mask_id] = -float("inf")
                probs = F.softmax(logits, dim=-1)

                # top-p (nucleus) filter
                sorted_p, sorted_i = probs.sort(descending=True)
                cum = sorted_p.cumsum(dim=-1)
                mask = cum - sorted_p > top_p
                sorted_p = sorted_p.masked_fill(mask, 0.0)
                sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                idx_in_sorted = torch.multinomial(sorted_p, 1)
                nxt = sorted_i.gather(-1, idx_in_sorted)

                chosen_p = probs.gather(-1, nxt).squeeze(-1).clamp(min=1e-12)
                log_probs_total = log_probs_total + torch.where(
                    done, torch.zeros_like(chosen_p), chosen_p.log())
                done = done | (nxt.squeeze(-1) == eos_id)

                cur = torch.cat([cur, nxt], dim=1)
                if done.all(): break

        return cur, log_probs_total

model = PasswordTransformer(VOCAB, CONTEXT).to(DEVICE)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")


# ---- training ----
opt = torch.optim.AdamW(model.parameters(), lr=LR)
PAD_ID = stoi[PAD]

def run_epoch(dl, train):
    model.train() if train else model.eval()
    total, count = 0.0, 0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        # ignore_index=PAD makes loss skip padding tokens, important because
        # most positions in a short password are pad
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                               ignore_index=PAD_ID)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x.size(0); count += x.size(0)
    return total / count

print("training...")
for epoch in range(EPOCHS):
    t0 = time.perf_counter()
    train_loss = run_epoch(train_dl, train=True)
    with torch.no_grad():
        val_loss = run_epoch(val_dl, train=False)
    print(f"  epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.3f}  "
          f"val_loss={val_loss:.3f}  ({time.perf_counter()-t0:.1f}s)")


# ---- candidate generation ----
# nucleus (top-p) sampling with KV cache + fp16 autocast + big batch.
# at each step we sort the next char distribution, take the smallest set
# of tokens whose cumulative probability >= p, sample from that set, and
# dedupe across all generated sequences.
#
# all heavy lifting happens inside model.generate() which uses the kv
# cache so step t only attends to the new token (vs the naive version
# that re-processes the whole prefix every step). batch size is set high
# enough to actually load the T4: 4096 sequences in parallel is well
# within the ~1GB the model + activations need.

# bump SAMPLE_BATCH if you have an A100 or want more parallelism, drop
# if running on a smaller GPU and you OOM
SAMPLE_BATCH = 4096
SAMPLE_TOP_P = 0.85
SAMPLE_TEMP  = 0.9

def sample(n, max_len=CONTEXT, top_p=SAMPLE_TOP_P, temperature=SAMPLE_TEMP,
           batch=SAMPLE_BATCH):
    """generate `n` unique candidates. returns list of (password, log_prob)."""
    seen = set()
    out = []
    while len(out) < n:
        cur, log_probs = model.generate(
            n_samples=batch, max_len=max_len,
            bos_id=stoi[BOS], eos_id=stoi[EOS], bos_mask_id=stoi[BOS],
            top_p=top_p, temperature=temperature,
            use_amp=(DEVICE == "cuda"),
        )
        # single sync per batch: pull everything off the GPU at once,
        # then dedupe in python. way less CPU<->GPU chatter than going
        # through the loop on device
        cur_cpu = cur.cpu().tolist()
        lp_cpu = log_probs.cpu().tolist()
        for i in range(batch):
            # inline decode, avoids re-allocating a tensor per row
            chars = []
            for tok in cur_cpu[i]:
                if tok == stoi[EOS]: break
                if tok != stoi[BOS] and tok != stoi[PAD]:
                    chars.append(itos[tok])
            pw = "".join(chars)
            if pw and pw not in seen:
                seen.add(pw)
                out.append((pw, lp_cpu[i]))
                if len(out) >= n: break
    return sorted(out, key=lambda x: -x[1])         # descending log_prob

print(f"sampling {N_CANDIDATES_TO_EMIT:,} candidates "
      f"(batch={SAMPLE_BATCH}, top_p={SAMPLE_TOP_P}, T={SAMPLE_TEMP})...")
t0 = time.perf_counter()
candidates = sample(N_CANDIDATES_TO_EMIT)
elapsed = time.perf_counter() - t0
print(f"  generated {len(candidates):,} unique in {elapsed:.1f}s "
      f"({len(candidates)/elapsed:,.0f} cand/s)")
print("  top 20:")
for pw, lp in candidates[:20]:
    print(f"    {pw!r:30}  logP={lp:.2f}")


# ---- export wordlist for hashcat ----
OUT_PATH = "neural_candidates.txt"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for pw, _ in candidates:
        f.write(pw + "\n")
print(f"wrote {len(candidates):,} candidates to {OUT_PATH}")


# ---- demo target hashes ----
# hashcat -m 0 (raw md5) expects one 32 char hex digest per line. if the
# file has plaintexts or anything else you get "token length exception"
# and "no hashes loaded". so we generate a small set of targets at varied
# difficulty levels and write the md5s out as `hashes.txt`. swap these
# for real stolen hashes when you have them.

demo_targets = [
    "monkey",            # in seclists, neural cracks instantly
    "summer2024",        # high prob shape
    "iloveyou1",         # in seclists
    "password123",       # in seclists
    "qwerty2024",        # high prob shape
    "Tr0ub4dor&3",       # xkcd mangled, probably wont crack at this budget
]

HASHES_PATH = "hashes.txt"
with open(HASHES_PATH, "w") as f:
    for p in demo_targets:
        f.write(hashlib.md5(p.encode()).hexdigest() + "\n")
print(f"wrote {len(demo_targets)} target hashes to {HASHES_PATH}")
print(f"\nrun:  hashcat -a 0 -m 0 {HASHES_PATH} {OUT_PATH} --quiet --potfile-disable")
print(f"then: hashcat --show {HASHES_PATH}")
