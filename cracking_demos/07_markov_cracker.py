"""
07_markov_cracker.py
====================
OMEN-lite: an ordered markov model password cracker (duermuth et al., 2015).

why this exists.
plain dictionary + rules attacks the password as "word + suffix". a markov
model attacks the password as "a sequence of character n-grams whose joint
probability the model learned from breach data". thats the smarter attacker
the paper needs to defend its length claim against.

architecture:
1. train. count k-gram frequencies in a breach corpus, convert to log probs
   with add 1 smoothing.
2. generate. priority queue beam search. push (negative_log_prob, prefix)
   onto a min heap. pop the most probable incomplete prefix, extend by
   every possible next character, push extensions back. emit prefixes that
   hit an end of word token as complete candidates.
3. crack. hash each emitted candidate, check against a target set.

the priority queue is the key trick. without it you have to enumerate all
|alphabet|^L candidates and sort by probability after the fact. with it
you emit in approximate descending probability order so you can stop
early once budget is exhausted.

usage:
    python3 07_markov_cracker.py                    # demo against built in targets
    python3 07_markov_cracker.py --emit-wordlist out.txt --budget 1000000
    # then on a gpu box:
    # hashcat -a 0 -m 0 hashes.txt out.txt
"""

import argparse
import hashlib
import heapq
import math
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict


# training corpus.
# we reuse the same seclists 10k list the existing notebook downloads. small
# enough to train on instantly, overlaps the rockyou/comb tail enough to be
# representative for a poc. swap WORDLIST_URL for a bigger list (rockyou.txt,
# hibp) to crank up the models coverage

WORDLIST_URL = ("https://raw.githubusercontent.com/danielmiessler/SecLists/"
                "refs/heads/master/Passwords/Common-Credentials/10k-most-common.txt")
WORDLIST_PATH = "data/seclist_10k.txt"

START_TOK = "\x02"   # ascii STX, wont appear in real passwords
END_TOK   = "\x03"   # ascii ETX


def load_corpus(path=WORDLIST_PATH, url=WORDLIST_URL):
    """download the seclists 10k file on first run, return list of passwords."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"  downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [w.strip() for w in f if w.strip() and len(w.strip()) <= 32]


# train a k-gram model.
# for k=3 (the standard choice in OMEN) the model stores P(c_n | c_{n-2}, c_{n-1}).
# the context for the very first character is START_TOK * (k-1). the model
# emits END_TOK to terminate a candidate, we use that tokens position to
# tell how long a generated password is

def train_markov(corpus, k=3):
    """return (log_probs, vocab, k) where log_probs[ctx][c] = log P(c|ctx)."""
    counts = defaultdict(Counter)
    for word in corpus:
        padded = START_TOK * (k - 1) + word + END_TOK
        for i in range(len(padded) - k + 1):
            ctx = padded[i:i + k - 1]
            nxt = padded[i + k - 1]
            counts[ctx][nxt] += 1

    # vocab = every character we ever saw following any context. END_TOK is
    # in here, START_TOK isnt (its never a "next char")
    vocab = sorted({c for ctr in counts.values() for c in ctr})

    # add 1 (laplace) smoothing: P(c|ctx) = (count + 1) / (total + |vocab|).
    # prevents log(0) for never seen transitions, costs a small amount of
    # mass given to the rare ones
    log_probs = {}
    V = len(vocab)
    for ctx, ctr in counts.items():
        total = sum(ctr.values()) + V
        log_probs[ctx] = {c: math.log((ctr.get(c, 0) + 1) / total) for c in vocab}
    return log_probs, vocab, k


def score(model, password):
    """return log P(password) under the model.

    backoff: when the (k-1)-gram context was never observed in training we
    fall back to a uniform distribution over the vocabulary. that gives
    log(1/|vocab|) per character instead of -inf, which is the right
    behaviour for a random char that looks like the alphabet but appears
    in a never seen context. without this, random_uniform targets all
    score -inf and the experiment cant distinguish them.
    """
    log_probs, vocab, k = model
    uniform_lp = -math.log(len(vocab))
    padded = START_TOK * (k - 1) + password + END_TOK
    total = 0.0
    for i in range(len(padded) - k + 1):
        ctx = padded[i:i + k - 1]
        nxt = padded[i + k - 1]
        if ctx in log_probs and nxt in log_probs[ctx]:
            total += log_probs[ctx][nxt]
        else:
            total += uniform_lp                    # backoff
    return total


# ordered enumeration via priority queue.
# we push (cost, prefix) where cost = -log_prob (a non negative number that
# acts like a path cost). pythons heapq is a min heap so popping returns
# the lowest cost = highest probability prefix. each pop expands by every
# possible next character. when we pop a prefix ending in END_TOK we yield
# it as a complete candidate.
#
# this is approximate descending order, not exact. two prefixes of different
# lengths can have similar cost even if extending the shorter one would
# give a more probable complete password. in practice its close enough that
# the first N candidates emitted are roughly the N most probable, which is
# what crackers care about

def enumerate_candidates(model, max_length=16, budget=1_000_000):
    """yield (candidate, log_prob) tuples in ~descending probability order."""
    log_probs, vocab, k = model
    start_ctx = START_TOK * (k - 1)
    heap = [(0.0, start_ctx)]                     # (cost, prefix-incl-padding)
    emitted = 0

    while heap and emitted < budget:
        cost, prefix = heapq.heappop(heap)
        ctx = prefix[-(k - 1):]
        if ctx not in log_probs:
            continue

        for nxt, lp in log_probs[ctx].items():
            new_cost = cost - lp                  # adding -lp (lp is negative)
            new_prefix = prefix + nxt

            if nxt == END_TOK:
                # strip padding + END_TOK to recover the actual password
                password = new_prefix[k - 1:-1]
                if 1 <= len(password) <= max_length:
                    yield password, -new_cost
                    emitted += 1
                    if emitted >= budget:
                        return
            else:
                actual_len = len(new_prefix) - (k - 1)
                if actual_len < max_length:
                    heapq.heappush(heap, (new_cost, new_prefix))


def enumerate_fixed_length(model, length, budget=1_000_000, beam_size=200_000):
    """
    same as enumerate_candidates but emits ONLY passwords of exactly `length`.
    used by the length experiment: without this the heap fires off short
    high probability completions and never reaches the test length.

    differences from the unbounded version:
    - ignore END_TOK transitions entirely (force the chain to keep going)
    - emit when the unpadded prefix hits `length`
    - bound the heap to `beam_size` to prevent OOM. beyond ~10 char depth
      a 48 char vocab grows the heap by 48x per level, without pruning
      memory explodes. pruning drops the lowest probability prefixes,
      same approximation real OMEN uses
    """
    log_probs, vocab, k = model
    start_ctx = START_TOK * (k - 1)
    heap = [(0.0, start_ctx)]
    emitted = 0

    while heap and emitted < budget:
        cost, prefix = heapq.heappop(heap)
        actual_len = len(prefix) - (k - 1)

        if actual_len == length:
            yield prefix[k - 1:], -cost
            emitted += 1
            if emitted >= budget:
                return
            continue

        ctx = prefix[-(k - 1):]
        if ctx not in log_probs:
            continue

        for nxt, lp in log_probs[ctx].items():
            if nxt == END_TOK:                    # skip, we want fixed length
                continue
            new_cost = cost - lp
            heapq.heappush(heap, (new_cost, prefix + nxt))

        # beam prune: keep only the `beam_size` most probable partial prefixes.
        # heapq.nsmallest pulls the lowest cost (= highest prob) entries from
        # the current heap, rest get discarded
        if len(heap) > beam_size * 2:
            heap = heapq.nsmallest(beam_size, heap)
            heapq.heapify(heap)


# crack a set of md5 hashes.
# hash each candidate ONCE and check against every target simultaneously.
# same trick as script 03

def crack(model, target_hashes, budget=1_000_000, max_length=16):
    """return {target_hash: (plaintext, rank)} for cracked hashes."""
    remaining = dict(target_hashes)               # hash -> label
    cracked = {}
    for rank, (cand, lp) in enumerate(
        enumerate_candidates(model, max_length=max_length, budget=budget), start=1
    ):
        h = hashlib.md5(cand.encode()).hexdigest()
        if h in remaining:
            cracked[h] = (cand, rank, lp)
            print(f"  cracked {remaining[h]:20} -> {cand!r:30}  rank {rank:>8,}  logP={lp:.2f}")
            del remaining[h]
            if not remaining:
                break
    return cracked, rank


# cli
def main():
    parser = argparse.ArgumentParser(description="OMEN-lite markov password cracker")
    parser.add_argument("--k", type=int, default=3, help="n-gram order (default 3)")
    parser.add_argument("--budget", type=int, default=200_000,
                        help="max candidates to generate (default 200k)")
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--emit-wordlist", type=str, default=None,
                        help="write candidates to a file for hashcat instead of cracking")
    args = parser.parse_args()

    print(f"loading corpus from {WORDLIST_PATH}")
    corpus = load_corpus()
    print(f"  {len(corpus):,} training passwords")

    print(f"training {args.k}-gram model")
    t0 = time.perf_counter()
    model = train_markov(corpus, k=args.k)
    print(f"  done in {time.perf_counter() - t0:.2f}s, |vocab|={len(model[1])}")

    if args.emit_wordlist:
        # emit only mode for piping into hashcat
        with open(args.emit_wordlist, "w", encoding="utf-8") as f:
            t0 = time.perf_counter()
            for i, (cand, lp) in enumerate(
                enumerate_candidates(model, args.max_length, args.budget), start=1
            ):
                f.write(cand + "\n")
                if i % 50_000 == 0:
                    print(f"  emitted {i:,} at {i/(time.perf_counter()-t0):,.0f} cand/s")
        print(f"wrote {i:,} candidates to {args.emit_wordlist}")
        print(f"run:  hashcat -a 0 -m 0 hashes.txt {args.emit_wordlist}")
        return

    # built in demo: pick five targets spanning the easy/hard spectrum
    # and see how far the markov model gets with the given budget
    targets_plain = [
        "monkey",            # in training corpus, trivial
        "monkey123",         # not in corpus but high probability shape
        "qwerty2024",        # high probability shape
        "Tr0ub4dor&3",       # xkcd style mangled, low probability
        "correcthorsebatterystaple",  # long human passphrase
    ]
    targets = {hashlib.md5(p.encode()).hexdigest(): p for p in targets_plain}
    print(f"\ncracking {len(targets)} targets with budget={args.budget:,}\n")
    t0 = time.perf_counter()
    cracked, last_rank = crack(model, targets, budget=args.budget,
                                max_length=args.max_length)
    elapsed = time.perf_counter() - t0
    print(f"\n{len(cracked)}/{len(targets)} cracked in {elapsed:.2f}s")
    print(f"effective rate: {last_rank/elapsed:,.0f} candidates/s on this cpu")
    for plain in targets_plain:
        if hashlib.md5(plain.encode()).hexdigest() not in cracked:
            print(f"  NOT CRACKED in budget: {plain!r}")


if __name__ == "__main__":
    main()
