"""
08_length_experiment.py
=======================
the experiment that tests the papers central claim:

    "length absolutely dominates everything else."

we test it against a smart attacker (the markov cracker from script 07),
not just brute force. two flavours of target passwords at each length:

    - random_uniform : true crypto random over [a-z0-9]
    - human_shaped   : sampled from the SAME distribution the attacker
                       trained on. worst case for the defender: the
                       attacker has a perfect model of how you pick

for each (length, flavour) cell we generate N targets, run the cracker
against them with a fixed candidate budget, record:
    - cracked_count  : how many of the N fell within budget
    - median_rank    : median guess number to crack (for the cracked ones)
    - median_log_p   : log P(target) under the model, ANALYTICAL prediction
                       of crack rank (covers cells the enumerator times out on)

the hypothesis the paper makes ("length dominates") predicts cracked_count
drops to ~0 as length grows, in BOTH flavours.

a more refined hypothesis (what we actually see) predicts that length crushes
random_uniform exponentially but only slows down human_shaped linearly. if
thats what we observe the paper should say "length only buys strength to the
extent it adds entropy the attacker hasnt modeled".

usage:
    python3 08_length_experiment.py --n-per-cell 20 --budget 200000
"""

import argparse
import csv
import hashlib
import math
import os
import random
import secrets
import string
import sys
import time

# pull in the markov module from script 07. importing a numbered module name
# needs importlib because '07_...' isnt a valid python identifier
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("markov07",
                                              os.path.join(HERE, "07_markov_cracker.py"))
markov = importlib.util.module_from_spec(spec)
sys.modules["markov07"] = markov
spec.loader.exec_module(markov)


# target generation.
# random_uniform: secrets.choice over a fixed alphabet, true randomness.
# this is the cryptographic best case, every position is independent and
# uniformly distributed

ALPHABET = string.ascii_lowercase + string.digits

def gen_random_uniform(length, n):
    return [
        "".join(secrets.choice(ALPHABET) for _ in range(length))
        for _ in range(n)
    ]


# human_shaped: sample from the markov model itself. we walk the chain
# forward, at each step sampling the next character proportional to its
# learned probability under the current context. to force a target length
# we discard END_TOK proposals until weve hit the desired length, then
# stop. this biases the distribution slightly but keeps the per character
# transitions in distribution

def gen_human_shaped(model, length, n, rng):
    log_probs, vocab, k = model
    out = []
    while len(out) < n:
        prefix = markov.START_TOK * (k - 1)
        pw = ""
        for _ in range(length * 10):                # safety bound on backtracks
            ctx = prefix[-(k - 1):]
            if ctx not in log_probs:
                break
            # sample proportional to probability, skip END_TOK until we
            # hit the target length
            chars = list(log_probs[ctx].keys())
            weights = [pow(2.718281828, log_probs[ctx][c]) for c in chars]
            if len(pw) < length:
                # zero out END_TOK so we dont terminate early
                weights = [w if c != markov.END_TOK else 0.0 for c, w in zip(chars, weights)]
            total = sum(weights)
            if total == 0:
                break
            weights = [w / total for w in weights]
            c = rng.choices(chars, weights=weights, k=1)[0]
            if c == markov.END_TOK:
                break
            pw += c
            prefix += c
            if len(pw) == length:
                break
        if len(pw) == length:
            out.append(pw)
    return out


# run a single (length, flavour) cell.
# we hash all N targets, then crack with a single shared budget. the
# cracker emits candidates once, each candidate is checked against every
# target simultaneously (same many targets one pass trick from script 03)

def run_cell(model, targets, budget, length):
    """use the FIXED LENGTH enumerator so the budget is spent at the test
    length. without this the heap floods with short high probability
    completions and never reaches length 8+."""
    hash_to_plain = {hashlib.md5(p.encode()).hexdigest(): p for p in targets}
    remaining = dict(hash_to_plain)
    cracked_ranks = []

    for rank, (cand, _) in enumerate(
        markov.enumerate_fixed_length(model, length=length, budget=budget),
        start=1,
    ):
        h = hashlib.md5(cand.encode()).hexdigest()
        if h in remaining:
            cracked_ranks.append(rank)
            del remaining[h]
            if not remaining:
                break

    return cracked_ranks


def analytical_score(model, targets):
    """compute log P(target) under the model for each target.

    this is the models theoretical prediction of crack rank: if you
    enumerated candidates in perfect descending probability order, the
    expected rank of a target with probability P is approximately 1/P.
    so log_p = -k means we expect roughly e^k = 10^(k/2.3) guesses to
    crack that target.

    much faster than actually running the enumerator (one pass over the
    targets characters, not a beam search), and it gives full coverage
    at every length where the enumerator would time out.
    """
    scores = []
    for pw in targets:
        lp = markov.score(model, pw)
        # with the smoothed score() function (uniform backoff for unseen
        # contexts), -inf should no longer occur. keep a safety floor just
        # in case
        if lp == float("-inf"):
            lp = -100.0
        scores.append(lp)
    return scores


# main experiment loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-cell", type=int, default=20,
                        help="targets per (length, flavour) cell")
    parser.add_argument("--budget", type=int, default=200_000,
                        help="candidate budget per cell (0 = analytical only)")
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[6, 8, 10, 12, 14, 16])
    parser.add_argument("--out", type=str,
                        default="cracking_demos/results_length_experiment.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("loading corpus + training markov model")
    corpus = markov.load_corpus()
    model = markov.train_markov(corpus, k=3)
    print(f"  trained on {len(corpus):,} passwords, |vocab|={len(model[1])}")

    rng = random.Random(args.seed)
    rows = []
    print(f"\nrunning experiment: {len(args.lengths)} lengths "
          f"x 2 flavours x {args.n_per_cell} targets")
    print(f"  analytical (log P under model) at every cell, fast, full coverage")
    print(f"  empirical  (actual crack within budget) at every cell, budget={args.budget:,}\n")
    print(f"{'length':>6} {'flavour':>14} "
          f"{'median_logP':>12} {'predicted_rank':>16} "
          f"{'cracked':>8} {'time_s':>8}")

    for length in args.lengths:
        for flavour, gen in [
            ("random_uniform", lambda L=length: gen_random_uniform(L, args.n_per_cell)),
            ("human_shaped",  lambda L=length: gen_human_shaped(model, L, args.n_per_cell, rng)),
        ]:
            targets = gen()

            # 1) analytical: log P(target) under model. log_p = -k means
            #    expected rank ~ e^k. this is what crack time would be
            #    under an ideal probability ordered enumerator
            scores = analytical_score(model, targets)
            median_lp = sorted(scores)[len(scores) // 2]
            # convert to base 10 log of predicted rank for readability
            predicted_rank_log10 = -median_lp / math.log(10)

            # 2) empirical: actually run the enumerator with the budget.
            #    caps at args.budget. only meaningful at small lengths,
            #    at length 12+ even the analytical predicts ranks way
            #    beyond any feasible python budget, so 0 is expected.
            #    setting --budget 0 skips the empirical pass
            t0 = time.perf_counter()
            if args.budget > 0:
                ranks = run_cell(model, targets, args.budget, length=length)
            else:
                ranks = []
            elapsed = time.perf_counter() - t0
            n_cracked = len(ranks)

            print(f"{length:>6} {flavour:>14} "
                  f"{median_lp:>12.2f} 10^{predicted_rank_log10:>13.1f} "
                  f"{n_cracked:>4}/{args.n_per_cell:<3} {elapsed:>7.1f}s")
            rows.append({
                "length": length,
                "flavour": flavour,
                "median_logP": round(median_lp, 3),
                "predicted_log10_rank": round(predicted_rank_log10, 2),
                "n_cracked": n_cracked,
                "n_total": args.n_per_cell,
                "elapsed_s": round(elapsed, 2),
                "budget": args.budget,
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")
    print("\ninterpretation:")
    print("  cracked drops to 0 in both flavours ........ papers length claim holds")
    print("  cracked drops in random_uniform only ........ length only helps for ENTROPY,")
    print("                                                  not just CHARACTERS")


if __name__ == "__main__":
    main()
