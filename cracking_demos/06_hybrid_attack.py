"""
06_hybrid_attack.py
===================
mode -a 6 / -a 7 in hashcat. dictionary words on one side, brute forced
suffix or prefix on the other. this is the workflow real crackers run
AFTER the plain dictionary attack exhausts itself, before falling back
to pure brute force. its the attack that turns `summer` + `2024!` into a
foregone conclusion.
"""

import hashlib
import itertools
import string
import time


# two halves of the search.
# left:  a wordlist (could be rockyou, here a tiny demo list)
# right: every short suffix over a small alphabet. real engagements use
#        wordlist + mask, e.g. -a 6 wordlist.txt ?d?d?d?d for "word + 4 digits"

wordlist = ["summer", "winter", "spring", "monkey", "dragon",
            "letmein", "qwerty", "admin", "hunter"]

suffix_alphabet = string.digits + "!@#"        # what humans actually append
suffix_lengths  = [3, 4, 5]                    # try 3, 4, 5 char suffixes


# candidate generator.
# for each base word, for each suffix length, for each suffix combination
# yield base + suffix. we also try the suffix on the FRONT because some
# users prepend (`2024Summer`). doubles work for small gain in real numbers
# but mirrors what hashcat does with -a 7

def hybrid_candidates():
    for base in wordlist:
        for L in suffix_lengths:
            for combo in itertools.product(suffix_alphabet, repeat=L):
                suffix = "".join(combo)
                yield base + suffix           # append mode  (hashcat -a 6)
                yield suffix + base           # prepend mode (hashcat -a 7)


# run it against a target the user thought was "creative"
secret = "summer2024!"                        # base english word + year + bang
target_hash = hashlib.md5(secret.encode()).hexdigest()
print(f"target plaintext: {secret!r}  (md5 {target_hash[:16]}...)\n")

start = time.perf_counter()
guesses = 0
for candidate in hybrid_candidates():
    guesses += 1
    if hashlib.md5(candidate.encode()).hexdigest() == target_hash:
        elapsed = time.perf_counter() - start
        print(f"CRACKED: {candidate!r}")
        print(f"  guesses : {guesses:,}")
        print(f"  time    : {elapsed:.2f}s on one cpu core")
        print(f"  rate    : {guesses/elapsed:,.0f} H/s")
        break


# how this maps back to the regression.
# the notebooks `has_common_word` feature flags exactly these passwords.
# the crack time formula multiplies the wordlist size by a "mutation
# factor" meant to cover the suffix space, this script is what that
# mutation factor is approximating. for the paper a useful sanity check
# graph would be: plot predicted crack time vs the number of guesses this
# script actually takes across a few hundred sample passwords. tight
# correlation = the mutation factor heuristic is right. loose correlation
# = the heuristic underestimates the real attacker and you should bump
# the multiplier
