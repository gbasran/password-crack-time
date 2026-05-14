"""
02_brute_force.py
=================
mode -a 3 in hashcat. dumbest possible attack: try every string of every
length until one of them hashes to the target.

this is what the notebooks `charset_size ** length` formula is modeling.
the script shows the actual loop behind that formula so you can see why
each extra character multiplies the cost by ~|charset|.
"""

import hashlib
import itertools     # cartesian products
import string        # pre built lowercase, digits, ascii etc
import time


# target setup. we hash a known password so we know when weve cracked it.
# in a real engagement the attacker has stolen `target_hash` from a breach
# and doesnt know `secret`. here we pick `secret` ourselves so the script
# finishes in a reasonable time

secret = "cat7"                                              # what well "recover"
target_hash = hashlib.md5(secret.encode()).hexdigest()        # what the attacker sees
print(f"target hash (md5): {target_hash}")


# the charset and length range to search. this is the attackers only real
# knob in pure brute force. bigger charset or longer max length explodes
# the space exponentially. we use lowercase + digits (36 chars) so the
# demo finishes in seconds

charset = string.ascii_lowercase + string.digits             # 'abc...xyz0123456789' = 36 chars
max_len = 5                                                  # lengths 1..5
print(f"charset size = {len(charset)}, trying lengths 1..{max_len}")
total_space = sum(len(charset)**L for L in range(1, max_len + 1))
print(f"worst case = {total_space:,} guesses\n")


# the brute force loop.
# itertools.product(charset, repeat=L) yields every L length tuple of
# characters. we join the tuple into a string, hash it, compare. this is
# almost line for line what hashcat does internally, hashcat just runs it
# on the gpu in parallel across millions of candidates at once

start = time.perf_counter()
guesses = 0

for length in range(1, max_len + 1):                          # outer loop, increasing length
    for combo in itertools.product(charset, repeat=length):   # all length^|charset| tuples
        candidate = "".join(combo)                            # tuple to str, ('c','a','t','7') -> 'cat7'
        guesses += 1
        if hashlib.md5(candidate.encode()).hexdigest() == target_hash:
            elapsed = time.perf_counter() - start
            print(f"CRACKED: {candidate!r} after {guesses:,} guesses in {elapsed:.2f}s")
            print(f"effective rate: {guesses/elapsed:,.0f} H/s on this cpu")
            raise SystemExit                                  # stop on first match


# why this scales so badly.
# going from a 4 char password to a 12 char password with the same 36 char
# alphabet multiplies the worst case work by 36^8 = ~2.8 trillion. thats
# why the regression has `password_length` as the biggest positive
# coefficient, every extra character is exponential protection against
# the brute force fallback
