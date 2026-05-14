"""
01_hashing_basics.py
====================
sets up the threat model the paper uses.

what this shows:
1. why servers store hashes instead of plaintext passwords
2. why the choice of hash function is the single biggest defensive lever
3. how many md5 hashes/sec your cpu can actually do, vs bcrypt. the gap is
   roughly 8 orders of magnitude

run it:
    python3 01_hashing_basics.py
"""

import hashlib   # stdlib, gives us md5, sha1, sha256 etc
import time      # for wall clock timing
import os        # for random bytes (we only use it for the benchmark inputs, not a real salt)


# what a hash actually is. deterministic one way function. same input always
# makes the same output, but you cant reverse it. below: feed three passwords
# through md5 and print the digests
print("--- 1. hashes are deterministic, one way, fixed length ---")
for pw in ["monkey", "monkey", "Monkey"]:
    # .encode() turns the str into bytes (hash libs work on bytes)
    # .hexdigest() returns the digest as a printable hex string
    digest = hashlib.md5(pw.encode()).hexdigest()
    print(f"  md5({pw!r:10}) = {digest}")
# the two 'monkey' lines match (deterministic). changing one letter to a
# capital scrambles the entire output (the avalanche property)


# why an attacker with a leaked hash dump still has work to do.
# if a database leaks the attacker has rows like:
#     username, md5_hash
#     alice,    8c0a59049f30075d3dc4dc1d3f5da82d
# they cant just submit that hash to the login form because the login form
# hashes whatever you type and compares. so they have to GUESS plaintexts,
# hash each guess, look for a match. that guess and hash loop is what the
# rest of these scripts simulate


# benchmark: how fast can a single python thread do md5
# we hash N random byte strings and divide by elapsed time. pythons hashlib
# is actually a c extension so this is a reasonable lower bound. a real gpu
# attacker is roughly 1,000x to 10,000x faster than one cpu core for md5

print("\n--- 2. md5 throughput on this machine ---")
N = 500_000                                       # how many to hash
candidates = [os.urandom(8) for _ in range(N)]    # N random 8 byte inputs

start = time.perf_counter()                       # high resolution timer
for c in candidates:                              # tight loop, hash each one
    hashlib.md5(c).digest()                       # .digest() = raw bytes, faster than .hexdigest()
elapsed = time.perf_counter() - start

md5_rate = N / elapsed
print(f"  {N:,} md5 hashes in {elapsed:.2f}s  =>  {md5_rate:,.0f} H/s on one cpu core")
print(f"  a $2,000 gpu does ~10,000,000,000 H/s ({10_000_000_000 / md5_rate:,.0f}x faster)")


# same benchmark with bcrypt (the right answer).
# bcrypt is deliberately slow. the cost factor (rounds) lets the defender
# dial in how many ms each hash takes. point is to make the attackers loop
# economically painful

try:
    import bcrypt                          # pip install bcrypt
    print("\n--- 3. bcrypt throughput at cost=12 ---")
    M = 20                                 # only 20, bcrypt is SLOW on purpose
    salt = bcrypt.gensalt(rounds=12)       # cost 12 is the owasp 2025 minimum
    start = time.perf_counter()
    for _ in range(M):
        bcrypt.hashpw(b"monkey", salt)     # one full bcrypt
    elapsed = time.perf_counter() - start
    bcrypt_rate = M / elapsed
    print(f"  {M} bcrypt hashes in {elapsed:.2f}s  =>  {bcrypt_rate:,.1f} H/s")
    print(f"  bcrypt is {md5_rate / bcrypt_rate:,.0f}x slower than md5 on the SAME cpu")
    print(f"  the attackers gpu advantage mostly disappears, bcrypt is memory hard")
except ImportError:
    print("\n(install 'bcrypt' to see the comparison: pip install bcrypt)")


# takeaway for the paper.
# the crack time model in the notebook assumes 10^10 guesses/sec. thats
# realistic for md5/sha1 which lazy or legacy systems still use. against
# bcrypt at cost 12 the same attacker drops to ~10 guesses/sec per core,
# roughly a billion times slower. every estimate downstream scales linearly
# with that rate, so the servers choice of hash is worth ~9 orders of
# magnitude of password strength
