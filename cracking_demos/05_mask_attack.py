"""
05_mask_attack.py
=================
mode -a 3 with a mask in hashcat. brute force, but the attacker has
guessed the SHAPE of the password and only explores plaintexts matching it.

example mask: ?u?l?l?l?l?d?d?d?d
   ?u = one uppercase letter
   ?l = one lowercase letter
   ?d = one digit
matches things like "Apple1990", one cap, four lower, four digits. a LOT of
corporate passwords look exactly like this because of password policy plus
human muscle memory.
"""

import hashlib
import itertools
import string
import time


# the mask alphabet. each token in a mask maps to a character class.
# hashcat ships ?l ?u ?d ?s ?a (all) plus user defined ?1..?4. its just
# a dict

CLASSES = {
    "?l": string.ascii_lowercase,    # 26
    "?u": string.ascii_uppercase,    # 26
    "?d": string.digits,             # 10
    "?s": "!@#$%^&*",                # custom symbol set, real hashcat uses 33
}


# mask parser. walks the mask string and emits the list of charsets it
# represents. only has to know that '?' starts a 2 char token

def parse_mask(mask):
    """'?u?l?l?d' -> [uppercase, lowercase, lowercase, digits]"""
    positions = []
    i = 0
    while i < len(mask):
        if mask[i] == "?":               # placeholder token
            token = mask[i:i+2]          # e.g. '?u'
            positions.append(CLASSES[token])
            i += 2
        else:                            # literal character, only matches itself
            positions.append(mask[i])
            i += 1
    return positions


# the attack.
# itertools.product(*positions) takes the cartesian product across all
# positions. compared to script 02s blind brute force this is dramatically
# smaller because the attacker cut out every length and every shape that
# doesnt match the mask

def crack_with_mask(target_hash, mask):
    positions = parse_mask(mask)
    space = 1
    for p in positions:
        space *= len(p)
    print(f"mask {mask}  ->  {space:,} candidates (vs blind brute force much larger)")

    start = time.perf_counter()
    for tup in itertools.product(*positions):
        candidate = "".join(tup)
        if hashlib.md5(candidate.encode()).hexdigest() == target_hash:
            elapsed = time.perf_counter() - start
            print(f"  CRACKED: {candidate!r} in {elapsed:.2f}s")
            return candidate
    print("  not in mask space (would need a wider mask)")
    return None


# demo against two realistic policy compliant passwords. both meet a
# typical "1 upper, 1 lower, 1 digit, 8+ chars" policy. both fall to a
# mask that encodes that exact shape

if __name__ == "__main__":
    # keep the masks short so the demo finishes in a few seconds on cpu.
    # on a gpu the same masks at length 9-10 (Apple1990, Spring2024) crack
    # in well under a minute, shape of the search is identical
    for plain, mask in [
        ("App42",  "?u?l?l?d?d"),       # 26*26*26*10*10 = 1.76M candidates
        ("Spr24",  "?u?l?l?d?d"),       # same shape, different word
    ]:
        target = hashlib.md5(plain.encode()).hexdigest()
        print(f"\ntarget plaintext (hidden from attacker): {plain!r}")
        crack_with_mask(target, mask)


# why mask attacks matter in the paper.
# the notebooks crack time formula treats anything not in the wordlist as
# uniform `charset^length` brute force. that OVERSTATES strength because
# real attackers prune the search space with masks derived from leaked
# data (pack / prince generates masks ranked by frequency in rockyou). a
# 9 char mixed password is nominally ~10^17 guesses, but the top 1000
# masks cover roughly 80% of real human passwords, dropping the effective
# space by 3-4 orders of magnitude. a natural future work bullet for the
# paper would be replacing the brute force fallback with a mask frequency
# lookup
