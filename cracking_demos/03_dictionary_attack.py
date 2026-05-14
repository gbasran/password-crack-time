"""
03_dictionary_attack.py
=======================
mode -a 0 in hashcat. the single most effective attack in the real world.
roughly 80% of rockyou hashes fall to a plain wordlist with zero mutations.

this is the attack that the `in_breach_list` and `has_common_word` features
in the notebook are flagging. if either is true you are dead before the
brute force loop ever starts.
"""

import hashlib
import time


# the wordlist.
# real attackers use rockyou.txt (14M entries), seclists, or
# haveibeenpwneds 1B entry list. we use a tiny embedded list so the script
# is self contained and the prose lines up with the wordlist used in the
# notebook

wordlist = [
    "password", "123456", "qwerty", "letmein", "dragon",
    "monkey", "football", "iloveyou", "admin", "welcome",
    "sunshine", "princess", "shadow", "master", "trustno1",
    # the actual target is at index 16 to show the attack DOES have to scan
    "hunter2", "summer2024", "correcthorse",
]


# the leaked hash dump.
# in a real breach this is a file with millions of (username, hash) rows.
# here we simulate three stolen users. the attacker wants to crack as many
# as possible

stolen_dump = {
    "alice":   hashlib.md5(b"monkey").hexdigest(),       # very weak
    "bob":     hashlib.md5(b"summer2024").hexdigest(),   # weak ish (in our list)
    "carol":   hashlib.md5(b"Tr0ub4dor&3").hexdigest(),  # NOT in this wordlist
}

# build a reverse lookup so we can spot a crack in O(1) per guess. key
# insight: the attacker hashes EACH candidate once and checks it against
# every stolen user simultaneously. this is what makes dictionary attacks
# so cheap against large dumps
hash_to_user = {h: u for u, h in stolen_dump.items()}


# the attack loop
print(f"attacking {len(stolen_dump)} stolen hashes with a {len(wordlist)} word list\n")
cracked = {}
start = time.perf_counter()

for guess_index, word in enumerate(wordlist, start=1):
    h = hashlib.md5(word.encode()).hexdigest()      # hash the candidate
    if h in hash_to_user:                           # one O(1) dict lookup against ALL users
        user = hash_to_user[h]
        cracked[user] = word
        print(f"  guess #{guess_index:>3}: {word!r:15} -> cracked {user}")

elapsed = time.perf_counter() - start


# report
print(f"\ncracked {len(cracked)}/{len(stolen_dump)} accounts in {elapsed*1000:.2f}ms")
for user in stolen_dump:
    if user not in cracked:
        print(f"  {user}: not in wordlist (would need rules or brute force)")


# why this is the attack that matters.
# the crack time formula in the notebook assigns dictionary word passwords
# a cost of ~100k * 200 guesses (wordlist size * rule multiplier) rather
# than the full charset^length. even at single cpu speeds that finishes in
# well under a second. on a gpu rig the entire rockyou + best64 ruleset
# finishes in roughly a minute. this is why the t-test on `has_common_word`
# produces a p-value basically indistinguishable from zero
