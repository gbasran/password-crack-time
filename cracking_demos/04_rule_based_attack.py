"""
04_rule_based_attack.py
=======================
mode -a 0 with -r rules in hashcat. this is the answer to "but i added a !
and a 1 to the end, surely that helps". it doesnt, because the attacker
applies exactly those mutations programmatically to every word in the list.

hashcat ships rule files (best64.rule, dive.rule, OneRuleToRuleThemAll)
encoding hundreds of these transformations. we reimplement a handful of
the most productive ones below so you can see whats actually happening to
your `P@ssw0rd!`.
"""

import hashlib
import itertools


# the base wordlist (same kind as script 03)
wordlist = ["monkey", "password", "summer", "dragon", "qwerty"]


# the rule engine.
# a rule is just a deterministic function, word -> mutated word. each one
# below maps to a hashcat rule primitive (the comment shows the actual
# hashcat syntax). chaining rules lets the attacker explore the whole
# human "make my password look strong" design space

def r_identity(w):       return w                               # rule  :   (do nothing)
def r_capitalize(w):     return w.capitalize()                  # rule  c   (Monkey)
def r_upper(w):          return w.upper()                       # rule  u   (MONKEY)
def r_reverse(w):        return w[::-1]                         # rule  r   (yeknom)
def r_leet(w):                                                  # rule  sa@ se3 si1 so0 ss$
    table = str.maketrans("aeiosAEIOS", "4310$4310$")
    return w.translate(table)
def r_append_year(w):    return w + "2024"                      # rule  $2 $0 $2 $4
def r_append_bang(w):    return w + "!"                         # rule  $!
def r_append_123(w):     return w + "123"                       # rule  $1 $2 $3

rules = [r_identity, r_capitalize, r_upper, r_reverse,
         r_leet, r_append_year, r_append_bang, r_append_123]


# generate candidates by composing rules.
# real rule sets chain up to 3-4 rules. we chain triples here so the demo
# can actually crack a leet+capitalize+append target like 'M0nk3y!'.
# itertools.product(rules, repeat=3) is every ordered triple

def candidates(word):
    """yield every mutation of `word` from chaining up to 3 rules."""
    for rule_chain in itertools.product(rules, repeat=3):
        out = word
        for rule in rule_chain:
            out = rule(out)
        yield out


# the attack against a target the user thought was "strong".
# the victim picked `M0nk3y!` which has:
#   - a capital letter
#   - a number
#   - a symbol
# it passes every checkbox style strength meter. it also falls in ~100 guesses

target_plain = "M0nk3y!"
target_hash = hashlib.md5(target_plain.encode()).hexdigest()
print(f"target = {target_plain!r}  (md5 {target_hash[:16]}...)\n")

guesses = 0
for base in wordlist:
    for candidate in candidates(base):
        guesses += 1
        if hashlib.md5(candidate.encode()).hexdigest() == target_hash:
            print(f"CRACKED in {guesses} guesses: base={base!r} -> {candidate!r}")
            raise SystemExit


# why rules destroy "complexity only" passwords.
# a standard hashcat rule file (best64) has 64 rules. combined with rockyous
# 14M words and 2-rule chaining thats 14M * 64 * 64 ~= 57 billion candidates,
# which one gpu finishes in about 6 seconds against md5. the takeaway for
# the paper: complexity REQUIREMENTS produce passwords that look strong to
# humans and are trivial for rule engines. length and unpredictability are
# the only things that actually buy time
