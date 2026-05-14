# how modern password cracking actually works

companion code for the paper. six self contained python scripts that reimplement what hashcat / john the ripper / commercial cred stuffing rigs actually do, plus a smart probabilistic cracker (markov + neural) that tests whether the length argument from the main notebook holds up against a smarter attacker. every script cracks a target it generated itself so theres no real breach data flying around and nothing leaves the laptop. the scripts line up with the threat model from the notebook (~10^10 guesses/sec on a fast hash, dictionary + rules + brute force) so you can cite this stuff directly from section 2 of the writeup.

| # | file | hashcat mode | what it cracks |
|---|---|---|---|
| 1 | `01_hashing_basics.py` | n/a | benchmarks md5 vs bcrypt, sets up the 10^10 figure |
| 2 | `02_brute_force.py` | `-a 3` | `cat7` in 142k guesses |
| 3 | `03_dictionary_attack.py` | `-a 0` | 2 of 3 stolen hashes from an 18 word list |
| 4 | `04_rule_based_attack.py` | `-a 0 -r` | `M0nk3y!` in 103 guesses |
| 5 | `05_mask_attack.py` | `-a 3` mask | `App42` in 0.03s out of 1.76M candidates |
| 6 | `06_hybrid_attack.py` | `-a 6 / -a 7` | `summer2024!` (dict word + brute forced suffix) |

run any of them with `python3 NN_name.py`. no external data, no gpu. takes around 5 seconds end to end on a normal laptop. part 2 below covers the smart cracker (scripts 07-09) which is the actual research piece.

---

## 1 · hashing basics (`01_hashing_basics.py`)

this sets up why theres anything to crack in the first place.

```python
import hashlib, time, os
```
hashlib is pythons stdlib wrapper around openssl, gives us md5, sha1, sha256 etc. `time.perf_counter()` is the highest resolution wall clock available. `os.urandom` produces cryptographically random bytes (we only use it to make candidate inputs for the benchmark, not as a real salt).

```python
for pw in ["monkey", "monkey", "Monkey"]:
    digest = hashlib.md5(pw.encode()).hexdigest()
```
three loop iterations prove three properties at once. the two `monkey` lines produce identical digests (determinism, same input always hashes to the same output). changing one letter to a capital scrambles the entire output (avalanche, a 1 bit input change flips around 50% of the output bits). `.encode()` turns the str into bytes because hash functions operate on byte sequences. `.hexdigest()` formats the 16 byte raw digest as printable hex.

```python
N = 500_000
candidates = [os.urandom(8) for _ in range(N)]

start = time.perf_counter()
for c in candidates:
    hashlib.md5(c).digest()
elapsed = time.perf_counter() - start

md5_rate = N / elapsed
```
this is the benchmark. we pre generate the inputs so the loop is just "hash, throw away" with no overhead from generation. we call `.digest()` instead of `.hexdigest()` because the hex conversion is pure python overhead the attacker doesnt pay either. on a 2024 laptop you see around 1-2M H/s from one cpu core. a four gpu rig pushes ~40 billion H/s, so a single attacker box is roughly 4-5 orders of magnitude faster than this measurement, AND the attacker can rent thousands of boxes from aws for a dollar an hour each.

```python
salt = bcrypt.gensalt(rounds=12)
start = time.perf_counter()
for _ in range(M):
    bcrypt.hashpw(b"monkey", salt)
```
the bcrypt half. `rounds=12` is the owasp 2025 minimum. bcrypt is deliberately slow, `2^12 = 4096` rounds of an expensive blowfish derived key schedule per call. youll measure roughly 10 H/s on the same cpu that did 1.7M md5/s. the ratio (~10^5) is what makes bcrypt actually defensive, it removes the gpus advantage because each bcrypt invocation is memory bound not compute bound, and gpus have very little fast memory per core.

bottom line for the paper: every crack time number downstream assumes the server picked md5 or sha1. if the server picked bcrypt at cost 12 you divide attacker speed by ~10^9, so every prediction in the regression model becomes roughly a billion times longer in real wall clock time. this is the single biggest defensive lever and its not in the writeup yet.

---

## 2 · brute force (`02_brute_force.py`)

the dumb baseline. try every possible string in order. this is literally what the notebooks `charset_size ** length` formula is counting.

```python
secret = "cat7"
target_hash = hashlib.md5(secret.encode()).hexdigest()
```
we hash a known plaintext so the script has a target to recover and a way to know when it succeeded. in a real engagement the attacker has the `target_hash` from a leaked dump and doesnt know `secret`.

```python
charset = string.ascii_lowercase + string.digits   # 36 chars
max_len = 5
```
the attackers only knob in pure brute force. adding uppercase + symbols here would balloon the charset from 36 to ~95 and multiply the search space by `(95/36)^L`. this is exactly the `charset_size` feature in the dataset.

```python
for length in range(1, max_len + 1):
    for combo in itertools.product(charset, repeat=length):
        candidate = "".join(combo)
        guesses += 1
        if hashlib.md5(candidate.encode()).hexdigest() == target_hash:
            ...
```
three nested ideas. outer loop over length so we start short, shorter is cheaper to enumerate. `itertools.product(charset, repeat=length)` gives every ordered tuple of `length` characters drawn from `charset`, so for `length=4` and `|charset|=36` thats `36^4 = 1,679,616` tuples. `"".join(combo)` because product yields tuples like `('c','a','t','7')` and we want strings.

this is why `password_length` is the biggest positive coefficient in the regression. the inner loop runs `|charset|^length` times. every extra character multiplies the worst case work by `|charset|`. going from 8 to 12 chars with a 95 char alphabet is a `95^4 ≈ 81 million` x cost increase. EXPONENT always wins.

---

## 3 · dictionary attack (`03_dictionary_attack.py`)

the attack that cracks 80%+ of real world passwords. ignore brute force, try the words humans actually pick first.

```python
wordlist = ["password", "123456", ...]
```
a toy 18 word list standing in for rockyou.txt (14M entries), the haveibeenpwned password list (1B entries), or any of the seclists aggregates. the attack code is the same, only the wordlist scales.

```python
stolen_dump = {
    "alice":   hashlib.md5(b"monkey").hexdigest(),
    "bob":     hashlib.md5(b"summer2024").hexdigest(),
    "carol":   hashlib.md5(b"Tr0ub4dor&3").hexdigest(),
}
hash_to_user = {h: u for u, h in stolen_dump.items()}
```
two design choices that matter here. first, a dump is many hashes not one. real breaches hand the attacker millions of (user, hash) rows. the economics that follow only work because of this. second, the reverse lookup `hash_to_user` is the entire trick. the attacker hashes each candidate ONCE and checks it against every stolen hash simultaneously with a single dict lookup. so one pass over rockyou.txt cracks across every user whose password is in rockyou, no per user cost.

```python
for guess_index, word in enumerate(wordlist, start=1):
    h = hashlib.md5(word.encode()).hexdigest()
    if h in hash_to_user:
        cracked[hash_to_user[h]] = word
```
the whole loop. one hash, one dict lookup per word. on a gpu rig that loop runs at 10^10/sec, so a 14M entry rockyou exhausts in 1.4 milliseconds.

this is why `has_common_word` and `in_breach_list` dominate the regression. this loop runs BEFORE anything else, so if your password lives in either list your `log_crack_time` is basically negative infinity no matter what length or complexity you have. its exactly the pattern the t-test detects.

---

## 4 · rule based attack (`04_rule_based_attack.py`)

the answer to "but i added a `!` and a `1`". this is the most important script for the paper because its where the P@ssw0rd! argument actually gets killed.

```python
def r_leet(w):
    table = str.maketrans("aeiosAEIOS", "4310$4310$")
    return w.translate(table)
def r_append_year(w):    return w + "2024"
def r_append_bang(w):    return w + "!"
def r_append_123(w):     return w + "123"
def r_capitalize(w):     return w.capitalize()
```
each function is one hashcat rule primitive. the real hashcat rule syntax for the leet rule is `sa@ se3 si1 so0 ss$` (single character substitutions). `str.maketrans` builds a translation table and `.translate()` applies it in one c call. hashcat ships rule files with 64 (best64), 25k (dive), or 51k (OneRuleToRuleThemAll) primitives. we use 8 for clarity.

```python
def candidates(word):
    for rule_chain in itertools.product(rules, repeat=3):
        out = word
        for rule in rule_chain:
            out = rule(out)
        yield out
```
the chaining step. `product(rules, repeat=3)` is every ordered triple of rules, so `8^3 = 512` chains per base word. apply them in order and yield. so `monkey` becomes 512 variants including `M0nk3y!`, `Monkey123`, `YEKNOM2024`, `!monkeymonkey` and so on.

hashcat rules are interpreted strings rather than python functions but the semantics are exactly this. with 5 base words and 512 chains thats 2560 candidates total. cracks `M0nk3y!` at guess 103.

the "complexity is good" intuition is that an 8 char password with one cap, one digit, one symbol is stronger than a 12 char all lowercase word. the rule engine treats them identically, both are "base word + 1-2 known transformations". the thing rules cant do cheaply is increase the unpredictable LENGTH of the base.

---

## 5 · mask attack (`05_mask_attack.py`)

smart brute force. instead of enumerating every string of every length, the attacker assumes the password follows a known shape.

```python
CLASSES = {
    "?l": string.ascii_lowercase,
    "?u": string.ascii_uppercase,
    "?d": string.digits,
    "?s": "!@#$%^&*",
}
```
the character classes. hashcat ships `?l ?u ?d ?s ?a` plus user definable `?1..?4`. nothing magic, just named subsets of ascii.

```python
def parse_mask(mask):
    positions = []
    i = 0
    while i < len(mask):
        if mask[i] == "?":
            token = mask[i:i+2]
            positions.append(CLASSES[token])
            i += 2
        else:
            positions.append(mask[i])
            i += 1
    return positions
```
the parser walks the mask string, treats `?x` as one token (looked up in CLASSES) and any other character as a literal that only matches itself. so mask `Pass?d?d?d?d` parses to `[P, a, s, s, digits, digits, digits, digits]` and only enumerates `Pass0000`...`Pass9999`, 10k candidates instead of `95^8 = 6.6 quadrillion`.

```python
for tup in itertools.product(*positions):
    candidate = "".join(tup)
    if hashlib.md5(candidate.encode()).hexdigest() == target_hash:
        ...
```
`*positions` unpacks the parsed list as separate args to `product` so each position contributes its own alphabet. for a 5 position mask `?u?l?l?d?d` thats `26 * 26 * 26 * 10 * 10 = 1.76M` candidates, small enough that the demo cracks it on cpu in under a second.

real world impact: tools like pack and prince take a leaked dataset (rockyou, comb) and rank masks by frequency. the top 1000 masks cover roughly 80% of real human passwords, so even when a password isnt in the wordlist the attacker doesnt pay full `charset^length`, they pay `top_mask_count * average_mask_size` which is 3-4 orders of magnitude cheaper. this is a known gap in the notebooks crack time formula and a natural future work bullet.

---

## 6 · hybrid attack (`06_hybrid_attack.py`)

what actually happens after the pure dictionary attack fails. take every word in the list, glue every short brute forced string onto it.

```python
suffix_alphabet = string.digits + "!@#"
suffix_lengths  = [3, 4, 5]
```
the "what humans append" alphabet. almost every real world appendix is digits + a handful of symbols (`!@#$`, occasionally `.`). 3-5 chars covers years (`2024`), birthdays, the `123!` meme.

```python
def hybrid_candidates():
    for base in wordlist:
        for L in suffix_lengths:
            for combo in itertools.product(suffix_alphabet, repeat=L):
                suffix = "".join(combo)
                yield base + suffix       # hashcat -a 6
                yield suffix + base       # hashcat -a 7
```
triple loop. for each base, for each suffix length, generate every suffix and yield both `base+suffix` and `suffix+base`. cardinality is `|words| * Σ |suffix_alphabet|^L * 2`. for the demo: `9 * (13^3 + 13^4 + 13^5) * 2 ≈ 7M`. python finishes in a few seconds at ~1.4M H/s.

`summer2024!` falls out of this in 176k guesses. note the structure: the attacker didnt have to know the user picked `summer`, only that `[word][digits+symbols]` is one of the most common patterns in every leaked dataset.

the notebook estimates dictionary + mutation crack time as `100k * 200` guesses (wordlist size * mutation multiplier). the 200 multiplier is meant to cover exactly this suffix space. one sanity check the paper could add: run this script against 500 sampled passwords with `has_common_word=1`, record guess counts, plot vs the models predicted guess count. if the trendline slope is far from 1.0 the multiplier is miscalibrated.

---

# part 2 — smart probabilistic crackers + the length experiment

scripts 01-06 are the standard hashcat attack modes, theyre the THREAT MODEL the paper is defending against. part 2 is the research contribution: a smart attacker built from scratch, used to test the papers main claim.

| # | file | what it is |
|---|---|---|
| 7 | `07_markov_cracker.py` | OMEN-lite, 3-gram markov + ordered enumeration |
| 8 | `08_length_experiment.py` | tests the length claim against the smart attacker |
| 9 | `09_neural_cracker_colab.py` + `.ipynb` | char level transformer for colab |
|   | `run_with_hashcat.sh` | pipes python generated wordlists to hashcat |

## 7 · OMEN-lite markov cracker (`07_markov_cracker.py`)

reimplementation of dürmuth et al.s OMEN (ESSoS 2015), simplified. three pieces.

**train.** count every (k-1) character context → next character pair in a breach corpus, convert to log probs with add 1 smoothing. with `k=3` and the 10k seclists corpus we get ~48 distinct characters and a few thousand observed contexts.

```python
for word in corpus:
    padded = START_TOK * (k - 1) + word + END_TOK
    for i in range(len(padded) - k + 1):
        ctx = padded[i:i + k - 1]
        nxt = padded[i + k - 1]
        counts[ctx][nxt] += 1
```
the `START_TOK`/`END_TOK` padding lets the model know "this character appears at the start" and "the password is about to end". without them the model cant learn that capitals cluster at position 0 or that digits cluster at the end.

**score.** given a password, return `log P(password)` by summing log probs of each (context, next) pair. this is the ANALYTICAL prediction of how hard the password is to crack. a password with `log P = -k` is expected to take roughly `e^k = 10^(k/2.3)` guesses under an ideal probability ordered enumerator.

**enumerate.** priority queue beam search.
```python
heap = [(0.0, START_TOK * (k - 1))]
while heap and emitted < budget:
    cost, prefix = heapq.heappop(heap)
    for nxt, lp in log_probs[ctx].items():
        new_cost = cost - lp
        heapq.heappush(heap, (new_cost, prefix + nxt))
```
pop the lowest cost (highest probability) prefix, extend by every possible next char, push back. emit when a prefix ends. the beam prunes to top `beam_size` prefixes when the heap explodes (length 10+), same trick real OMEN uses.

theres also `enumerate_fixed_length(model, length)` which forces emitted candidates to a specific length. the length experiment uses this, without it the heap floods with short high probability candidates and never reaches the test length.

result on the built in demo (budget 500k): `monkey` cracks at rank 169,246, everything else exceeds budget. expected since 10k seclists is way smaller than real OMENs rockyou (14M).

## 8 · the length experiment (`08_length_experiment.py`)

the research question: does the papers "length dominates" claim hold against a smart attacker who learned the distribution of human chosen passwords? or does length only help to the extent it adds entropy the attacker hasnt already modeled?

setup. two flavours of target at each length.
- `random_uniform` — `secrets.choice` from `[a-z0-9]` (true crypto random)
- `human_shaped` — sampled from the same markov model the attacker trained on (worst case for the defender, the attacker has a perfect model of how you pick)

each cell of (length × flavour) generates 30 targets and:
1. computes the analytical median `log P` under the model
2. optionally runs the enumerator and counts how many it cracks within `--budget`

the analytical metric is the real signal because it covers every length without the enumerator timing out. the empirical run is a sanity check.

**results** (30 targets per cell, 3-gram markov on seclists 10k):

| length | random_uniform 10^(rank) | human_shaped 10^(rank) | gap |
|---|---|---|---|
| 6  | 10^12.1 | 10^9.3  | ~10^3 |
| 8  | 10^15.9 | 10^11.3 | ~10^4.6 |
| 10 | 10^19.0 | 10^13.9 | ~10^5 |
| 12 | 10^23.1 | 10^16.2 | ~10^7 |
| 14 | 10^26.3 | 10^18.9 | ~10^7.4 |
| 16 | 10^30.2 | 10^21.1 | ~10^9 |
| 20 | 10^37.1 | 10^26.5 | ~10^11 |

three things the paper should claim with this data.

1. **length dominates both categories.** each extra character adds ~1.9 orders of magnitude to random_uniform crack rank and ~1.5 OoM to human_shaped. both grow exponentially. the papers headline holds.

2. **human shaped passwords pay a multi order penalty.** at length 10 human shaped is 10^5 easier to crack than random. so a 10 char human chosen password is roughly as crackable as an 8 char random one.

3. **length only advice has a conversion rate.** a 20 char human passphrase (10^26.5) is roughly equivalent to a 14 char random password (10^26.3). so if users insist on human shaped passwords (and they will), "use 20 characters" gets them to the same place as "use 14 random chars" gets a security conscious user.

honest limitations to put in the paper:

- 10k seclists is tiny. real OMEN trains on rockyou (14M). bigger corpus = stronger model = human_shaped curve shifts down another 2-3 OoM. the 14 char "safe threshold" probably becomes 16-17 against a rockyou trained attacker.
- 3-gram markov misses long range structure. a neural model (script 09) captures more.
- this is a cpu only candidate generator. on a real gpu attack every number translates to wall clock time as `rank / 10^10 H/s` (md5) or `rank / 10 H/s` (bcrypt).

## 9 · neural cracker for colab (`09_neural_cracker_colab.py` + `.ipynb`)

char level gpt style transformer, ~810k parameters. designed to fit on a colab t4 (free tier) in under 25 minutes end to end.

architecture matches passbert / melicher 2016 in shape, scaled down:
- vocab built from data (~50 chars on seclists 10k, more on rockyou)
- context window 32 chars
- 4 transformer blocks, 4 heads, d_model 128
- causal self attention via `F.scaled_dot_product_attention` (fused pytorch 2.x kernel, lets a t4 train in minutes)

generation is nucleus sampling (top-p 0.95) with deduplication and log probability ranking. output is a plain wordlist, feed it to hashcat via `run_with_hashcat.sh neural`.

why neural beats markov: markov sees only the last 2 characters. a transformer sees the entire prefix at every step so it learns patterns like "capital at position 0 → digits at position L−4..L−1" that markov cant represent. published benchmarks (pasquini 2021) show ~25-50% more cracks at the same budget. for the paper that means the markov numbers above are an UNDERESTIMATE of how exposed human shaped passwords actually are.

colab setup:
1. upload `09_neural_cracker_colab.ipynb` to colab.research.google.com
2. runtime → change runtime type → t4 gpu
3. run all cells, outputs `neural_candidates.txt`
4. download it, then on a gpu box: `hashcat -a 0 -m 0 hashes.txt neural_candidates.txt`

## hashcat wrapper (`run_with_hashcat.sh`)

the honest version of "the cracker". python generates ranked candidates, hashcat does the hashing on the gpu. same architecture every serious modern cracker uses (passgan papers, pcfg_cracker, OMEN, korelogics competition tooling). the wrapper just wires them together.

```bash
./run_with_hashcat.sh markov hashes.txt       # markov candidates
./run_with_hashcat.sh neural hashes.txt       # neural candidates
./run_with_hashcat.sh both   hashes.txt       # ensemble
```

`hashes.txt` is one md5 hex digest per line. change the `HASHCAT_MODE` env var for other algorithms (-m 100 sha1, -m 1000 ntlm, -m 3200 bcrypt).

---

## what changed in the papers recommendations

before (from the existing notebook + writeup):

> length absolutely dominates everything else (a 20 char lowercase passphrase is around a trillion times stronger than an 8 char mix of everything just because of length)

after (with the smart attacker measurements):

> length dominates against both random and human shaped attacks, but human shaped passwords pay a 3-10 order of magnitude penalty that grows with length. against a markov trained attacker (and worse against a neural one), a 20 char human passphrase is roughly equivalent to a 14 char random password. the trillion fold claim is right in SHAPE but underestimates the gap between random and human passwords by several orders of magnitude.

for the paper specifically:

- §2 background — add bcrypt comparison (script 01) and replace the implicit "brute force is the only attack" framing with markov/neural.
- §3 methodology — cite scripts 07-09 as the smart attacker baseline. the current `charset^length` formula is now the UPPER bound, the analytical markov score is the predicted crack rank.
- §4 results — add the length experiment table from script 08. this is the new headline.
- §7 discussion — replace "length dominates" with the refined claim above. state the seclists 10k limitation honestly, note that a rockyou trained model would shift the human_shaped curve further down.
- appendix — drop scripts 07 and 08 in full, theyre 200 lines each, fit cleanly.

next thing worth doing on this: retrain the neural model on rockyou (14M passwords) instead of the seclists 10k, rerun the length experiment with the neural cracker instead of markov. the human_shaped curve should shift down another 2-3 orders of magnitude, which gives stronger numbers for the discussion section.
