#!/usr/bin/env bash
# run_with_hashcat.sh -- pipe python generated candidates into hashcat
#
# the python research scripts (07 markov, 09 neural) emit a plain wordlist,
# one candidate per line in approximate descending probability order. this
# wrapper hands that wordlist to hashcat which is what actually does the
# high throughput hashing on your gpu.
#
# why this two stage setup is the standard:
# - python is the wrong tool for the hashing inner loop (~1.7M H/s on a
#   cpu core vs ~10B H/s for hashcat on a gpu = 6000x gap)
# - python is the RIGHT tool for the candidate generation logic, thats
#   where the research happens (markov, pcfg, neural)
# - hashcat reads candidates from a file or stdin, so you can pipe
#   `python3 generator.py | hashcat -a 0 -m 0 hashes.txt -`
#
# usage:
#   ./run_with_hashcat.sh markov  hashes.txt       # 1M markov candidates
#   ./run_with_hashcat.sh neural  hashes.txt       # neural wordlist
#   ./run_with_hashcat.sh both    hashes.txt       # run both
#
# `hashes.txt` is one md5 hex digest per line. -m 0 = raw md5. change to
# -m 100 for sha1, -m 1000 for ntlm (windows), -m 3200 for bcrypt etc.

set -euo pipefail

MODE="${1:-markov}"
HASHES="${2:-hashes.txt}"
BUDGET="${BUDGET:-1000000}"
HASHCAT_MODE="${HASHCAT_MODE:-0}"          # 0 = md5

if [[ ! -f "$HASHES" ]]; then
    echo "no $HASHES, generating a 5 target demo" >&2
    {
        printf '%s' "monkey"        | md5sum | cut -d' ' -f1
        printf '%s' "summer2024"    | md5sum | cut -d' ' -f1
        printf '%s' "Tr0ub4dor&3"   | md5sum | cut -d' ' -f1
        printf '%s' "qwerty123"     | md5sum | cut -d' ' -f1
        printf '%s' "Password1!"    | md5sum | cut -d' ' -f1
    } > "$HASHES"
fi

if ! command -v hashcat >/dev/null; then
    echo "hashcat not installed. install: https://hashcat.net/hashcat/" >&2
    echo "(this script also emits the wordlist file so you can run it elsewhere)" >&2
    HAS_HASHCAT=0
else
    HAS_HASHCAT=1
fi

run_attack() {
    local name="$1"
    local wordlist="$2"
    local generator_cmd="$3"

    echo "===== $name ====="
    echo "+ $generator_cmd"
    eval "$generator_cmd"
    echo "  wordlist: $(wc -l < "$wordlist") candidates in $wordlist"

    if [[ "$HAS_HASHCAT" == "1" ]]; then
        # --quiet trims hashcats normal output, --potfile-disable so reruns
        # dont see previous cracks
        hashcat -a 0 -m "$HASHCAT_MODE" --quiet --potfile-disable \
                -o "${name}_cracked.txt" "$HASHES" "$wordlist" || true
        echo "  cracked rows ($name):"
        cat "${name}_cracked.txt" 2>/dev/null | sed 's/^/    /'
    fi
    echo
}

case "$MODE" in
    markov)
        run_attack markov markov_candidates.txt \
            "python3 cracking_demos/07_markov_cracker.py \
                --emit-wordlist markov_candidates.txt --budget $BUDGET"
        ;;
    neural)
        run_attack neural neural_candidates.txt \
            "python3 cracking_demos/09_neural_cracker_colab.py"
        ;;
    both)
        run_attack markov markov_candidates.txt \
            "python3 cracking_demos/07_markov_cracker.py \
                --emit-wordlist markov_candidates.txt --budget $BUDGET"
        run_attack neural neural_candidates.txt \
            "python3 cracking_demos/09_neural_cracker_colab.py"
        ;;
    *)
        echo "unknown mode: $MODE (try: markov | neural | both)" >&2
        exit 1
        ;;
esac
