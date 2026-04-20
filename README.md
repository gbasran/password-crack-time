# The Anatomy of a Password: Predicting Crack Time

Final project for **DASC 4850** (Foundations of Data Science and AI in Python) at the University of Lethbridge. April 2026.

A linear regression model that predicts how long it would take a modern attacker to crack a given password. Trained on ~12,000 passwords combining real breach data (RockYou 2009, LinkedIn 2012, Adobe 2013, Dropbox 2016, and the Pwdb 2021 / COMB compilation) with generated strong passwords. I built this because i run [Vaultwarden](https://github.com/dani-garcia/vaultwarden) as my password manager and wanted to actually quantify what makes a password strong instead of trusting workplace IT advice.

## Live presentation

**[View the slides on GitHub Pages](https://gbasran.github.io/password-crack-time/slides.html)** — arrow keys or spacebar to navigate, F for fullscreen.

## Deliverables

| File | What it is |
|---|---|
| [`final_project.ipynb`](./final_project.ipynb) | Main analysis notebook. Build-up-from-zero explainers, feature engineering, EDA, t-test, Ridge regression, results |
| [`report.pdf`](./report.pdf) | Written report |
| [`slides.html`](./slides.html) | 6-slide presentation deck (also live on GitHub Pages) |

## Key results

| Metric | Test set |
|---|---|
| R² | **0.988** |
| RMSE (log10 sec) | 1.032 |
| MAE (log10 sec) | 0.531 |

The model explains about 99% of the variance in crack time across **32 orders of magnitude**. Two headline findings:

1. **Length dominates.** A 20-character lowercase passphrase like `correcthorsebatterystaple` (~10²⁸ combinations) is roughly a **trillion times stronger** than an 8-character mix of all four character types like `K$3p9!aB` (~10¹⁵ combinations), despite using a smaller alphabet. Every workplace policy that emphasizes complexity over length is focused on the wrong thing.
2. **Dictionary words are catastrophic.** Passwords without any common english word are about **14,000× harder to crack** on average (p < 0.001 via Welch t-test). `Monkey123!` is essentially as crackable as just `monkey` because the modifications are standard rules every cracker tests automatically.

## Practical takeaway

Use a password manager (Vaultwarden, Bitwarden, 1Password, KeePass) and let it generate 16+ character random passwords for everything. For the master password you have to memorize, use a 5-word [diceware](https://www.eff.org/dice) passphrase. Length wins.

## Running locally

The notebook auto-downloads breach data on first run (about 170k passwords across the four sources). Then samples down to 10k for runtime sanity.

```bash
pip install pandas numpy seaborn matplotlib scikit-learn scipy
jupyter notebook final_project.ipynb
```

Total runtime is about 30 seconds on a modern laptop.

## AI use

Topic selection, dataset sourcing, feature design (10 features picked based on hashcat / zxcvbn knowledge), and all interpretation are mine. Modeling decisions like Ridge over OLS, standardization, cross validation, and the train/val/test split came from MATH 3850 (numerical optimization and machine learning). AI helped with the math formalization for the crack time formula, the Cohen's d formula, scikit-learn API plumbing, and a few feature engineering edge cases. Full disclosure in the AI Use Appendix at the bottom of the notebook and report.
