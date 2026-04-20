# the anatomy of a password

dasc 4850 final project, april 2026. linear regression that predicts how long it takes to crack a given password, trained on around 12 thousand passwords pulled from real breach data (RockYou 2009, LinkedIn 2012, Adobe 2013, Dropbox 2016, Pwdb 2021 / COMB) plus 2 thousand strong passwords I generated to cover the high end. i run vaultwarden on my homelab and ive always wanted to actually quantify what makes a password strong instead of just trusting whatever advice the IT guy at any random workplace tells me to do.

slides are live at https://gbasran.github.io/password-crack-time/slides.html (arrow keys to navigate, f for fullscreen).

the model hit R² of 0.988 on the held out test set across 32 orders of magnitude in crack time, RMSE 1.032 in log10 seconds. two main takeaways from the coefficients: password length absolutely dominates everything else (a 20 char lowercase passphrase is around a trillion times stronger than an 8 char mix of everything just because of length), and dictionary words are catastrophic no matter what `!` and `123` you tack on (a t-test put it at around 14 thousand times harder to crack on average without one).

the notebook auto-downloads the breach data on first run so you can just clone and go. `pip install pandas numpy seaborn matplotlib scikit-learn scipy` and open final_project.ipynb. takes around 30 seconds end to end on a normal laptop.

deliverables in the repo: final_project.ipynb has the full analysis with explainer text written assuming no cybersec background, report.pdf is the written writeup, slides.html is the deck (also live on github pages above). data/password_dataset.csv has the engineered features if you want to skip extraction and just play with the features directly.
