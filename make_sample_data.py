"""Membuat dataset SINTETIS kecil hanya untuk smoke test pipeline.
Jangan dipakai untuk laporan! Gunakan dataset asli (PhiUSIIL)."""
import random
import string
import pandas as pd

random.seed(42)
legit_domains = ["google.com", "wikipedia.org", "github.com", "detik.com", "kompas.com", "unsil.ac.id",
                 "tokopedia.com", "youtube.com", "stackoverflow.com", "bbc.com", "amazon.com", "kaggle.com"]
brands = ["paypal", "bca", "bni", "tokopedia", "google", "facebook", "apple", "dana"]
words = ["login", "verify", "secure", "update", "account", "confirm", "billing"]
tlds = ["xyz", "top", "info", "click", "ru", "tk"]
rnd = lambda n: "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

rows = []
for i in range(1500):
    d = random.choice(legit_domains)
    sub = random.choice(["www.", "", "blog.", "news."])
    path = "/".join(random.choice(["article", "wiki", "docs", "repo", "products"]) for _ in range(random.randint(0, 3)))
    rows.append((f"https://{sub}{d}/{path}/{rnd(random.randint(3, 8))}", 0))
for i in range(1500):
    b, w = random.choice(brands), random.choice(words)
    kind = random.random()
    if kind < 0.3:
        host = ".".join(str(random.randint(1, 255)) for _ in range(4))
    elif kind < 0.7:
        host = f"{b}-{w}.{rnd(random.randint(4, 9))}.{random.choice(tlds)}"
    else:
        host = f"{w}.{b}.{rnd(5)}-{rnd(3)}.{random.choice(tlds)}"
    scheme = random.choice(["http", "http", "https"])
    rows.append((f"{scheme}://{host}/{w}/{rnd(10)}.php?id={rnd(12)}&token={rnd(16)}", 1))

random.shuffle(rows)
pd.DataFrame(rows, columns=["URL", "label"]).to_csv("data/sample_synthetic.csv", index=False)
print("Saved data/sample_synthetic.csv (label: 1=phishing, 0=legit)")
