"""
Ανάλυση συναισθήματος των κριτικών του Inside Airbnb
με το πολυγλωσσικό μοντέλο XLM-RoBERTa (XLM-T).

Το script εκτελεί τα ακόλουθα στάδια:
1. Φόρτωση των κριτικών για κάθε περιοχή.
2. Διατήρηση των κριτικών από την επιλεγμένη ημερομηνία και μετά.
3. Καθαρισμό του κειμένου και αφαίρεση αυτοματοποιημένων μηνυμάτων.
4. Ταξινόμηση των κριτικών σε θετικό, ουδέτερο και αρνητικό συναίσθημα.
5. Υπολογισμό συνεχούς δείκτη sentiment από -1 έως +1.
6. Αποθήκευση ενδιάμεσων και τελικών αποτελεσμάτων ανά περιοχή.

Παράδειγμα εκτέλεσης:
    python sentiment_reviews.py \
        --inputs athens=data/athens_reviews.csv.gz \
                 thessaloniki=data/thessaloniki_reviews.csv.gz \
                 crete=data/crete_reviews.csv.gz \
        --since 2023-01-01 \
        --out results
"""
import argparse
import math
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Αυτόματα μηνύματα της πλατφόρμας (ακυρώσεις) που δεν είναι πραγματικές κριτικές
AUTO_PATTERN = r"automated posting|canceled this reservation|cancelled this reservation"


def load_reviews(path, since):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] >= since].copy()
    df["comments"] = (
        df["comments"].fillna("").astype(str)
        .str.replace(r"<br\s*/?>", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["comments"].str.len() >= 3]
    df = df[~df["comments"].str.contains(AUTO_PATTERN, case=False, regex=True)]
    return df[["id", "listing_id", "date", "comments"]].reset_index(drop=True)


@torch.no_grad()
def score(texts, tok, model, device, batch_size):
    # Ταξινόμηση των κριτικών βάσει μήκους για τη μείωση του περιττού padding
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    probs = [None] * len(texts)
    use_amp = device.type == "cuda"
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = [texts[i] for i in idx]
        enc = tok(batch, padding=True, truncation=True, max_length=256,
                  return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(**enc).logits
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for k, i in enumerate(idx):
            probs[i] = p[k]
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="region=path/to/reviews.csv.gz")
    ap.add_argument("--since", default="2023-01-01", help="κρατά κριτικές από αυτή την ημερομηνία")
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]

    for item in args.inputs:
        region, path = item.split("=", 1)
        df = load_reviews(path, pd.Timestamp(args.since))
        print(f"{region}: {len(df):,} κριτικές μετά το φιλτράρισμα")

        rdir = os.path.join(args.out, region)
        os.makedirs(rdir, exist_ok=True)
        n_chunks = math.ceil(len(df) / args.chunk)

        for c in tqdm(range(n_chunks), desc=region):
            part = os.path.join(rdir, f"part_{c:05d}.csv.gz")
            if os.path.exists(part):
                continue
            sub = df.iloc[c * args.chunk:(c + 1) * args.chunk].copy()
            probs = score(sub["comments"].tolist(), tok, model, device, args.batch_size)
            for j, lab in enumerate(labels):
                sub[f"p_{lab}"] = [p[j] for p in probs]
            sub["sentiment_label"] = [labels[int(p.argmax())] for p in probs]
            sub["sentiment_score"] = sub["p_positive"] - sub["p_negative"]  # από -1 έως +1
            tmp = part + ".tmp"
            sub.drop(columns="comments").to_csv(tmp, index=False, compression="gzip")
            os.replace(tmp, part)

        parts = [pd.read_csv(os.path.join(rdir, f)) for f in sorted(os.listdir(rdir))
                 if f.startswith("part_") and f.endswith(".csv.gz")]
        final = pd.concat(parts, ignore_index=True)
        final.insert(0, "region", region)
        out_path = os.path.join(args.out, f"{region}_sentiment.csv.gz")
        final.to_csv(out_path, index=False, compression="gzip")
        print(f"Αποθηκεύτηκε: {out_path} ({len(final):,} γραμμές)")
        print(final["sentiment_label"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
