"""Step 1 (run ONCE on your own PC): build the Kaggle upload package.

What it does
  * collects the balanced bona fide / PAI pairs of IDIAP (full), SCUT (full) and PLUS
    (course rule: only genuine samples that have a fake counterpart)
  * resizes every image to height 160 px, keeping the aspect ratio
    (IDIAP 160x426, SCUT 160x356 after a 90 degree rotation to landscape, PLUS 160x613)
  * assigns every SUBJECT to one of 5 folds (seed 42), so no person is in two folds
  * writes images + metadata.csv into OUT_DIR and zips it to fvpad_data.zip

Usage (Windows PowerShell or Git Bash, from D:\\AIS\\IPI\\kaggle):
  python 0_prepare_data.py --raw "D:/AIS/IPI/Data_FV_Spoofing_WS2025_26/Data_FV_Spoofing_WS2025_26" --out "D:/AIS/IPI/kaggle/fvpad_data"
"""
import argparse
import os
import re
import shutil
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

HEIGHT = 160


def idiap_items(raw):
    """IDIAP full: train/dev in subject folders, test anonymised + translator file."""
    items = []
    for split in ["train", "dev"]:
        for cls in ["real", "spoof"]:
            d = os.path.join(raw, "IDIAP", "full", split, cls)
            for subj in sorted(os.listdir(d)):
                for f in sorted(os.listdir(os.path.join(d, subj))):
                    if f.endswith(".png"):
                        stem = f[:-4]  # e.g. 001_L_1
                        items.append(dict(src=os.path.join(d, subj, f), subject=stem[:3], finger=stem[:5],
                                          sample=stem, label=int(cls == "spoof")))
    tr = os.path.join(raw, "IDIAP", "translator_full_files.txt")
    with open(tr) as fh:
        for line in fh:
            if not line.strip():
                continue
            anon, orig = line.split()
            spoof = orig.endswith("_spoof.png")
            stem = orig.replace("_spoof.png", "").replace(".png", "")
            items.append(dict(src=os.path.join(raw, "IDIAP", "full", "test", anon), subject=stem[:3],
                              finger=stem[:5], sample=stem, label=int(spoof)))
    return items


def scut_items(raw):
    """SCUT full: ID_finger_session_shot_light.bmp, subject = folder name."""
    items = []
    for split in ["train", "dev", "test"]:
        for cls in ["real", "spoof"]:
            d = os.path.join(raw, "SCUT", "full", split, cls)
            for subj in sorted(os.listdir(d), key=int):
                for f in sorted(os.listdir(os.path.join(d, subj))):
                    if f.lower().endswith(".bmp"):
                        p = f[:-4].split("_")
                        items.append(dict(src=os.path.join(d, subj, f), subject=f"{int(subj):03d}",
                                          finger=f"{int(subj):03d}_{p[1]}", sample=f"{f[:-4]}_{cls}",
                                          label=int(cls == "spoof")))
    return items


def plus_items(raw):
    """PLUS: keep all PAIs and only the bona fide images with the identical file name."""
    items = []
    sd = os.path.join(raw, "PLUS", "spoof")
    for subj in sorted(os.listdir(sd)):
        for f in sorted(os.listdir(os.path.join(sd, subj))):
            if not f.endswith(".png"):
                continue
            real = os.path.join(raw, "PLUS", "real", subj, f)
            if not os.path.exists(real):
                print("  PLUS: no bona fide counterpart for", subj, f, "-> skipped")
                continue
            p = f[:-4].split("_")  # scanner, side, user, session, finger, image
            for lab, src in [(1, os.path.join(sd, subj, f)), (0, real)]:
                items.append(dict(src=src, subject=subj, finger=f"{subj}_{p[4]}", sample=f[:-4], label=lab))
    return items


def assign_folds(df, n_folds=5, seed=42):
    subjects = sorted(df.subject.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    fold_of = {}
    for k, chunk in enumerate(np.array_split(np.array(subjects), n_folds)):
        for s in chunk:
            fold_of[s] = k + 1
    return df.subject.map(fold_of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder that contains IDIAP, SCUT, PLUS")
    ap.add_argument("--out", required=True, help="output folder (will be created)")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    rows = []
    for name, fn in [("IDIAP", idiap_items), ("SCUT", scut_items), ("PLUS", plus_items)]:
        items = fn(a.raw)
        df = pd.DataFrame(items)
        df["dataset"] = name
        df["fold"] = assign_folds(df)
        c = Counter(df.label)
        print(f"{name}: {len(df)} images, bona fide {c[0]}, PAI {c[1]}, subjects {df.subject.nunique()}")
        rows.append(df)
    meta = pd.concat(rows, ignore_index=True)

    # ---- write resized images
    os.makedirs(a.out, exist_ok=True)
    paths = []
    for i, r in enumerate(meta.itertuples()):
        im = Image.open(r.src).convert("L")
        if r.dataset == "SCUT":  # portrait 288x640 -> landscape
            im = im.rotate(90, expand=True)
        w = round(im.width * HEIGHT / im.height)
        im = im.resize((w, HEIGHT), Image.LANCZOS)
        rel = f"images/{r.dataset}/{'pai' if r.label else 'bf'}/{r.subject}/{i:05d}.png"
        os.makedirs(os.path.join(a.out, os.path.dirname(rel)), exist_ok=True)
        im.save(os.path.join(a.out, rel))
        paths.append(rel)
        if i % 1000 == 0:
            print(f"  wrote {i}/{len(meta)}")
    meta["path"] = paths
    meta["src"] = meta.src.map(lambda s: re.sub(r"^.*Data_FV_Spoofing_WS2025_26[\\/]", "", s).replace("\\", "/"))
    meta = meta[["path", "dataset", "subject", "finger", "sample", "label", "fold", "src"]]
    meta.to_csv(os.path.join(a.out, "metadata.csv"), index=False)

    # ---- sanity checks
    for ds, g in meta.groupby("dataset"):
        assert g.groupby("subject").fold.nunique().max() == 1, f"{ds}: subject in two folds!"
        t = g.groupby(["fold", "label"]).size().unstack()
        t.columns = ["bona fide", "PAI"]
        t["subjects"] = g.groupby("fold").subject.nunique()
        print(f"\n{ds} per fold:\n{t}")
    print("\nmetadata.csv written:", len(meta), "rows")

    if not a.no_zip:
        z = shutil.make_archive(a.out, "zip", a.out)
        print("ZIP ready for Kaggle upload:", z, f"({os.path.getsize(z) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
