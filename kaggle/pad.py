"""Finger-vein PAD with (fine-tuned) pre-trained CNN features + k-NN.   (version 2)

Commands
  python pad.py baseline  --data DIR --out results.jsonl        # Experiment 1: histogram + frozen CNN (L0)
  python pad.py finetune  --data DIR --out results.jsonl --arch mobilenet_v2 --level L3   # Experiment 2+
  python pad.py summary   --out results.jsonl                   # mean +- std table -> summary.csv

Protocol (see project proposal, Section 5.4)
  * 5 subject-disjoint folds per dataset (column `fold` in metadata.csv)
  * training size: BA = the 4 other folds, S1 = 2 of them, S2 = 1 of them (chosen at random per seed)
  * inner validation = 20 % of the TRAINING subjects (early stopping + choice of k); the test fold is never touched
  * k-NN: L2-normalised embeddings, Euclidean distance, k chosen from K_GRID on the inner validation split,
    then refitted on the whole training pool
  * metrics (ISO/IEC 30107-3, attack = positive class): APCER, BPCER, ACER

New in version 2 (after Experiment 1 hit the 0 % ceiling)
  * score-based metrics for every run, from a k-NN distance-ratio score d_BF / (d_BF + d_PAI):
      EER, AUC, BPCER20 (BPCER at APCER <= 5 %) and d' (class separability; does not saturate at 0 % error)
  * --norm none|zscore|equalize   per-image grey-value normalisation = brightness-cue control
  * --mask-mode both|test         with --mask-left N: "test" trains WITH the PLUS label visible and tests WITHOUT it
                                  (shortcut test); "both" masks train and test (v1 behaviour)
  * --exp NAME                    experiment tag in the result key (default exp1 / ft)
  * summary now separates conditions (mask / normalisation / learning rate) instead of pooling them
  Result keys for default settings are identical to version 1, so old result files stay resumable.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from PIL import Image

K_GRID = [1, 3, 5, 7, 9, 15]
SIZES = {"BA": 4, "S1": 2, "S2": 1}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------- metrics
def pad_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    att, bf = y_true == 1, y_true == 0
    apcer = float(((y_pred == 0) & att).sum() / max(att.sum(), 1))  # attacks accepted as bona fide
    bpcer = float(((y_pred == 1) & bf).sum() / max(bf.sum(), 1))  # bona fide rejected as attacks
    return dict(APCER=100 * apcer, BPCER=100 * bpcer, ACER=100 * (apcer + bpcer) / 2,
                n_attack=int(att.sum()), n_bonafide=int(bf.sum()))


def score_metrics(y_true, score):
    """Threshold-free metrics. score: higher = more likely an attack."""
    from sklearn.metrics import roc_auc_score, roc_curve
    y, s = np.asarray(y_true), np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return {}
    fpr, tpr, _ = roc_curve(y, s)  # fpr = BPCER, 1 - tpr = APCER
    apcer = 1 - tpr
    i = int(np.argmin(np.abs(fpr - apcer)))
    eer = (fpr[i] + apcer[i]) / 2
    ok = apcer <= 0.05
    bpcer20 = float(fpr[ok].min()) if ok.any() else 1.0
    a, b = s[y == 1], s[y == 0]
    dprime = float(abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-12))
    return dict(EER=100 * float(eer), AUC=100 * float(roc_auc_score(y, s)), BPCER20=100 * bpcer20, dprime=dprime)


# ----------------------------------------------------------------------------- splits
def make_split(meta_ds, test_fold, size, seed, val_frac=0.2):
    """Returns index arrays (fit, val, test) into meta_ds. fit+val = training pool."""
    rng = np.random.default_rng(1000 * seed + test_fold)
    others = [f for f in range(1, 6) if f != test_fold]
    train_folds = sorted(rng.choice(others, SIZES[size], replace=False).tolist())
    pool = meta_ds[meta_ds.fold.isin(train_folds)]
    subjects = np.array(sorted(pool.subject.unique()))
    n_val = max(1, int(round(val_frac * len(subjects))))
    val_subj = set(rng.choice(subjects, n_val, replace=False).tolist())
    fit = pool.index[~pool.subject.isin(val_subj)].to_numpy()
    val = pool.index[pool.subject.isin(val_subj)].to_numpy()
    test = meta_ds.index[meta_ds.fold == test_fold].to_numpy()
    # leakage guards
    assert not set(meta_ds.loc[test, "subject"]) & set(pool.subject), "subject leak train/test"
    assert not set(meta_ds.loc[fit, "subject"]) & val_subj, "subject leak fit/val"
    return fit, val, test, train_folds


# ----------------------------------------------------------------------------- k-NN
def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def knn_score(Xpool, ypool, Xq, k):
    """Distance-ratio score d_BF / (d_BF + d_PAI); d_c = mean distance to the k nearest samples of class c.
    0 = close to bona fide, 1 = close to attacks."""
    from sklearn.neighbors import NearestNeighbors
    d = []
    for c in (0, 1):
        ref = Xpool[ypool == c]
        kk = max(1, min(k, len(ref)))
        d.append(NearestNeighbors(n_neighbors=kk).fit(ref).kneighbors(Xq)[0].mean(1))
    return d[0] / (d[0] + d[1] + 1e-12)


def knn_eval(X, y, fit, val, test, normalise=True):
    """Choose k on (fit -> val) by ACER, refit on fit+val, predict test. Also returns a continuous score."""
    from sklearn.neighbors import KNeighborsClassifier
    Xn = l2n(X) if normalise else X
    best = None
    for k in K_GRID:
        if k > len(fit):
            break
        pred = KNeighborsClassifier(n_neighbors=k).fit(Xn[fit], y[fit]).predict(Xn[val])
        acer = pad_metrics(y[val], pred)["ACER"]
        if best is None or acer < best[1]:
            best = (k, acer)
    k = best[0]
    pool = np.concatenate([fit, val])
    knn = KNeighborsClassifier(n_neighbors=k).fit(Xn[pool], y[pool])
    pred = knn.predict(Xn[test])
    score = knn_score(Xn[pool], y[pool], Xn[test], k)
    return k, best[1], pred, score


# ----------------------------------------------------------------------------- data
def load_meta(data_dir):
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"), dtype={"subject": str})
    return meta


def normalise_image(a, norm):
    """Per-image grey-value normalisation (uint8 in, uint8 out)."""
    if norm == "none":
        return a
    if norm == "equalize":
        from PIL import ImageOps
        return np.asarray(ImageOps.equalize(Image.fromarray(a)))
    if norm == "zscore":  # every image gets mean 128 and std 48
        f = a.astype(np.float32)
        return np.clip(128 + 48 * (f - f.mean()) / (f.std() + 1e-6), 0, 255).astype(np.uint8)
    raise ValueError(norm)


def load_images(data_dir, meta_ds, mask_left=0, norm="none"):
    """All images of one dataset as uint8 array (N, H, W). Same size within a dataset.
    The mask is applied before the normalisation."""
    arr = np.stack([np.asarray(Image.open(os.path.join(data_dir, p)).convert("L")) for p in meta_ds.path])
    if mask_left > 0:
        arr = arr.copy()
        arr[:, :, :mask_left] = 0
    if norm != "none":
        arr = np.stack([normalise_image(a, norm) for a in arr])
    return arr


def variant_of(a, ds):
    """Short tag for the image condition. Equals the v1 tag ("m0", "m53") for default settings."""
    mask = a.mask_left if ds == "PLUS" else 0
    v = f"m{mask}" + ("t" if (mask and a.mask_mode == "test") else "")
    return v + (f"-{a.norm}" if a.norm != "none" else ""), mask


def load_condition(a, ds, mds):
    """Returns (train view, test view, variant, mask). The two views differ only with --mask-mode test."""
    variant, mask = variant_of(a, ds)
    if mask and a.mask_mode == "test":
        return load_images(a.data, mds, 0, a.norm), load_images(a.data, mds, mask, a.norm), variant, mask
    imgs = load_images(a.data, mds, mask, a.norm)
    return imgs, imgs, variant, mask


def histogram_features(imgs, bins=10):
    h = np.stack([np.histogram(im, bins=bins, range=(0, 256))[0] for im in imgs]).astype(np.float32)
    return h / h.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- results file
def done_keys(out):
    keys = set()
    if os.path.exists(out):
        with open(out) as fh:
            for line in fh:
                if line.strip():
                    keys.add(json.loads(line)["key"])
    return keys


def append(out, rec):
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def fmt(sm, name):
    return f"{sm.get(name, float('nan')):5.2f}"


# ============================================================================= torch part
def torch_parts():
    import torch
    import torch.nn as nn
    import torchvision
    return torch, nn, torchvision


def build_model(arch, level):
    torch, nn, tv = torch_parts()

    class PADNet(nn.Module):
        def __init__(self):
            super().__init__()
            if arch == "mobilenet_v2":
                m = tv.models.mobilenet_v2(weights=tv.models.MobileNet_V2_Weights.IMAGENET1K_V2)
                self.body, self.dim = m.features, 1280
                self.blocks = list(m.features)  # 19 blocks, index 0..18
                start = {"L0": None, "L0h": None, "L1": 17, "L2": 14, "L3": 0}[level]
                trainable = [] if start is None else self.blocks[start:]
            elif arch == "resnet18":
                m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
                self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
                self.dim = 512
                self.blocks = list(self.body)
                trainable = {"L0": [], "L0h": [], "L1": [m.layer4], "L2": [m.layer3, m.layer4],
                             "L3": self.blocks}[level]
            else:
                raise ValueError(arch)
            for p in self.body.parameters():
                p.requires_grad = False
            for b in trainable:
                for p in b.parameters():
                    p.requires_grad = True
            self.frozen_blocks = [b for b in self.blocks if not any(p.requires_grad for p in b.parameters())]
            self.head_pre = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.dim, 256), nn.ReLU())
            self.head_out = nn.Linear(256, 1)
            self.level = level

        def gap(self, x):
            return self.body(x).mean(dim=(2, 3))

        def forward(self, x):
            return self.head_out(self.head_pre(self.gap(x))).squeeze(1)

        def embed(self, x):
            g = self.gap(x)
            return self.head_pre[1:](g) if self.level == "L0h" else g  # L0h: 256-d bottleneck (no dropout)

        def train_mode(self):
            self.train()
            for b in self.frozen_blocks:  # keep BN statistics of frozen layers fixed
                b.eval()

    return PADNet()


def to_batch(torch, imgs_u8, device, augment=False):
    """uint8 (B,H,W) numpy -> normalised float (B,3,H,W) tensor on device, optional light augmentation."""
    x = torch.from_numpy(imgs_u8).to(device, non_blocking=True).float().div_(255).unsqueeze(1)
    if augment:
        B, _, H, W = x.shape
        # brightness / contrast +-10 %
        b = 1 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.2
        c = 1 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.2
        m = x.mean(dim=(2, 3), keepdim=True)
        x = ((x - m) * c + m) * b
        # horizontal flip
        flip = torch.rand(B, device=device) < 0.5
        x[flip] = x[flip].flip(-1)
        # small rotation (+-3 deg) and shift (+-4 %) via affine grid
        ang = (torch.rand(B, device=device) - 0.5) * 2 * np.deg2rad(3)
        tx = (torch.rand(B, device=device) - 0.5) * 2 * 0.08
        ty = (torch.rand(B, device=device) - 0.5) * 2 * 0.08
        cos, sin = torch.cos(ang), torch.sin(ang)
        theta = torch.stack([torch.stack([cos, -sin * H / W, tx], 1), torch.stack([sin * W / H, cos, ty], 1)], 1)
        grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
        x = torch.nn.functional.grid_sample(x, grid, padding_mode="border", align_corners=False)
        x = x.clamp_(0, 1)
    x = x.expand(-1, 3, -1, -1)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def extract(model, imgs, device, bs=64):
    torch, _, _ = torch_parts()
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        for i in range(0, len(imgs), bs):
            out.append(model.embed(to_batch(torch, imgs[i:i + bs], device)).float().cpu().numpy())
    return np.concatenate(out)


def head_predict(model, imgs, device, bs=64):
    torch, _, _ = torch_parts()
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        for i in range(0, len(imgs), bs):
            out.append(torch.sigmoid(model(to_batch(torch, imgs[i:i + bs], device)).float()).cpu().numpy())
    return np.concatenate(out)


def train_model(model, imgs, y, fit, val, device, a, seed):
    torch, nn, _ = torch_parts()
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model.to(device)
    head = list(model.head_pre.parameters()) + list(model.head_out.parameters())
    body = [p for p in model.body.parameters() if p.requires_grad]
    groups = [{"params": head, "lr": a.lr_head}]
    if body:
        groups.append({"params": body, "lr": a.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(fit) / a.batch))
    total = a.epochs * steps_per_epoch
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y, dtype=torch.float32)
    best, best_state, bad, best_ep = 1e9, None, 0, 0
    for ep in range(a.epochs):
        model.train_mode()
        order = np_rng.permutation(fit)
        for i in range(0, len(order), a.batch):
            idx = order[i:i + a.batch]
            if len(idx) < 2:
                continue
            xb = to_batch(torch, imgs[idx], device, augment=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = lossf(model(xb).float(), yt[idx].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        # validation loss
        p = np.clip(head_predict(model, imgs[val], device), 1e-6, 1 - 1e-6)
        vl = float(-np.mean(y[val] * np.log(p) + (1 - y[val]) * np.log(1 - p)))
        if vl < best - 1e-4:
            best, bad, best_ep = vl, 0, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return best_ep, best


def get_device():
    torch, _, _ = torch_parts()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================= commands
def cmd_baseline(a):
    """Experiment 1: B0 grey histogram + k-NN and L0 frozen ImageNet CNN + k-NN."""
    meta = load_meta(a.data)
    keys = done_keys(a.out)
    device = get_device() if a.archs else None
    print("device:", device)
    exp = a.exp or "exp1"
    for ds in a.datasets:
        mds = meta[meta.dataset == ds].reset_index(drop=True)
        imgs_tr, imgs_te, variant, mask = load_condition(a, ds, mds)
        same = imgs_tr is imgs_te
        y = mds.label.to_numpy()
        feats = {"hist10": (histogram_features(imgs_tr), None if same else histogram_features(imgs_te))}
        for arch in a.archs:
            t = time.time()
            model = build_model(arch, "L0").to(device)
            ftr = extract(model, imgs_tr, device)
            feats[f"{arch}-L0"] = (ftr, None if same else extract(model, imgs_te, device))
            print(f"[{ds}] {arch} features {ftr.shape} in {time.time() - t:.0f}s")
            del model
        for method, (Xtr, Xte) in feats.items():
            for size in a.sizes:
                for seed in a.seeds:
                    for fold in a.folds:
                        key = f"{exp}|{ds}|{method}|{size}|f{fold}|s{seed}|{variant}"
                        if key in keys:
                            continue
                        is_test = (mds.fold == fold).to_numpy()[:, None]
                        X = Xtr if Xte is None else np.where(is_test, Xte, Xtr)  # test rows from the test view
                        fit, val, test, tf = make_split(mds, fold, size, seed)
                        k, vacer, pred, score = knn_eval(X, y, fit, val, test, normalise=method != "hist10")
                        m = pad_metrics(y[test], pred)
                        sm = score_metrics(y[test], score)
                        rec = dict(key=key, exp=exp, dataset=ds, method=method, size=size, fold=fold, seed=seed,
                                   variant=variant, mask_left=mask, mask_mode=a.mask_mode, norm=a.norm,
                                   train_folds=tf, n_fit=len(fit), n_val=len(val), k=k, val_ACER=vacer, **m, **sm)
                        append(a.out, rec)
                        print(f"{ds:5s} {method:16s} {variant:12s} {size} fold{fold} seed{seed} k={k:2d} "
                              f"APCER={m['APCER']:5.1f} BPCER={m['BPCER']:5.1f} ACER={m['ACER']:5.1f} "
                              f"EER={fmt(sm, 'EER')} d'={fmt(sm, 'dprime')}")


def cmd_finetune(a):
    """Experiment 2+: fine-tune to a given depth, then k-NN on the embeddings (+ head readout)."""
    torch, _, _ = torch_parts()
    meta = load_meta(a.data)
    keys = done_keys(a.out)
    device = get_device()
    print("device:", device)
    if a.level == "L0":
        raise SystemExit("L0 has no training - use the baseline command")
    exp = a.exp or "ft"
    for ds in a.datasets:
        mds = meta[meta.dataset == ds].reset_index(drop=True)
        imgs, imgs_te, variant, mask = load_condition(a, ds, mds)
        y = mds.label.to_numpy()
        for size in a.sizes:
            for seed in a.seeds:
                for fold in a.folds:
                    key = f"{exp}|{ds}|{a.arch}-{a.level}|{size}|f{fold}|s{seed}|{variant}|lr{a.lr_backbone}"
                    if key in keys:
                        continue
                    t = time.time()
                    fit, val, test, tf = make_split(mds, fold, size, seed)
                    model = build_model(a.arch, a.level)
                    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    ep, vloss = train_model(model, imgs, y, fit, val, device, a, seed)
                    X = extract(model, imgs, device)
                    if imgs_te is not imgs:  # shortcut test: test images are seen without the label
                        X[test] = extract(model, imgs_te[test], device)
                    k, vacer, pred, score = knn_eval(X, y, fit, val, test)
                    m = pad_metrics(y[test], pred)
                    sm = score_metrics(y[test], score)
                    hp = (head_predict(model, imgs_te[test], device) > 0.5).astype(int)
                    mh = {f"head_{kk}": v for kk, v in pad_metrics(y[test], hp).items() if kk in ("APCER", "BPCER", "ACER")}
                    rec = dict(key=key, exp=exp, dataset=ds, method=f"{a.arch}-{a.level}", size=size, fold=fold,
                               seed=seed, variant=variant, mask_left=mask, mask_mode=a.mask_mode, norm=a.norm,
                               lr_backbone=a.lr_backbone, train_folds=tf, n_fit=len(fit), n_val=len(val),
                               trainable_params=n_train, best_epoch=ep, val_loss=vloss, k=k, val_ACER=vacer,
                               seconds=round(time.time() - t, 1), **m, **sm, **mh)
                    append(a.out, rec)
                    print(f"{ds:5s} {a.arch}-{a.level} {variant} {size} fold{fold} seed{seed} ep={ep:2d} k={k:2d} "
                          f"kNN ACER={m['ACER']:5.1f} (APCER {m['APCER']:.1f} / BPCER {m['BPCER']:.1f}) "
                          f"EER={fmt(sm, 'EER')} d'={fmt(sm, 'dprime')} head ACER={mh['head_ACER']:5.1f}  "
                          f"[{time.time() - t:.0f}s]")
                    if a.save_emb:
                        os.makedirs(a.save_emb, exist_ok=True)
                        np.save(os.path.join(a.save_emb, key.replace("|", "_") + ".npy"), X[test].astype(np.float16))
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()


def cmd_summary(a):
    rows = [json.loads(l) for l in open(a.out) if l.strip()]
    df = pd.DataFrame(rows)
    if "variant" not in df:  # version-1 result files: the condition is the 7th key field (m0 / m53)
        df["variant"] = None
    df["variant"] = df["variant"].fillna(df.key.str.split("|").str[6])
    if "lr_backbone" in df:
        df["variant"] = df["variant"] + df["lr_backbone"].map(lambda v: "" if pd.isna(v) else f" lr{v}")
    cols = ["APCER", "BPCER", "ACER"] + [c for c in ["EER", "AUC", "BPCER20", "dprime", "head_ACER", "seconds"] if c in df]
    g = df.groupby(["exp", "dataset", "method", "variant", "size"])
    s = g[cols].agg(["mean", "std"]).round(3)
    s.columns = [f"{c}_{st}" for c, st in s.columns]
    s["runs"] = g.size()
    s = s.reset_index()
    order = {"BA": 0, "S1": 1, "S2": 2}
    s = s.sort_values(["exp", "dataset", "variant", "size", "method"],
                      key=lambda c: c.map(order) if c.name == "size" else c)
    s.to_csv(a.summary, index=False)
    with pd.option_context("display.max_rows", 500, "display.width", 250):
        print(s.to_string(index=False))
    print("\nwritten:", a.summary)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["baseline", "finetune", "summary"]:
        p = sub.add_parser(name)
        p.add_argument("--out", default="results.jsonl")
        if name == "summary":
            p.add_argument("--summary", default="summary.csv")
            continue
        p.add_argument("--data", required=True, help="folder containing metadata.csv and images/")
        p.add_argument("--datasets", nargs="+", default=["IDIAP", "PLUS", "SCUT"])
        p.add_argument("--sizes", nargs="+", default=["BA", "S1", "S2"])
        p.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
        p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
        p.add_argument("--mask-left", type=int, default=0, help="black out N left columns (PLUS label control; ~53 px at height 160)")
        p.add_argument("--mask-mode", default="both", choices=["both", "test"],
                       help="both = mask train and test images; test = train with the label visible, test with it masked")
        p.add_argument("--norm", default="none", choices=["none", "zscore", "equalize"],
                       help="per-image grey-value normalisation (brightness-cue control)")
        p.add_argument("--exp", default="", help="experiment tag in the result key (default: exp1 / ft)")
        if name == "baseline":
            p.add_argument("--archs", nargs="*", default=["mobilenet_v2", "resnet18"])
        else:
            p.add_argument("--arch", default="mobilenet_v2", choices=["mobilenet_v2", "resnet18"])
            p.add_argument("--level", default="L3", choices=["L0h", "L1", "L2", "L3"])
            p.add_argument("--epochs", type=int, default=30)
            p.add_argument("--patience", type=int, default=5)
            p.add_argument("--batch", type=int, default=32)
            p.add_argument("--lr-head", type=float, default=1e-3)
            p.add_argument("--lr-backbone", type=float, default=3e-5)
            p.add_argument("--save-emb", default="", help="folder to save test-fold embeddings (.npy)")
    a = ap.parse_args()
    {"baseline": cmd_baseline, "finetune": cmd_finetune, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
