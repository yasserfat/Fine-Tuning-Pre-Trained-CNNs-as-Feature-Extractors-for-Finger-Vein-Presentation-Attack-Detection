# Experiment 2 on Kaggle: real fine-tuning with TinyNet

Notebook: `2_exp2_finetune_tinynet.ipynb`. It is self-contained: all code is in the notebook, and no `pad.py` is needed.

## What it does differently from Experiment 1

Experiment 1 kept the CNN **frozen** and ran k-NN on its features. This notebook **trains the network weights** with backpropagation. It compares three modes on the same subject-disjoint splits as Experiment 1:

| Mode | Trained | Trainable params (tinynet_e) |
|---|---|---|
| `frozen` | classifier head only (reference) | ~0.33 M (head) |
| `partial` | head + last 2 stages + final conv | ~0.95 M |
| `full` | **all layers**, BatchNorm re-estimated | ~1.09 M (all) |

Each run logs `backbone_rel_change` = ||θ_after − θ_before|| / ||θ_before||. It is exactly **0** for `frozen` and **> 0** for `partial`/`full`, which proves the backbone was actually fine-tuned.

Model: **TinyNet** from `timm`. `tinynet_e` is the smallest and fastest. To use a larger model, change `MODEL` to `tinynet_d` … `tinynet_a`.

## Step by step

1. **Prerequisite:** your Kaggle account must be phone-verified (*Settings → Phone verification*). Without that, GPU and Internet stay disabled.
2. **Dataset:** use the same private dataset **fvpad-data** as in Experiment 1. If it does not exist yet, go to *Datasets → New Dataset*, upload `D:\AIS\IPI\kaggle\fvpad_data.zip`, name it `fvpad-data` and set it to **Private**.
3. **Import the notebook:** go to *Code → New Notebook*, then *File → Import Notebook*, and pick `D:\AIS\IPI\kaggle\2_exp2_finetune_tinynet.ipynb`.
4. **Attach the data:** in the right sidebar, click *Input → Add Input*, then *Your Datasets*, and click **+** next to **fvpad-data**.
5. **Session options** (right sidebar):
   - *Accelerator*: **GPU T4 x2** (or **GPU P100**). The notebook uses one GPU, which is enough.
   - *Internet*: **On**. This is needed to download the TinyNet ImageNet weights, and `timm` if it is missing.
6. **Smoke test first (about 3 min):**
   - In cell *1. Configuration*, set `QUICK = True`.
   - Click *Run All*.
   - Check the outputs:
     - **Cell 0** prints the GPU name, `cuda True` and the `DATA = /kaggle/input/...` path.
     - **Cell 3** prints the trainable parameters per mode. `full` should show 100 %.
     - **Cell 4** prints one line per run, for example `IDIAP BA f1 s0 full ep=3/3 Δw=0.0123 | kNN ACER=... | head ACER=...`.
     - **Section 6** shows `backbone_rel_change`: 0 for `frozen`, > 0 for the others.
7. **Full run:**
   - Set `QUICK = False`.
   - Delete the smoke-test CSVs, either with a new cell containing `!rm -f /kaggle/working/tinynet_*.csv` or by restarting the session.
   - Click *Save Version → Save & Run All (Commit)*. This runs headless, so you can close the browser.
   - The default grid is 3 datasets × 3 sizes × 5 folds × 1 seed × 3 modes = **135 runs**, which takes about **1.5–3 h** on a T4. The Kaggle limit is 12 h per session.
8. **Download the results:**
   - Open the finished version and go to the *Output* tab.
   - Download these files into `D:\AIS\IPI\results`:
     - `tinynet_runs.csv`
     - `tinynet_summary.csv`
     - `tinynet_predictions.csv`

## Output files

| File | Content |
|---|---|
| `tinynet_runs.csv` | One row per run: key, mode, dataset, size, fold, seed, train folds, trainable params, `backbone_rel_change`, best epoch, k, **k-NN** metrics (`knn_APCER/BPCER/ACER/EER/AUC/BPCER20/dprime`), **head** metrics (`head_...`), seconds, and the loss history per epoch |
| `tinynet_summary.csv` | Mean and std per model × variant × dataset × mode × size |
| `tinynet_predictions.csv` | Per test image: run key, path, label, head probability, k-NN score and prediction (use it for ROC/DET curves) |

Metrics follow ISO/IEC 30107-3, with attack as the positive class. All values are in %, except `dprime`.

## Useful settings (cell 1)

| Variable | Meaning |
|---|---|
| `MODEL` | `tinynet_e` (default, fastest) … `tinynet_a` |
| `MODES` | e.g. `['full']` to run only full fine-tuning (about 2× faster) |
| `DATASETS`, `SIZES`, `FOLDS`, `SEEDS` | restrict or extend the grid, e.g. `SEEDS = [0, 1, 2]` for 3 seeds |
| `EPOCHS`, `PATIENCE` | max epochs (default 20) and early-stopping patience (default 5) |
| `LR_BACKBONE`, `LR_HEAD` | learning rates (default 1e-4 and 1e-3) |
| `NORM = 'zscore'` | removes the global brightness cue, which makes the task harder and shows differences between the modes |
| `PLUS_MASK_LEFT = 53` | blacks out the printed label on PLUS images |

**Tip:** Experiment 1 already reached about 0 % ACER with frozen features, so fine-tuning cannot improve ACER much under the default settings. The differences show up in **S1/S2**, in **EER / d′**, and especially with `NORM = 'zscore'` and `PLUS_MASK_LEFT = 53`. Each setting gets its own `variant` in the CSV, so you can run them one after another into the same files.

## Troubleshooting

- **`metadata.csv not found`:** the dataset is not attached. Repeat step 4.
- **Weight download fails or `timm` is missing:** Internet is off. Repeat step 5.
- **`cuda False`:** no GPU is selected, or your weekly GPU quota (30 h) is used up.
- **Session stopped mid-run:** run cell 4 again. Finished runs (by `key`) are skipped and the CSVs are appended to.
- **CUDA out of memory:** set `BATCH = 16`.
