# Experiment 2 on Kaggle: real fine-tuning with TinyNet

**This guide is for group members running the notebook for the first time.** You don't need a GPU on your own PC: everything runs on Kaggle for free. Plan about 20 min for the setup and the quick test, then 1.5–3 h of unattended running.

Notebook: [`2_exp2_finetune_tinynet.ipynb`](2_exp2_finetune_tinynet.ipynb). All the code is inside the notebook, so you don't need any other file from the repo.

## What the experiment does

In Experiment 1 (already done, results in [`../results/`](../results/)) the CNN stayed **frozen**: its weights were never trained, and k-NN was run on its features. This notebook **trains the network weights** with backpropagation. It compares three modes on the same subject-disjoint splits as Experiment 1:

| Mode | What is trained | Trainable params (tinynet_e) |
|---|---|---|
| `frozen` | classifier head only (reference) | ~0.33 M (head) |
| `partial` | head + last 2 stages + final conv | ~0.95 M |
| `full` | **all layers**, BatchNorm re-estimated | ~1.09 M (all) |

Each run logs `backbone_rel_change` = ||θ_after − θ_before|| / ||θ_before||. It is exactly **0** for `frozen` and **> 0** for `partial`/`full`, which proves the backbone was actually fine-tuned.

The model is **TinyNet** from the `timm` library. `tinynet_e` is the smallest and fastest version.

---

## Part A: one-time setup

### A1. Create and verify a Kaggle account
1. Sign up at [kaggle.com](https://www.kaggle.com).
2. Go to *your profile picture → Settings → Phone verification* and verify your number. **Without this, GPU and Internet are disabled**, and the notebook cannot run.

### A2. Get access to the dataset `fvpad-data`
The images are licensed course data, so they are **not on GitHub**. They are in a **private** Kaggle dataset owned by the repo owner (`yasserfat`).

1. Send the repo owner your **Kaggle username** (shown on your Kaggle profile page).
2. The owner opens the dataset and goes to *Settings → Sharing*, adds your username, and clicks *Save*.
3. You'll see the dataset under *Datasets → Your Work → Shared with you*.

*Only if you can't get access:* you can build the dataset yourself from the raw course data (`Data_FV_Spoofing_WS2025_26`, from the course page):
```
git clone https://github.com/yasserfat/Fine-Tuning-Pre-Trained-CNNs-as-Feature-Extractors-for-Finger-Vein-Presentation-Attack-Detection.git
cd Fine-Tuning-Pre-Trained-CNNs-as-Feature-Extractors-for-Finger-Vein-Presentation-Attack-Detection
pip install numpy pandas pillow
python kaggle/0_prepare_data.py --raw "<path>/Data_FV_Spoofing_WS2025_26/Data_FV_Spoofing_WS2025_26" --out kaggle/fvpad_data
```
This creates `kaggle/fvpad_data.zip` (about 216 MB). Upload it on Kaggle with *Datasets → New Dataset*, name it `fvpad-data`, and set it to **Private**. Never make it public.

### A3. Download the notebook
1. On GitHub, open `kaggle/2_exp2_finetune_tinynet.ipynb`.
2. Click the **Download raw file** button (↓ icon, top right of the file view). If you cloned the repo, the file is already on your PC.

---

## Part B: run the notebook on Kaggle

1. **Import:** on Kaggle, go to *Code → New Notebook*, then *File → Import Notebook*, and pick the downloaded `2_exp2_finetune_tinynet.ipynb`.
2. **Attach the data:** in the right sidebar, click *Input → Add Input*, search for **fvpad-data** (look under *Shared with you* or *Your Datasets*), and click **+**.
3. **Session options** (right sidebar, or the ⋮ menu → *Accelerator*):
   - *Accelerator*: **GPU T4 x2** (or **GPU P100**).
   - *Internet*: **On**. This is needed to download the pre-trained TinyNet weights.
4. **Quick test first (about 3 min):**
   - In the cell *1. Configuration*, change `QUICK = False` to `QUICK = True`.
   - Click *Run All* at the top.
   - Check the output:
     - **Cell 0** shows a GPU name (e.g. Tesla T4), `cuda True` and `DATA = /kaggle/input/...`.
     - **Section 3** prints the trainable parameters per mode. `full` should show 100 %.
     - **Section 4** prints one line per run, e.g. `IDIAP BA f1 s0 full ep=3/3 Δw=0.0123 | kNN ACER=... | head ACER=...`.
     - **Section 6:** `backbone_rel_change` is 0 for `frozen` and > 0 for the others.
   - If something fails, see *Troubleshooting* below.
5. **Full run:**
   - Set `QUICK = False` again.
   - The quick test wrote `tinynet_quicktest_*.csv`, and the real run writes `tinynet_*.csv`, so nothing needs to be deleted.
   - Click **Save Version → Save & Run All (Commit) → Save**. Kaggle now runs everything in the background, so you can close the browser.
   - The default grid is 3 datasets × 3 sizes × 5 folds × 1 seed × 3 modes = **135 runs**, about **1.5–3 h** on a T4. Free accounts get 30 GPU hours per week.
6. **Get the results:**
   - Go to *Code → Your Work*, open the notebook, and select the finished version.
   - On the **Output** tab, download:
     - `tinynet_runs.csv`
     - `tinynet_summary.csv`
     - `tinynet_predictions.csv`

---

## Part C: put the results on GitHub

You need **write access** to the repo. Ask the owner to add you under *Settings → Collaborators* on GitHub.

1. Clone the repo (once), or update your copy:
   ```
   git clone https://github.com/yasserfat/Fine-Tuning-Pre-Trained-CNNs-as-Feature-Extractors-for-Finger-Vein-Presentation-Attack-Detection.git
   cd Fine-Tuning-Pre-Trained-CNNs-as-Feature-Extractors-for-Finger-Vein-Presentation-Attack-Detection
   git pull
   ```
2. Copy the three CSVs into a folder named after the setting you ran, so that results from different people don't overwrite each other:
   - `results/exp2/default/` for the default settings
   - `results/exp2/zscore/` for `NORM = 'zscore'`
   - `results/exp2/zscore_mask53/` for `NORM = 'zscore'` + `PLUS_MASK_LEFT = 53`
3. Commit and push:
   ```
   git add results/exp2
   git commit -m "Exp2 results: tinynet_e, default settings"
   git push
   ```
4. Tell the group in your chat which setting you ran, so nobody runs the same thing twice.

---

## Output files

| File | Content |
|---|---|
| `tinynet_runs.csv` | One row per run: key, mode, dataset, size, fold, seed, train folds, trainable params, `backbone_rel_change`, best epoch, k, **k-NN** metrics (`knn_APCER/BPCER/ACER/EER/AUC/BPCER20/dprime`), **head** metrics (`head_...`), seconds, and the loss history per epoch |
| `tinynet_summary.csv` | Mean and std per model × variant × dataset × mode × size. **Use this one for the report tables** |
| `tinynet_predictions.csv` | Per test image: run key, path, label, head probability, k-NN score and prediction (for ROC/DET curves) |

The metrics follow ISO/IEC 30107-3, with attack as the positive class. All values are in %, except `dprime`. Lower APCER/BPCER/ACER/EER is better. Higher AUC and d′ are better.

## Settings you can change (cell 1)

| Variable | Meaning |
|---|---|
| `MODEL` | `tinynet_e` (default, fastest) … `tinynet_a` (largest) |
| `MODES` | e.g. `['full']` to run only full fine-tuning (faster) |
| `DATASETS`, `SIZES`, `FOLDS`, `SEEDS` | shrink or extend the grid, e.g. `SEEDS = [0, 1, 2]` for 3 seeds (3× time) |
| `EPOCHS`, `PATIENCE` | max epochs (default 20) and early-stopping patience (default 5) |
| `LR_BACKBONE`, `LR_HEAD` | learning rates (default 1e-4 and 1e-3) |
| `NORM = 'zscore'` | removes the global brightness difference, which makes the task harder |
| `PLUS_MASK_LEFT = 53` | blacks out the printed label on PLUS images |

**Why the extra settings matter:** Experiment 1 already reached about 0 % ACER with frozen features, so with the default settings fine-tuning cannot improve ACER much. The differences between the modes show up at the small training sizes (**S1/S2**), in **EER / d′**, and especially with `NORM = 'zscore'` and `PLUS_MASK_LEFT = 53`. A good split of work is one person per setting (see Part C, step 2).

## Troubleshooting

| Problem | Fix |
|---|---|
| `metadata.csv not found` | The dataset is not attached: repeat B2. If you can't find it, the owner hasn't shared it yet (A2) |
| Weight download fails / `timm` error | Internet is off: repeat B3. Not possible? Your phone isn't verified (A1) |
| `cuda False` or very slow | No GPU selected (B3), or your weekly GPU quota is used up |
| Session stopped in the middle | Run the cell in section 4 again. Finished runs are skipped |
| `CUDA out of memory` | Set `BATCH = 16` in cell 1 |
| `git push` rejected | Run `git pull` first, then push again. "Permission denied" means you're not a collaborator yet (Part C) |
