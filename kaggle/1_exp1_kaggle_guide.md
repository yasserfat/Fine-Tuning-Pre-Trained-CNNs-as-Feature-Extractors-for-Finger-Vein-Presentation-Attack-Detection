# Experiment 1 (done): frozen CNN baseline on Kaggle

Status: **done**. Results are in `../results/` (`results_exp1.jsonl`, `summary_exp1.csv`, `Experiment1_Report.pdf`). The next step is Experiment 2, see `2_exp2_kaggle_guide.md`.

## Files in this folder

| File | What it is |
|---|---|
| `fvpad_data.zip` (216 MB, not in git) | **Upload this to Kaggle.** 8,866 resized images (height 160 px) + `metadata.csv` with labels and subject-disjoint folds |
| `fvpad_data/` (not in git) | The same content unzipped (for local use) |
| `1_exp1_frozen_baseline.ipynb` | **Import this into Kaggle.** Runs Experiment 1 (pipeline code embedded) |
| `extra_exp1b_controls.ipynb` | Optional extra controls for Experiment 1 (not run yet) |
| `pad.py` | The pipeline (baseline / finetune / summary). Identical to the code inside the notebook |
| `2_exp2_finetune_tinynet.ipynb` | **Experiment 2:** real (end-to-end) fine-tuning of TinyNet, writes result CSVs. See `2_exp2_kaggle_guide.md` |
| `0_prepare_data.py` | Rebuilds `fvpad_data.zip` from the raw course data (already done, no need to re-run) |

Data in the zip (balanced pairs only, per course rule):

- IDIAP: 440 + 440 images, 110 subjects, 22 per fold.
- SCUT: 3,600 + 3,600 images, 100 subjects, 20 per fold.
- PLUS: 393 + 393 images, 22 subjects, 5/5/4/4/4 per fold.

## One-time setup (≈10 min)

1. **Account:** log in at kaggle.com. Go to *Settings → Phone verification* and verify. Without it, GPU and Internet stay disabled in notebooks.
2. **Upload the data:**
   - Go to *Datasets → New Dataset*, then drag in `D:\AIS\IPI\kaggle\fvpad_data.zip`.
   - Title: `fvpad-data`.
   - Visibility: **Private** (the biometric data is licensed: do not make it public).
   - Click *Create*.
   - Kaggle unzips the archive automatically. Wait until processing finishes.
3. **Import the notebook:**
   - Go to *Code → New Notebook*, then *File → Import Notebook*, and pick `D:\AIS\IPI\kaggle\1_exp1_frozen_baseline.ipynb`.
4. **Attach the data:**
   - In the right sidebar, click *Input → Add Input*, search for **fvpad-data** (under *Your Datasets*), and click **+**.
5. **Notebook settings** (right sidebar → *Session options*):
   - *Accelerator*: **GPU T4 x2** (or GPU P100).
   - *Internet*: **On** (needed to download the ImageNet weights).
   - *Persistence*: Files (optional).

## Run Experiment 1

6. Run the cells from top to bottom (*Run All*). Check the output of each:
   - **Cell 1:** it should print the GPU name, `DATA = /kaggle/input/...`, and `8866 images`.
   - **Cell 2** (`%%writefile pad.py`): it should print `Writing pad.py`.
   - **"Run Experiment 1":** prints one line per run, for example `IDIAP mobilenet_v2-L0 BA fold1 seed0 k= 1 APCER=… BPCER=… ACER=…`. That is 405 runs; expect about 10–20 min. If the session stops, re-run the cell: finished runs are skipped.
   - **Summary cells:** a table of mean ± std per dataset × method × size, plus the ACER pivot tables.
7. *Optional:* run the PLUS label-mask cell (section 4) and the fine-tuning smoke test (section 5). The smoke test tells you the real GPU minutes per fine-tuning run. Write that number down: it is needed to plan Experiment 2.
8. **Save:**
   - Click *Save Version → Save & Run All (Commit)*. This re-runs everything headless and freezes the outputs with the notebook version, which makes the run reproducible.
   - When it finishes, open the version, go to the *Output* tab, and download `results_exp1.jsonl` and `summary_exp1.csv`.

## What to look at in the results

- **IDIAP / PLUS near 0 % ACER for the frozen CNN at BA:** this is the ceiling effect predicted in the proposal. The S1/S2 columns and SCUT are where the methods differ.
- **`hist10` being already very good on PLUS** means a simple brightness cue separates the classes. Mention this in the paper: it motivates the label-mask control.
- **`k`** (selected per run) and **`train_folds`** are logged in the `.jsonl` file for every run, which you need to describe the protocol exactly.

## Optional: Experiment 1b (not run yet)

Optional extra controls. Experiment 1 hit the ceiling: frozen CNNs reached 0–0.2 % ACER. `pad.py` version 2 adds score metrics (EER, AUC, BPCER20, d′), `--norm zscore|equalize` (a brightness-cue control) and `--mask-mode test` (the PLUS shortcut test).

1. In Kaggle, open *Code → New Notebook*, then *File → Import Notebook*, and pick `extra_exp1b_controls.ipynb`.
2. Use the same settings as before:
   - attach **fvpad-data** via *Add Input*;
   - *Accelerator*: **GPU T4 x2**;
   - *Internet*: **On**.
3. Click *Run All*. It takes about 1–1.5 h.
4. Click *Save Version → Save & Run All*.
5. From the *Output* tab, download `results_exp1b.jsonl`, `summary_exp1b.csv` and `results_timing.jsonl` into `D:\AIS\IPI\results`.

## Next: Experiment 2 (real fine-tuning)

See `2_exp2_kaggle_guide.md` and the notebook `2_exp2_finetune_tinynet.ipynb`.
