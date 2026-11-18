# Finger-Vein Presentation Attack Detection: fine-tuning pre-trained CNNs

Group project (IPI, WS 2025/26). Datasets: IDIAP, SCUT and PLUS finger-vein spoofing data. Task: separate bona fide finger-vein images from presentation attacks (PAI). Metrics follow ISO/IEC 30107-3: APCER, BPCER, ACER, EER.

## Status: what is done and what comes next

| # | Step | Status | Files |
|---|---|---|---|
| 0 | Build the Kaggle dataset (resize, subject-disjoint 5 folds) | ✅ done | `kaggle/0_prepare_data.py` |
| 1 | **Experiment 1**: frozen ImageNet CNNs (MobileNetV2, ResNet18) + k-NN, histogram baseline | ✅ done | `kaggle/1_exp1_frozen_baseline.ipynb`, `kaggle/1_exp1_kaggle_guide.md`, `results/` |
| 2 | **Experiment 2**: **real fine-tuning** of TinyNet (frozen vs. partial vs. full), results as CSV, as the teacher asked | ⏭ **next, to do** | `kaggle/2_exp2_finetune_tinynet.ipynb`, `kaggle/2_exp2_kaggle_guide.md` |
| – | Experiment 1b: extra controls (score metrics, brightness normalisation, PLUS label test) | optional, not run | `kaggle/extra_exp1b_controls.ipynb` |

**Why Experiment 2:** In Experiment 1 the CNN weights were never trained. The frozen features already reached about 0 % ACER, so it shows nothing about fine-tuning. Experiment 2 trains the network weights (up to all layers) and logs `backbone_rel_change`, which proves the backbone actually changed.

## How to run Experiment 2 (short version)

Full instructions: [`kaggle/2_exp2_kaggle_guide.md`](kaggle/2_exp2_kaggle_guide.md).

1. Kaggle: *New Notebook → File → Import Notebook* and pick `kaggle/2_exp2_finetune_tinynet.ipynb`.
2. *Add Input*: **fvpad-data** (the private dataset; ask the group owner for access, or build it with step 0).
3. Session options: *Accelerator* **GPU T4** or **P100**, *Internet* **On**.
4. First set `QUICK = True` and click *Run All* (about 3 min) as a check. Then set `QUICK = False` and click *Save Version → Save & Run All* (about 1.5–3 h).
5. Download `tinynet_runs.csv`, `tinynet_summary.csv` and `tinynet_predictions.csv` from the *Output* tab into `results/`, then commit them.

## Repository layout

```
README.md                         <- you are here
Project_Proposal_FingerVein_PAD_FineTuning.pdf
kaggle/
  0_prepare_data.py               raw course data -> fvpad_data/ + fvpad_data.zip (Kaggle upload)
  1_exp1_frozen_baseline.ipynb    Experiment 1 (done)
  1_exp1_kaggle_guide.md          Kaggle setup + Experiment 1 steps
  2_exp2_finetune_tinynet.ipynb   Experiment 2 (next): real fine-tuning, writes CSVs
  2_exp2_kaggle_guide.md          step-by-step Kaggle guide for Experiment 2
  extra_exp1b_controls.ipynb      optional controls
  pad.py                          pipeline used by the Experiment 1 / 1b notebooks
results/
  results_exp1.jsonl, summary_exp1.csv, Experiment1_Report.pdf
```

## Data and licence

The biometric data is licensed for the course only and is **not in this repository** (see `.gitignore`). This covers `Data_FV_Spoofing_WS2025_26/`, `kaggle/fvpad_data/` and `fvpad_data.zip`. Third-party papers (`papers/`) and course slides are also excluded. Keep the Kaggle dataset **private**.

To rebuild the data locally:

```
python kaggle/0_prepare_data.py --raw "<path>/Data_FV_Spoofing_WS2025_26/Data_FV_Spoofing_WS2025_26" --out kaggle/fvpad_data
```
