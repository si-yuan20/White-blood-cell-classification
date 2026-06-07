# Edge-prior and reliability-guided collaborative learning for white blood cell classification

This repository provides the implementation resources for the study:

**Edge-prior and reliability-guided collaborative learning for white blood cell classification**

The project implements a dual-stream collaborative learning framework for image-level white blood cell (WBC) classification on public benchmark datasets. The framework combines ConvNeXt-based local morphological representation with Mamba-based contextual modeling, and integrates the two branches through Structure-Aided Attention Fusion (SAF) and reliability-guided bilateral fusion (RGBF).

---

## Overview

White blood cell subtype classification is challenging because different leukocyte categories may share similar nuclear contours, cytoplasmic texture, staining appearance, and cell boundary patterns. This project investigates a collaborative framework that coordinates local and global representations for fine-grained WBC image classification.

The framework contains:

- **ConvNeXt branch** for local morphological feature extraction, including nuclear boundaries, chromatin texture, and cytoplasmic granularity.
- **Mamba-based branch** for long-range contextual representation.
- **Structure-Aided Attention Fusion (SAF)** for edge-prior-guided interaction between heterogeneous branch features.
- **Reliability-guided bilateral fusion (RGBF)** for adaptive branch weighting using confidence-derived reliability cues.
- **Classification head** for five-class WBC subtype prediction.

The confidence-derived reliability cues include predictive entropy, maximum class probability, and classification margin. These quantities are used only for branch weighting and are **not** treated as calibrated or Bayesian uncertainty estimates.

---

## Paper

If you use this repository, please cite our paper:

> Rong Gao, Qi Ke, Aiquan Li, Xinning Qin, Sichao Zhao.
> **Edge-prior and reliability-guided collaborative learning for white blood cell classification**.
> Under review.

The official citation and DOI will be updated after publication.

---

## Repository structure

```text
White-blood-cell-classification/
├── config.py                         # Training/model/data configuration
├── data_prepare.py                   # Dataset loading, transforms, split utilities
├── dual_model.py                     # All model components (ConvNeXt, Mamba, SAF, RGBF, dual-stream)
├── main.py                           # Training entry point
├── evaluate.py                       # Evaluation entry point (with per-class metrics)
├── utils.py                          # Metrics, visualization, seed utilities
├── requirements.txt                  # Python dependencies
├── evaluation/
│   ├── cross_dataset_eval.py         # Cross-dataset transfer evaluation
│   ├── robustness_eval.py            # Salt-and-pepper noise robustness evaluation
│   └── eval_ablation.py              # Ablation study evaluation (Table 7)
├── visualization/
│   ├── grad_cam.py                   # Grad-CAM heatmap interpretability
│   └── quantitative_interpretability.py  # Quantitative interpretability metrics (Table 14)
├── tools/
│   ├── benchmark_model.py            # Params/FLOPs/inference time measurement
│   └── noise.py                      # Salt-and-pepper noise generation
├── scripts/
│   ├── train_full.sh                 # Train full proposed framework
│   ├── train_ablation.sh             # Train all ablation variants
│   ├── train_mobilevit.sh            # Train MobileViT baseline
│   ├── train_davit.sh                # Train DaViT baseline
│   ├── eval_cross_dataset.sh         # Cross-dataset transfer evaluation
│   └── eval_robustness.sh            # Robustness evaluation
├── docs/
│   └── complexity_notes.md           # Theoretical complexity analysis (Table 13)
├── splits/                           # Saved dataset split files (generated)
├── logs/                             # Training logs (generated)
├── results/                          # Experimental results and visual outputs (generated)
└── README.md
```

---

## Datasets

This study uses three publicly available WBC image datasets:

| Dataset    | Description                            | Source                                                         |
| ---------- | -------------------------------------- | -------------------------------------------------------------- |
| PBC        | Peripheral blood cell image dataset    | https://www.kaggle.com/datasets/bzhbzh35/peripheral-blood-cell |
| LDWBC      | Large-scale WBC image dataset          | https://biod.whu.edu.cn/sjj.htm                                |
| Raabin-WBC | WBC dataset with cell type annotations | https://www.kaggle.com/datasets/raabindata/raabin-wbc          |

Raw images are **not redistributed** in this repository because of dataset licenses and source-specific usage conditions. Users should download the datasets from the official sources.

---

## Task setting

The experiments focus on five WBC subtypes shared across the three datasets:

- Basophil
- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil

Images are resized to `224 x 224` and normalized before training. Data augmentation is applied only to the training subset.

The experiments use image-level stratified splits because complete patient-level identifiers are not available for all public datasets. Therefore, the reported results should be interpreted as **image-level public benchmark performance**, not as patient-level clinical validation.

---

## Main experimental settings

| Item                  | Setting                   |
| --------------------- | ------------------------- |
| Input size            | 224 x 224                 |
| Optimizer             | AdamW                     |
| Initial learning rate | 3e-4                      |
| Weight decay          | 0.003                     |
| Scheduler             | Cosine annealing          |
| Minimum learning rate | 1e-6                      |
| Batch size            | 64                        |
| Training epochs       | 200                       |
| Precision strategy    | Automatic mixed precision |
| Main GPU              | NVIDIA RTX 3090           |

The Mamba branch uses a patch size of `16 x 16`, resulting in `14 x 14 = 196` visual tokens for each input image. Each token is projected into a 512-dimensional embedding space, and four Mamba blocks are used in the final configuration.

---

## Installation

Create a Python environment and install the required dependencies:

```bash
conda create -n wbc_classification python=3.10 -y
conda activate wbc_classification

pip install -r requirements.txt
```

**Note for Mamba-SSM**: The `mamba-ssm` package requires CUDA toolkit and a compatible PyTorch/CUDA combination. Install separately:

```bash
pip install mamba-ssm
```

If mamba-ssm is unavailable, ConvNeXt-only and timm-based baselines (mobilevit, davit) can still be trained and evaluated. Mamba-related models require mamba-ssm.

See: https://github.com/state-spaces/mamba

---

## Data preparation

After downloading the datasets from their official sources, organize the data in the following format:

```text
dataset_root/
├── Basophil/
├── Eosinophil/
├── Lymphocyte/
├── Monocyte/
└── Neutrophil/
```

The dataset split files are automatically generated and saved to `splits/` on first run. Example:

```text
splits/PBC_seed42.json
splits/LDWBC_seed42.json
splits/Raabin-WBC_seed42.json
```

To reuse a previously saved split, the split file is automatically loaded during training when present.

---

## Training

### Full proposed framework

```bash
python main.py \
  --data_dir /path/to/processed_dataset \
  --model_name dual_full \
  --epochs 200 \
  --batch_size 64 \
  --lr 3e-4 \
  --seed 42
```

Or use the script:

```bash
DATA_DIR=/path/to/dataset bash scripts/train_full.sh
```

### Supported model variants

| CLI argument      | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `convnext`        | ConvNeXt-only baseline                               |
| `mamba`           | Mamba-only baseline                                  |
| `dual_naive`      | ConvNeXt+Mamba naive concat (SAF off, RGBF off)      |
| `dual_saf`        | ConvNeXt+Mamba+SAF (RGBF off)                        |
| `dual_rgbf`       | ConvNeXt+Mamba+RGBF (SAF off)                        |
| `dual_full`       | ConvNeXt+Mamba+SAF+RGBF (full framework)             |
| `mobilevit`       | MobileViT baseline (requires timm)                   |
| `davit`           | DaViT baseline (requires timm)                       |

### Ablation training

```bash
bash scripts/train_ablation.sh /path/to/dataset 42
```

### MobileViT / DaViT baselines

```bash
bash scripts/train_mobilevit.sh /path/to/dataset 42
bash scripts/train_davit.sh /path/to/dataset 42
```

The training logs are saved in `results/<dataset>/<model_name>/logs/`.

---

## Evaluation

### Single-dataset evaluation (with per-class metrics)

```bash
python evaluate.py \
  --data_dir /path/to/dataset \
  --checkpoint /path/to/best_model.pth \
  --model_name dual_full \
  --output_dir results/eval
```

The evaluation reports: Accuracy, Precision, Recall, Macro-F1, AUC, Confusion matrix, ROC curves, and **per-class Precision/Recall/F1**.

Per-class results are saved to `results/classwise/<dataset>_classwise_metrics.csv`.

### Ablation evaluation (Table 7)

Evaluate all ablation variants with a single command:

```bash
python evaluation/eval_ablation.py \
  --dataset_name PBC \
  --data_dir /path/to/PBC \
  --checkpoint_dir models \
  --output_dir results/ablation/PBC
```

Alternatively, provide a JSON mapping of variant → checkpoint path:

```bash
python evaluation/eval_ablation.py \
  --dataset_name PBC \
  --data_dir /path/to/PBC \
  --checkpoints_json checkpoints/PBC/ablation_checkpoints.json \
  --output_dir results/ablation/PBC
```

Outputs: CSV and JSON with all metrics needed for Table 7.

### Cross-dataset transfer evaluation (without target-domain fine-tuning)

Evaluates a model trained on a source dataset directly on a target dataset:

```bash
python evaluation/cross_dataset_eval.py \
  --source_name PBC \
  --target_name LDWBC \
  --target_dir /path/to/LDWBC \
  --checkpoint /path/to/pbc_best.pth \
  --model_name dual_full \
  --output_dir results/cross_dataset/PBC_to_LDWBC
```

Supports six transfer directions: PBC↔LDWBC, PBC↔Raabin-WBC, LDWBC↔Raabin-WBC.

Run all directions:

```bash
bash scripts/eval_cross_dataset.sh /path/to/checkpoints /path/to/PBC /path/to/LDWBC /path/to/Raabin-WBC
```

### Salt-and-pepper noise robustness evaluation

```bash
python evaluation/robustness_eval.py \
  --data_dir /path/to/dataset \
  --checkpoint /path/to/best_model.pth \
  --model_name dual_full \
  --noise_levels 0.0 0.01 0.05 0.10 \
  --output_dir results/robustness
```

Or:

```bash
bash scripts/eval_robustness.sh /path/to/dataset /path/to/checkpoint.pth
```

Outputs per-noise-level Accuracy, Macro-F1, AUC, Macro-F1 drop, Accuracy drop, per-class F1, robustness bar plot, and noisy example images.

---

## Interpretability

### Grad-CAM heatmap visualization

Generate Grad-CAM heatmaps:

```bash
python visualization/grad_cam.py \
  --data_dir /path/to/dataset \
  --checkpoint /path/to/best_model.pth \
  --model_name dual_full \
  --target_layer auto \
  --max_samples 20 \
  --output_dir results/grad_cam
```

Supports target layer selection: `auto`, `convnext_last`, `saf_out`. Outputs original image, raw heatmap, and overlay for each sample.

### Quantitative interpretability (Table 14)

Compute quantitative interpretability metrics (foreground activation ratio, background activation ratio, pointing accuracy, deletion score):

```bash
python visualization/quantitative_interpretability.py \
  --data_dir /path/to/PBC \
  --checkpoint /path/to/best.pth \
  --model_name dual_full \
  --target_layer saf \
  --max_samples 100 \
  --output_dir results/interpretability/PBC_dual_full
```

To compare ConvNeXt-only baseline:

```bash
python visualization/quantitative_interpretability.py \
  --data_dir /path/to/PBC \
  --checkpoint /path/to/convnext_best.pth \
  --model_name convnext \
  --target_layer convnext_last \
  --max_samples 100 \
  --output_dir results/interpretability/PBC_convnext
```

For quick dry-run testing without real checkpoints:

```bash
python visualization/quantitative_interpretability.py \
  --data_dir /path/to/PBC \
  --model_name dual_full \
  --dry_run \
  --max_samples 10
```

Outputs: per-sample JSON/CSV, summary JSON, and Table 14-format CSV with all four metrics (mean ± std).

**Note:** Foreground masks are generated automatically via Otsu thresholding with morphological post-processing. These masks are intended for computational interpretability assessment only — **not** as expert pathological annotation.

---

## Computational complexity

### Empirical measurement (Table 12)

Measure parameters, FLOPs, and inference time:

```bash
# All models
python tools/benchmark_model.py --model_name all

# Single model
python tools/benchmark_model.py --model_name dual_full

# Dry-run (quick test)
python tools/benchmark_model.py --model_name all --dry_run
```

Results saved to `results/complexity/complexity_results.csv`. Includes model name, params (M), trainable params (M), FLOPs (G), inference time (ms), and GPU name.

### Theoretical complexity (Table 13)

See [docs/complexity_notes.md](docs/complexity_notes.md) for the theoretical Big-O analysis of each component.

---

## Reproducibility

The revised manuscript reports experiments on PBC, LDWBC, and Raabin-WBC. The main analyses include:

- within-dataset classification performance (Table 5)
- class-wise performance evaluation (Tables 8-10)
- controlled comparison with MobileViT and DaViT (Table 6)
- ablation analysis of ConvNeXt, Mamba, SAF, and reliability-guided bilateral fusion (Table 7)
- cross-dataset transfer without target-domain fine-tuning (Table 11)
- training and validation curves (Fig. 12)
- salt-and-pepper noise robustness analysis (Figs. 15-16)
- computational complexity analysis — params, FLOPs, inference time (Table 12)
- theoretical complexity analysis (Table 13)
- heatmap-based interpretability analysis — Grad-CAM (Fig. 17)
- quantitative interpretability metrics (Table 14)

For reproducibility:

1. Use the provided configuration file (`config.py`)
2. Keep the same image-level split protocol (splits auto-saved to `splits/`)
3. Use fixed seeds: `--seed 42` (also supported: 2024, 2025, 2026)
4. Deterministic mode is enabled by default
5. All commands listed above are designed to be executable

---

## Notes on interpretation

The reported results are based on public image-level WBC datasets. Complete patient-level identifiers are not available for all datasets, so strict patient-level partitioning could not be performed.

The current results should therefore be interpreted as **public benchmark-level evidence for image-level WBC classification**. They should not be interpreted as patient-level clinical validation or evidence of clinical readiness.

Grad-CAM heatmaps and reliability cues are computed automatically from model outputs. Any quantitative interpretability metrics (e.g., foreground activation ratio, deletion score) are based on automatically generated foreground masks and are intended for computational assessment only — not as expert pathological annotation.

Cross-dataset transfer evaluation is performed **without target-domain fine-tuning**. It demonstrates model generalization across datasets but does not constitute clinical external validation.

Future work should include:

- patient-level multi-center validation
- prospective evaluation under realistic laboratory workflows
- calibrated reliability estimation
- broader robustness analysis
- lightweight model design for resource-constrained deployment
- expert-reviewed quantitative interpretability assessment

---

## Self-test

Verify that all model variants can be instantiated and forward-pass:

```bash
python dual_model.py
```

This runs a self-test that creates all 8 model variants with random inputs and verifies output shapes.

---

## Acknowledgments

We thank the contributors of the PBC, LDWBC, and Raabin-WBC datasets for making these valuable resources publicly available. These datasets provide an important foundation for the development and evaluation of automated WBC image classification methods.

---

## License

Please check the licenses of the original datasets before use. This repository only provides code and related experimental resources. Raw dataset files are not redistributed.
