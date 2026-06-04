# Edge-prior and reliability-guided collaborative learning for white blood cell classification

This repository provides the implementation resources for the study:

**Edge-prior and reliability-guided collaborative learning for white blood cell classification**

The project implements a dual-stream collaborative learning framework for image-level white blood cell (WBC) classification on public benchmark datasets. The framework combines ConvNeXt-based local morphological representation with Mamba-based contextual modeling, and integrates the two branches through Structure-Aided Attention Fusion (SAF) and reliability-guided bilateral fusion.

The current repository is intended to support reproducibility of the main experiments, including model training, evaluation, ablation analysis, and result visualization.

---

## Overview

White blood cell subtype classification is challenging because different leukocyte categories may share similar nuclear contours, cytoplasmic texture, staining appearance, and cell boundary patterns. This project investigates a collaborative framework that coordinates local and global representations for fine-grained WBC image classification.

The framework contains:

* **ConvNeXt branch** for local morphological feature extraction, including nuclear boundaries, chromatin texture, and cytoplasmic granularity.
* **Mamba-based branch** for long-range contextual representation.
* **Structure-Aided Attention Fusion (SAF)** for edge-prior-guided interaction between heterogeneous branch features.
* **Reliability-guided bilateral fusion** for adaptive branch weighting using confidence-derived reliability cues.
* **Classification head** for five-class WBC subtype prediction.

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
├── datasets/              # Dataset description and split information
├── logs/                  # Training logs
├── results/               # Experimental results and visual outputs
├── README.md              # Project documentation
├── config.py              # Configuration file
├── data_prepare.py        # Data preprocessing and dataset preparation
├── dual_model.py          # Dual-stream model implementation
├── main.py                # Training entry
├── test.py                # Evaluation entry
├── utils.py               # Utility functions
└── requirements.txt       # Python dependencies
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

This repository provides or will maintain:

* dataset preparation scripts;
* preprocessing instructions;
* image-level train/validation/test split information;
* evaluation scripts for reproducing the reported results.

---

## Task setting

The experiments focus on five WBC subtypes shared across the three datasets:

* Basophil
* Eosinophil
* Lymphocyte
* Monocyte
* Neutrophil

Images are resized to `224 × 224` and normalized before training. Data augmentation is applied only to the training subset.

The experiments use image-level stratified splits because complete patient-level identifiers are not available for all public datasets. Therefore, the reported results should be interpreted as **image-level public benchmark performance**, not as patient-level clinical validation.

---

## Main experimental settings

The main training configuration used in the study is summarized below:

| Item                  | Setting                   |
| --------------------- | ------------------------- |
| Input size            | 224 × 224                 |
| Optimizer             | AdamW                     |
| Initial learning rate | 3e-4                      |
| Weight decay          | 0.003                     |
| Scheduler             | Cosine annealing          |
| Minimum learning rate | 1e-6                      |
| Batch size            | 64                        |
| Training epochs       | 200                       |
| Precision strategy    | Automatic mixed precision |
| Main GPU              | NVIDIA RTX 3090           |

The Mamba branch uses a patch size of `16 × 16`, resulting in `14 × 14 = 196` visual tokens for each input image. Each token is projected into a 512-dimensional embedding space, and four Mamba blocks are used in the final configuration.

---

## Installation

Create a Python environment and install the required dependencies:

```bash
conda create -n wbc_classification python=3.8 -y
conda activate wbc_classification

pip install -r requirements.txt
```

The code was developed with PyTorch 1.13. A CUDA-enabled GPU is recommended for training.

---

## Data preparation

After downloading the datasets from their official sources, organize the data in the following general format:

```text
dataset_root/
├── Basophil/
├── Eosinophil/
├── Lymphocyte/
├── Monocyte/
└── Neutrophil/
```

Then run the dataset preparation script:

```bash
python data_prepare.py \
  --data_dir /path/to/dataset_root \
  --output_dir /path/to/processed_dataset
```

The exact folder names may need to be adjusted according to the downloaded dataset structure.

---

## Training

Example command for training the proposed dual-stream framework:

```bash
python main.py \
  --data_dir /path/to/processed_dataset \
  --model_name dual \
  --epochs 200 \
  --batch_size 64 \
  --lr 3e-4
```

The training logs will be saved in the `logs/` directory.

---

## Evaluation

After training, evaluate the model using:

```bash
python test.py \
  --data_dir /path/to/processed_dataset \
  --checkpoint /path/to/checkpoint.pth \
  --model_name dual
```

The evaluation script reports common classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* AUC
* Confusion matrix

---

## Reproducibility

The revised manuscript reports experiments on PBC, LDWBC, and Raabin-WBC. The main analyses include:

* within-dataset classification performance;
* class-wise performance evaluation;
* controlled comparison with MobileViT and DaViT;
* ablation analysis of ConvNeXt, Mamba, SAF, and reliability-guided bilateral fusion;
* cross-dataset transfer without target-domain fine-tuning;
* training and validation curves;
* salt-and-pepper noise robustness analysis;
* computational complexity analysis;
* heatmap-based interpretability analysis.

For reproducibility, please use the provided configuration file and keep the same image-level split protocol.

---

## Notes on interpretation

The reported results are based on public image-level WBC datasets. Complete patient-level identifiers are not available for all datasets, so strict patient-level partitioning could not be performed.

The current results should therefore be interpreted as public benchmark-level evidence for image-level WBC classification. They should not be interpreted as patient-level clinical validation or evidence of clinical readiness.

Future work should include:

* patient-level multi-center validation;
* prospective evaluation under realistic laboratory workflows;
* calibrated reliability estimation;
* broader robustness analysis;
* lightweight model design for resource-constrained deployment;
* expert-reviewed quantitative interpretability assessment.

---

## Acknowledgments

We thank the contributors of the PBC, LDWBC, and Raabin-WBC datasets for making these valuable resources publicly available. These datasets provide an important foundation for the development and evaluation of automated WBC image classification methods.

---

## License

Please check the licenses of the original datasets before use. This repository only provides code and related experimental resources. Raw dataset files are not redistributed.

---
