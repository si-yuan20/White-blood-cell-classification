# config.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import os


# =========================
# Data
# =========================
@dataclass
class DataConfig:
    data_dir: str = ""
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # split (7:1:2) — train:val:test = 70:10:20
    test_size: float = 0.3
    val_size_in_test: float = 1/3

# =========================
# Model Ablations
# =========================
@dataclass
class AblationConfig:
    # 2.1 Morphology-aware Dual-Stream Encoder
    use_convnext: bool = True
    use_mamba: bool = True

    # 2.2 Structure-Aided Attention Fusion (SAF)
    use_saf: bool = True
    saf_prior: str = "edge"
    saf_dim: int = 256
    saf_fuse: str = "cat"

    # 2.3 Reliability-guided bilateral fusion (RGBF)
    use_rgbf: bool = True
    rgbf_temperature: float = 1.0
    detach_gate: bool = False
    gate_min: float = 0.05


# =========================
# Model
# =========================
@dataclass
class ModelConfig:
    # convnext | mamba | dual
    model_name: str = "dual"

    # ConvNeXt
    pretrained: bool = True
    pretrained_path: str = "convnext_base-6075fbad.pth"
    in_channels: int = 3

    # dual ablation
    dual: AblationConfig = field(default_factory=AblationConfig)


# =========================
# Optim / Scheduler
# =========================
@dataclass
class OptimConfig:
    optimizer: str = "adamw"  # adamw | sgd
    lr: float = 3e-4
    weight_decay: float = 0.003
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    name: str = "cosine"    # cosine | none
    t_max: int = 100        # will be overwritten by epochs
    eta_min: float = 1e-6
    step_every_epochs: int = 1


# =========================
# Train
# =========================
@dataclass
class TrainConfig:
    epochs: int = 200
    gpus: int = 1
    seed: int = 42

    amp: bool = True
    grad_accum_steps: int = 1
    clip_grad_norm: float = 0.0
    cudnn_benchmark: bool = True

    # logging / saving
    save_dir: str = "models"
    log_dir: str = "logs"
    result_dir: str = "results"
    save_best_metric: str = "macro_f1"
    save_name: str = "best_model.pth"
    eval_every_epoch: bool = True
    deterministic: bool = True


# =========================
# DDP (optional)
# =========================
@dataclass
class DDPConfig:
    enabled: bool = False
    backend: str = "nccl"
    find_unused_parameters: bool = False
    sync_bn: bool = False


# =========================
# Root config
# =========================
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sched: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)


def make_default_cfg() -> Config:
    return Config()


def ensure_dirs(cfg: Config, dataset_name: str, model_name: str) -> None:
    os.makedirs(os.path.join(cfg.train.result_dir, dataset_name, model_name), exist_ok=True)
