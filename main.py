# main.py
import os
import sys
import time
import argparse
import warnings
import socket


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from contextlib import suppress
from dataclasses import asdict
from datetime import datetime

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data_prepare import create_loaders
from utils import MetricTracker, plot_confusion_matrix, plot_roc_curve, plot_learning_curves
from dual_model import create_classifier
from config import make_default_cfg, ensure_dirs, AblationConfig
from utils import build_model_name, prepare_result_dirs, set_global_seed, get_dataset_name

warnings.filterwarnings("ignore")

class TeeLogger:
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

# =========================
# Pretty console helpers
# =========================
class Console:
    def __init__(self, enable: bool = True):
        self.enable = enable

    @staticmethod
    def _term_width(default: int = 120) -> int:
        with suppress(Exception):
            return os.get_terminal_size().columns
        return default

    def line(self, ch: str = "─"):
        if not self.enable:
            return
        w = self._term_width()
        print(ch * w)

    def title(self, text: str):
        if not self.enable:
            return
        w = self._term_width()
        text = f" {text} "
        pad = max(0, (w - len(text)) // 2)
        print("═" * pad + text + "═" * (w - pad - len(text)))

    def kv(self, k: str, v: str, k_width: int = 18):
        if not self.enable:
            return
        print(f"{k:<{k_width}}: {v}")

    def table_header(self):
        if not self.enable:
            return
        # epoch | train(loss/acc) | val(loss/acc/auc/f1/p/r) | lr | best | time
        print(
            f"{'Epoch':>5}  "
            f"{'Train(L/A)':>14}  "
            f"{'Val(L/A/AUC)':>16}  "
            f"{'Val(F1/P/R)':>14}  "
            f"{'LR':>10}  "
            f"{'BestAUC':>8}  "
            f"{'Time':>8}"
        )
        self.line("─")

    def table_row(
        self,
        epoch: int,
        epochs: int,
        tr_loss: float,
        tr_acc: float,
        va_loss: float,
        va_acc: float,
        va_auc: float,
        va_f1: float,
        va_precision: float,
        va_recall: float,
        lr: float,
        best_auc: float,
        secs: float
    ):
        if not self.enable:
            return

        def _fmt_time(s: float) -> str:
            if s < 60:
                return f"{s:5.1f}s"
            m = int(s // 60)
            r = s - 60 * m
            return f"{m:2d}m{r:02.0f}s"

        print(
            f"{epoch:>2}/{epochs:<2}  "
            f"{tr_loss:>6.4f}/{tr_acc:>5.4f}  "
            f"{va_loss:>6.4f}/{va_acc:>5.4f}/{va_auc:>5.4f}  "
            f"{va_f1:>5.4f}/{va_precision:>5.4f}/{va_recall:>5.4f}  "
            f"{lr:>10.6f}  "
            f"{best_auc:>8.4f}  "
            f"{_fmt_time(secs):>8}"
        )

    def note(self, text: str):
        if not self.enable:
            return
        print(f"• {text}")


# =========================
# DDP helpers
# =========================
def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_dist_avail_and_initialized() else 0


def _is_main_process() -> bool:
    return _get_rank() == 0


def _setup_ddp(rank: int, world_size: int, port: int, backend: str = "nccl"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup_ddp():
    with suppress(Exception):
        dist.barrier()
    with suppress(Exception):
        dist.destroy_process_group()


# =========================
# existing helpers
# =========================
def _infer_feat_dim(model_name: str) -> int:
    name = model_name.lower()
    if "resnet" in name:
        return 2048
    if "convnext" in name:
        return 1024
    return 1024


def _merge(cli_val, cfg_val):
    return cfg_val if cli_val is None else cli_val

def build_model(cfg, num_classes: int, device: torch.device) -> nn.Module:
    model_name = cfg.model.model_name.lower().strip()

    def _load_convnext_ckpt(convnext_backbone: nn.Module):
        ckpt_path = cfg.model.pretrained_path
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"ConvNeXt pretrained weight not found: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        msg = convnext_backbone.load_state_dict(state_dict, strict=False)
        if _is_main_process():
            print(f"[ConvNeXt] Loaded pretrained weights from {ckpt_path}")
            print(f"[ConvNeXt] Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")

    # Build ablation config for dual-stream variants
    if model_name.startswith("dual"):
        ab = AblationConfig(
            use_convnext=cfg.model.dual.use_convnext,
            use_mamba=cfg.model.dual.use_mamba,
            use_saf=cfg.model.dual.use_saf,
            saf_prior=cfg.model.dual.saf_prior,
            saf_dim=getattr(cfg.model.dual, "saf_dim", 256),
            saf_fuse=cfg.model.dual.saf_fuse,
            use_rgbf=cfg.model.dual.use_rgbf,
            rgbf_temperature=cfg.model.dual.rgbf_temperature,
            detach_gate=cfg.model.dual.detach_gate,
            gate_min=cfg.model.dual.gate_min,
        )

        model = create_classifier(
            model_name,
            num_classes=num_classes,
            pretrained=False,
            ablation=ab,
        )

        if cfg.model.pretrained:
            if hasattr(model, "model") and hasattr(model.model, "convnext_backbone"):
                _load_convnext_ckpt(model.model.convnext_backbone)

        model = model.to(device)
        return model

    # Single-stream or external baselines
    model = create_classifier(model_name, num_classes=num_classes,
                              pretrained=cfg.model.pretrained)

    if cfg.model.pretrained and model_name == "convnext":
        if hasattr(model, "model") and hasattr(model.model, "backbone"):
            _load_convnext_ckpt(model.model.backbone)

    model = model.to(device)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """
      - inputs/labels
      - AMP
      - sampler.set_epoch
    """
    model.train()
    tracker = MetricTracker()

    use_amp = getattr(getattr(model, "module", model), "use_amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    pbar = tqdm(
        loader,
        desc=f"Train {epoch}/{total_epochs}",
        leave=False,
        disable=(not _is_main_process()),
        dynamic_ncols=True
    )

    for batch in pbar:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, labels = batch
        elif isinstance(batch, dict):
            inputs, labels = batch["image"], batch["label"]
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        probs = F.softmax(outputs, dim=1)

        tracker.update(loss, preds, probs, labels)

        if _is_main_process():
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(preds==labels).float().mean().item():.4f}")

    return tracker.compute()



@torch.no_grad()
def validate(model, loader, criterion, device, mode="Val"):
    model.eval()
    tracker = MetricTracker()

    pbar = tqdm(
        loader,
        desc=f"{mode}",
        leave=False,
        disable=(not _is_main_process()),
        dynamic_ncols=True
    )

    for batch in pbar:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, labels = batch
        elif isinstance(batch, dict):
            inputs, labels = batch["image"], batch["label"]
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        probs = F.softmax(outputs, dim=1)

        tracker.update(loss, preds, probs, labels)

    return tracker.compute(with_raw=True)  



def parse_args():
    p = argparse.ArgumentParser()

    # ========= data =========
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--img_size', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=None)
    p.add_argument('--test_size', type=float, default=None)
    p.add_argument('--val_size_in_test', type=float, default=None)

    # ========= train =========
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--gpus', type=int, default=None)

    # ========= model =========
    p.add_argument('--model_name', type=str, default=None)     # convnext | mamba | dual
    p.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=None)

    # ========= optim/sched =========
    p.add_argument('--optimizer', type=str, default=None, help="adamw | sgd")
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--momentum', type=float, default=None)

    p.add_argument('--sched', type=str, default=None, help="cosine | none")
    p.add_argument('--eta_min', type=float, default=None)

    # ========= NEW: dual ablation for 3 innovations =========
    # 2.1 Morphology-aware Dual-Stream Encoder
    p.add_argument('--dual_use_convnext', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--dual_use_mamba', action=argparse.BooleanOptionalAction, default=None)

    # 2.2 SAF
    p.add_argument('--dual_use_saf', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--dual_saf_prior', type=str, default=None, choices=["edge", "none"])
    p.add_argument('--dual_saf_fuse', type=str, default=None, choices=["add", "cat"])

    # 2.3 RGBF (reliability-guided bilateral fusion)
    p.add_argument('--dual_use_rgbf', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--dual_use_ugbf', action=argparse.BooleanOptionalAction, default=None,
                   help="(deprecated) use --dual_use_rgbf instead")
    p.add_argument('--dual_rgbf_temperature', type=float, default=None)
    p.add_argument('--dual_ugbf_temperature', type=float, default=None,
                   help="(deprecated) use --dual_rgbf_temperature instead")
    p.add_argument('--dual_detach_gate', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--dual_gate_min', type=float, default=None)

    return p.parse_args()



def apply_cli_to_cfg(cfg, args):
    # data
    cfg.data.data_dir = _merge(args.data_dir, cfg.data.data_dir)
    cfg.data.img_size = _merge(args.img_size, cfg.data.img_size)
    cfg.data.batch_size = _merge(args.batch_size, cfg.data.batch_size)
    cfg.data.num_workers = _merge(args.num_workers, cfg.data.num_workers)
    cfg.data.test_size = _merge(args.test_size, cfg.data.test_size)
    cfg.data.val_size_in_test = _merge(args.val_size_in_test, cfg.data.val_size_in_test)

    # train
    cfg.train.epochs = _merge(args.epochs, cfg.train.epochs)
    cfg.train.seed = _merge(args.seed, cfg.train.seed)
    cfg.train.gpus = _merge(args.gpus, cfg.train.gpus)

    # model
    cfg.model.model_name = _merge(args.model_name, cfg.model.model_name)
    cfg.model.pretrained = _merge(args.pretrained, cfg.model.pretrained)

    # optim
    cfg.optim.optimizer = _merge(args.optimizer, cfg.optim.optimizer)
    cfg.optim.lr = _merge(args.lr, cfg.optim.lr)
    cfg.optim.weight_decay = _merge(args.weight_decay, cfg.optim.weight_decay)
    cfg.optim.momentum = _merge(args.momentum, cfg.optim.momentum)

    # sched
    cfg.sched.name = _merge(args.sched, cfg.sched.name)
    cfg.sched.eta_min = _merge(args.eta_min, cfg.sched.eta_min)
    cfg.sched.t_max = cfg.train.epochs

    # dual ablation (3 innovations)
    cfg.model.dual.use_convnext = _merge(args.dual_use_convnext, cfg.model.dual.use_convnext)
    cfg.model.dual.use_mamba = _merge(args.dual_use_mamba, cfg.model.dual.use_mamba)

    cfg.model.dual.use_saf = _merge(args.dual_use_saf, cfg.model.dual.use_saf)
    cfg.model.dual.saf_prior = _merge(args.dual_saf_prior, cfg.model.dual.saf_prior)
    cfg.model.dual.saf_fuse = _merge(args.dual_saf_fuse, cfg.model.dual.saf_fuse)

    # RGBF: prefer new arg name, fall back to legacy
    _use_rgbf = args.dual_use_rgbf if args.dual_use_rgbf is not None else args.dual_use_ugbf
    cfg.model.dual.use_rgbf = _merge(_use_rgbf, cfg.model.dual.use_rgbf)
    _rgbf_T = args.dual_rgbf_temperature if args.dual_rgbf_temperature is not None else args.dual_ugbf_temperature
    cfg.model.dual.rgbf_temperature = _merge(_rgbf_T, cfg.model.dual.rgbf_temperature)
    cfg.model.dual.detach_gate = _merge(args.dual_detach_gate, cfg.model.dual.detach_gate)
    cfg.model.dual.gate_min = _merge(args.dual_gate_min, cfg.model.dual.gate_min)

    return cfg



def _make_ddp_train_loader(train_loader: DataLoader, global_batch: int, num_workers: int, rank: int, world_size: int, console: Console):
    per_gpu_batch = max(1, global_batch // world_size)
    if rank == 0 and global_batch % world_size != 0:
        console.note(
            f"Global batch_size={global_batch} not divisible by world_size={world_size}. "
            f"Using per_gpu_batch={per_gpu_batch} (global≈{per_gpu_batch*world_size})."
        )

    ds = train_loader.dataset
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

    return DataLoader(
        ds,
        batch_size=per_gpu_batch,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    ), per_gpu_batch


def _resolve_world_size(cfg) -> int:
    available = torch.cuda.device_count()
    if available <= 0:
        return 1

    req = int(cfg.train.gpus) if cfg.train.gpus is not None else 0
    if req <= 0:
        return available
    return max(1, min(req, available))


def _print_run_summary(console: Console, cfg, dataset_name: str, num_classes: int, world_size: int,
                       per_gpu_batch: int, feat_dim=None):
    console.title("RUN CONFIG")
    console.kv("Dataset", f"{dataset_name}  (classes={num_classes})")
    console.kv("Model", f"{cfg.model.model_name}  (pretrained={cfg.model.pretrained})")
    console.kv("Image size", str(cfg.data.img_size))
    console.kv("Batch (global)", str(cfg.data.batch_size))
    console.kv("Batch (per GPU)", str(per_gpu_batch))
    console.kv("Workers", str(cfg.data.num_workers))
    console.kv("GPU/World size", f"{world_size}  (cfg.gpus={cfg.train.gpus})")

    # NEW: show ablation switches when dual
    if cfg.model.model_name.lower().strip() == "dual":
        console.kv("Dual.ConvNeXt", str(cfg.model.dual.use_convnext))
        console.kv("Dual.Mamba", str(cfg.model.dual.use_mamba))
        console.kv("SAF", f"{cfg.model.dual.use_saf}  prior={cfg.model.dual.saf_prior}  fuse={cfg.model.dual.saf_fuse}")
        console.kv("RGBF", f"{cfg.model.dual.use_rgbf}  T={cfg.model.dual.rgbf_temperature}")

    console.kv("Optimizer", f"{cfg.optim.optimizer}  lr={cfg.optim.lr}  wd={cfg.optim.weight_decay}")
    console.kv("Scheduler", f"{cfg.sched.name}  eta_min={cfg.sched.eta_min}")
    console.kv("Epochs", str(cfg.train.epochs))
    if feat_dim is not None:
        console.kv("Feature dim", str(feat_dim))
    console.line("─")



def main_worker(rank: int, world_size: int, port: int, args_namespace):


    if world_size > 1:
        _setup_ddp(rank, world_size, port, backend="nccl")

    console = Console(enable=_is_main_process())

    try:
        cfg = make_default_cfg()
        cfg = apply_cli_to_cfg(cfg, args_namespace)

        set_global_seed(cfg.train.seed, getattr(cfg.train, "deterministic", True))

        if not cfg.data.data_dir:
            raise ValueError("data_dir is empty. Please set config.data.data_dir or pass --data_dir in script.")

        dataset_name = os.path.basename(os.path.normpath(cfg.data.data_dir))

        # Split file for reproducible dataset splits
        split_file = os.path.join("splits", f"{dataset_name}_seed{cfg.train.seed}.json")

        run_model_name = build_model_name(cfg)

        dirs, log_file = prepare_result_dirs(dataset_name, run_model_name)

        if _is_main_process():
            sys.stdout = TeeLogger(log_file)

        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # 
        class_dirs = [d for d in os.listdir(cfg.data.data_dir) if os.path.isdir(os.path.join(cfg.data.data_dir, d))]
        class_dirs = sorted(class_dirs)
        num_classes = len(class_dirs)

        train_loader, val_loader, test_loader = create_loaders(
            cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            img_size=cfg.data.img_size,
            num_workers=cfg.data.num_workers,
            test_size=cfg.data.test_size,
            val_size_in_test=cfg.data.val_size_in_test,
            seed=cfg.train.seed,
            split_file=split_file,
            save_split=True,
        )


        per_gpu_batch = cfg.data.batch_size
        if world_size > 1:
            train_loader, per_gpu_batch = _make_ddp_train_loader(
                train_loader,
                global_batch=cfg.data.batch_size,
                num_workers=cfg.data.num_workers,
                rank=rank,
                world_size=world_size,
                console=console
            )

        base_model = build_model(cfg, num_classes, device).to(device)

        if world_size > 1:
            torch.cuda.set_device(rank)  #
            base_model = torch.nn.parallel.DistributedDataParallel(
                base_model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False
            )

        def build_optimizer(params):
            if cfg.optim.optimizer.lower() == "sgd":
                return SGD(params, lr=cfg.optim.lr, momentum=cfg.optim.momentum, weight_decay=cfg.optim.weight_decay)
            return AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

        def build_scheduler(opt):
            if cfg.sched.name.lower() == "cosine":
                return CosineAnnealingLR(opt, T_max=cfg.sched.t_max, eta_min=cfg.sched.eta_min)
            return None

        ce_criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(base_model.parameters())
        scheduler = build_scheduler(optimizer)

        if _is_main_process():
            ensure_dirs(cfg, dataset_name, run_model_name)

            feat_dim = None
            with suppress(Exception):
                m = base_model.module if hasattr(base_model, "module") else base_model
                feat_dim = getattr(m, "feature_dim", None)

            _print_run_summary(console, cfg, dataset_name, num_classes, world_size, per_gpu_batch, feat_dim)
            console.title("TRAINING")
            console.table_header()

        # best based on macro_f1
        best_score = -1.0

        history = {'train': {'loss': [], 'acc': []},
                   'val': {'loss': [], 'acc': [], 'auc': [], 'f1': [], 'precision': [], 'recall': []},
                   'lr': []}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"training_log_{timestamp}.txt"
        log_path = os.path.join(dirs["logs"], log_filename)

        best_acc = -1.0
        for epoch in range(cfg.train.epochs):
            t0 = time.time()

            train_res = train_one_epoch(
                base_model, train_loader, ce_criterion, optimizer,
                device, epoch + 1, cfg.train.epochs
            )

            if scheduler is not None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            if _is_main_process():
                val_res = validate(base_model, val_loader, ce_criterion, device, mode='Val')


                history['lr'].append(current_lr)
                for metric in history['train']:
                    history['train'][metric].append(train_res.get(metric, 0.0))
                for metric in history['val']:
                    history['val'][metric].append(val_res.get(metric, 0.0))

                console.table_row(
                    epoch=epoch + 1, epochs=cfg.train.epochs,
                    tr_loss=train_res['loss'], tr_acc=train_res['acc'],
                    va_loss=val_res['loss'], va_acc=val_res['acc'], va_auc=val_res.get('auc', 0.0),
                    va_f1=val_res.get('f1', 0.0),
                    va_precision=val_res.get('precision', 0.0),
                    va_recall=val_res.get('recall', 0.0),
                    lr=current_lr, best_auc=val_res.get("auc", float("nan")), secs=time.time() - t0
                )

                with open(log_path, "a", buffering=1) as f:
                    f.write(
                        f"[{epoch+1:03d}/{cfg.train.epochs:03d}] "
                        f"train(loss={train_res['loss']:.4f},acc={train_res['acc']:.4f}) "
                        f"val(loss={val_res['loss']:.4f},acc={val_res['acc']:.4f},auc={val_res.get('auc',0.0):.4f},"
                        f"f1={val_res.get('f1', 0.0):.4f},prec={val_res.get('precision', 0.0):.4f},rec={val_res.get('recall', 0.0):.4f}) "
                        f"lr={current_lr:.6f} best_f1={best_score:.4f}\n"
                    )

                if epoch % 10 == 0:
                    torch.save(
                        base_model.state_dict(),
                        os.path.join(dirs["models"], f"epoch{epoch:03d}.pth")
                    )

                if val_res["acc"] > best_acc:
                    best_acc = val_res["acc"]
                    state = base_model.module.state_dict() if hasattr(base_model, "module") else base_model.state_dict()
                    best_path = os.path.join(dirs["models"], "best_model.pth")
                    torch.save(state, best_path)
                    print(f"[CKPT] Saved best_model (best_acc={best_acc:.4f}) -> {best_path}")

                if epoch % 10 == 0 or val_res["acc"] >= best_acc:
                    tag = "best" if val_res["acc"] >= best_acc else f"epoch{epoch:03d}"

                    plot_roc_curve(
                        val_res["targets"],
                        val_res["probs"],
                        class_names=class_dirs,
                        dataset_name= dataset_name,
                        model_name=run_model_name,
                        save_path=os.path.join(dirs["visuals"], f"roc_{tag}.png"),
                    )

                    plot_confusion_matrix(
                        val_res["targets"],
                        val_res["preds"],
                        class_names=class_dirs,
                        dataset_name=dataset_name,
                        model_name=run_model_name,
                        save_path=os.path.join(dirs["visuals"], f"confusion_matrix_{tag}.png"),
                    )


            if _is_dist_avail_and_initialized():
                dist.barrier()

        # final eval
        if _is_main_process():
            console.line("─")
            console.title("FINAL EVAL")

            best_path = os.path.join(dirs["models"], "best_model.pth")
            if not os.path.exists(best_path):
                raise FileNotFoundError(
                    f"[FINAL EVAL] best_model.pth not found.\n"
                    f"Expected: {best_path}\n"
                    f"Please confirm you saved best checkpoints into dirs['models']."
                )

            state = torch.load(best_path, map_location=device)
            if hasattr(base_model, "module"):
                base_model.module.load_state_dict(state)
            else:
                base_model.load_state_dict(state)

            test_res = validate(base_model, test_loader, ce_criterion, device, mode='Test')
            plot_learning_curves(history['train'], history['val'], dataset_name, run_model_name)
            plot_confusion_matrix(test_res['targets'], test_res['preds'], class_names=class_dirs,
                                  dataset_name=dataset_name, model_name=run_model_name, title="Test Results confusion_matrix")
            plot_roc_curve(test_res['targets'], test_res['probs'], class_names=class_dirs,
                           dataset_name=dataset_name, model_name=run_model_name, title="Test Results roc_curve")
            console.kv("Final Acc", f"{test_res['acc']:.4f}")
            console.kv("Final AUC", f"{test_res.get('auc',0.0):.4f}")
            console.kv("Final F1", f"{test_res.get('f1', 0.0):.4f}")
            console.kv("Final Prec/Rec", f"{test_res.get('precision', 0.0):.4f} / {test_res.get('recall', 0.0):.4f}")
            console.kv("Best(F1)", f"{best_score:.4f}")
            console.line("═")

        if _is_dist_avail_and_initialized():
            dist.barrier()

    finally:
        if world_size > 1:
            _cleanup_ddp()



def main():
    args = parse_args()

    # merge cfg once to decide world_size
    cfg = make_default_cfg()
    cfg = apply_cli_to_cfg(cfg, args)
    set_global_seed(cfg.train.seed, cfg.train.deterministic)

    world_size = _resolve_world_size(cfg)

    if world_size >= 2:
        port = _find_free_port()
        mp.spawn(
            main_worker,
            args=(world_size, port, args),
            nprocs=world_size,
            join=True
        )
    else:
        main_worker(rank=0, world_size=1, port=0, args_namespace=args)


if __name__ == '__main__':
    main()

