##############################
# utils.py
##############################
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve, auc)

def get_dataset_name(cfg) -> str:

    data_dir = getattr(getattr(cfg, "data", None), "data_dir", "")
    if not data_dir:
        return "dataset"
    return os.path.basename(os.path.normpath(data_dir))

def set_global_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# utils.py
def build_model_name(cfg):

    model_type = str(getattr(cfg.model, "model_name", "model")).strip()
    dual_cfg = getattr(cfg.model, "dual", None)

    if model_type.lower() == "dual" and dual_cfg is not None:
        use_convnext = getattr(dual_cfg, "use_convnext", True)
        use_mamba = getattr(dual_cfg, "use_mamba", True)
        use_saf = getattr(dual_cfg, "use_saf", False)
        saf_prior = getattr(dual_cfg, "saf_prior", "none")
        saf_fuse = getattr(dual_cfg, "saf_fuse", "cat")
        use_rgbf = getattr(dual_cfg, "use_rgbf",
                  getattr(dual_cfg, "use_ugbf", False))

        parts = [
            "Dual",
            f"ConvNeXt{use_convnext}",
            f"Mamba{use_mamba}",
            f"SAF{use_saf}",
            f"prior{str(saf_prior)}",
            f"fuse{str(saf_fuse)}",
            f"RGBF{use_rgbf}",
        ]
        return "_".join(parts)

    # convnext / mamba / other
    return model_type


def prepare_result_dirs(dataset_name, model_name):
    base_dir = os.path.join("results", dataset_name, model_name)

    dirs = {
        "base": base_dir,
        "logs": os.path.join(base_dir, "logs"),
        "visuals": os.path.join(base_dir, "visuals"),
        "models": os.path.join(base_dir, "models"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(dirs["logs"], f"train_{timestamp}.log")

    return dirs, log_file

def calculate_auc(targets, scores, num_classes=None):
    """
    Robust multiclass AUC (OvR), handles missing classes.

    targets: [N]
    scores : [N, C]  (logits or probs)
    num_classes: C (optional). If None, infer from scores.shape[1]
    """
    targets = np.asarray(targets).astype(int)
    scores = np.asarray(scores)

    if scores.ndim != 2:
        return float("nan")

    n, c = scores.shape
    if num_classes is None:
        num_classes = c

    if c != num_classes:
        return float("nan")

    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-12)

    auc_list = []
    for k in range(num_classes):
        y_true = (targets == k).astype(int)
        if y_true.max() == y_true.min():
            continue
        try:
            auc_k = roc_auc_score(y_true, probs[:, k])
            auc_list.append(float(auc_k))
        except Exception:
            continue

    if len(auc_list) == 0:
        return float("nan")

    return float(np.mean(auc_list))


def calculate_precision(targets, preds):
    return precision_score(targets, preds, average='macro')


def calculate_recall(targets, preds):
    return recall_score(targets, preds, average='macro')

# def calculate_f1(targets, preds):
#     return f1_score(targets, preds, average='macro')


def calculate_sensitivity(targets, preds):
    cm = confusion_matrix(targets, preds)
    n_classes = cm.shape[0]
    sensitivity = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        sensitivity.append(tp / (tp + fn) if (tp + fn) != 0 else 0.0)
    
    return np.mean(sensitivity)

def calculate_specificity(targets, preds):
    cm = confusion_matrix(targets, preds)
    n_classes = cm.shape[0]
    specificity = []
    for i in range(n_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(tn / (tn + fp)) if (tn + fp) != 0 else 0.0
    return np.mean(specificity)
                           


class MetricTracker:
    """
    tracker.update(loss, preds, probs, labels)

    - preds  : Tensor [B]
    - probs  : Tensor [B, C]（可以是 logits）
    - labels : Tensor [B]
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_correct = 0

        self.all_targets = []
        self.all_preds = []
        self.all_probs = []

    def update(self, loss, preds, probs, labels):
        """
        update(loss, preds, probs, labels)
        """
        # -------- batch size --------
        if torch.is_tensor(labels):
            bsz = labels.numel()
        else:
            bsz = len(labels)

        # -------- loss --------
        if loss is not None:
            if torch.is_tensor(loss):
                loss_val = float(loss.detach().cpu().item())
            else:
                loss_val = float(loss)
        else:
            loss_val = 0.0

        self.total_loss += loss_val * bsz
        self.total_samples += bsz

        # -------- preds / labels --------
        if torch.is_tensor(preds):
            preds_np = preds.detach().cpu().numpy()
        else:
            preds_np = np.asarray(preds)

        if torch.is_tensor(labels):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)

        self.total_correct += int((preds_np == labels_np).sum())

        # -------- probs (logits or softmax) --------
        if torch.is_tensor(probs):
            probs_np = probs.detach().cpu().numpy()
        else:
            probs_np = np.asarray(probs)

        self.all_preds.append(preds_np)
        self.all_targets.append(labels_np)
        self.all_probs.append(probs_np)

    def compute(self, with_raw: bool = False):
        n = max(self.total_samples, 1)
        avg_loss = self.total_loss / n
        acc = self.total_correct / n

        targets = np.concatenate(self.all_targets, axis=0) if self.all_targets else np.array([])
        preds = np.concatenate(self.all_preds, axis=0) if self.all_preds else np.array([])
        probs = np.concatenate(self.all_probs, axis=0) if self.all_probs else np.array([])

        if targets.size and probs.size and probs.ndim == 2:
            auc = calculate_auc(targets, probs, num_classes=probs.shape[1])
        else:
            auc = float("nan")

        from sklearn.metrics import f1_score, precision_score, recall_score
        if targets.size and preds.size:
            f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
            precision = float(precision_score(targets, preds, average="macro", zero_division=0))
            recall = float(recall_score(targets, preds, average="macro", zero_division=0))
        else:
            f1 = precision = recall = float("nan")

        res = {
            "loss": float(avg_loss),
            "acc": float(acc),
            "auc": float(auc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }

        if with_raw:
            raw = {"targets": targets, "preds": preds, "probs": probs}
            res["raw"] = raw
            res.update(raw)
        return res


# utils.py
def plot_learning_curves(train_metrics, val_metrics, dataset_name=None, model_name=None, save_path=None, **kwargs):
    dataset_name = dataset_name or kwargs.get("dataset_name", "dataset")
    model_name = model_name or kwargs.get("model_name", "model")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_metrics["loss"], label="Train")
    plt.plot(val_metrics["loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_metrics["acc"], label="Train")
    plt.plot(val_metrics["acc"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    if save_path is None:
        save_path = os.path.join("results", dataset_name, model_name, "curves.png")

    save_dir = os.path.dirname(save_path)
    if save_dir and (not os.path.exists(save_dir)):
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# utils.py
def plot_confusion_matrix(targets, preds, class_names, save_path=None, title=None, **kwargs):

    dataset_name = kwargs.get("dataset_name", "dataset")
    model_name = kwargs.get("model_name", "model")
    epoch = kwargs.get("epoch", None)

    cm = confusion_matrix(targets, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if title is None:
        title = f"Confusion Matrix ({dataset_name} | {model_name}" + (f" | epoch {epoch}" if epoch is not None else "") + ")"
    plt.title(title)

    if save_path is None:
        # results/{dataset}/{model}/confusion_matrix_epochXX.png
        filename = f"confusion_matrix_epoch{int(epoch):03d}.png" if epoch is not None else "confusion_matrix.png"
        save_path = os.path.join("results", dataset_name, model_name, filename)

    save_dir = os.path.dirname(save_path)
    if save_dir and (not os.path.exists(save_dir)):
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# utils.py
def plot_roc_curve(targets, scores, class_names, save_path=None, title=None, **kwargs):

    dataset_name = kwargs.get("dataset_name", "dataset")
    model_name = kwargs.get("model_name", "model")
    epoch = kwargs.get("epoch", None)

    targets = np.asarray(targets).astype(int)
    scores = np.asarray(scores)

    if scores.ndim != 2:
        raise ValueError(f"plot_roc_curve expects scores shape [N,C], got {scores.shape}")

    if not np.isfinite(scores).all():
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-12)

    n, c = probs.shape
    if len(class_names) != c:
        class_names = [f"class_{i}" for i in range(c)]

    # one-hot labels
    y_true = np.zeros((n, c), dtype=np.int32)
    y_true[np.arange(n), targets] = 1

    plt.figure()

    valid_any = False
    for i in range(c):
        yi = y_true[:, i]
        si = probs[:, i]

        if yi.max() == yi.min():
            continue

        fpr, tpr, _ = roc_curve(yi, si)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
        valid_any = True

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    if title is None:
        title = f"ROC Curve ({dataset_name} | {model_name}" + (f" | epoch {epoch}" if epoch is not None else "") + ")"
    plt.title(title)

    if valid_any:
        plt.legend(loc="lower right")
    else:
        plt.text(0.5, 0.5, "ROC cannot be computed (missing classes)",
                 ha="center", va="center")

    if save_path is None:
        filename = f"roc_epoch{int(epoch):03d}.png" if epoch is not None else "roc.png"
        save_path = os.path.join("results", dataset_name, model_name, filename)

    save_dir = os.path.dirname(save_path)
    if save_dir and (not os.path.exists(save_dir)):
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def grad_cam(*args, **kwargs):
    """Deprecated. Use visualization/grad_cam.py instead (hook-based Grad-CAM)."""
    raise NotImplementedError(
        "utils.grad_cam is deprecated. "
        "Please use visualization/grad_cam.py for hook-based Grad-CAM heatmap generation."
    )


def plot_feature_heatmaps(images, heatmaps, path='results/'):
    """
    Deprecated. Use visualization/grad_cam.py instead.

    :param images:  (B, C, H, W)
    :param heatmaps: [stage1, stage2, stage3]
    :param path:
    """
    raise NotImplementedError(
        "utils.plot_feature_heatmaps is deprecated. "
        "Please use visualization/grad_cam.py instead."
    )


# def plot_learning_rates(history,dataset_name,model_name):
#     plt.figure(figsize=(10, 6))
#     plt.plot(history['lr'], 'o-', label='Learning Rate')
#     plt.xlabel('Epoch')
#     plt.ylabel('Learning Rate')
#     plt.title('Learning Rate Schedule')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"test_result/{dataset_name}/{model_name}_LR.png")