# data_prepare.py
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _safe_imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class MedicalDataset(Dataset):
    """
      - two_view=False: return (img, label)
      - two_view=True : return ([img1, img2], label)
    """
    def __init__(self, root_dir, transform=None, transform2=None, two_view=False):
        self.root_dir = root_dir
        self.transform = transform
        self.transform2 = transform2 if transform2 is not None else transform
        self.two_view = two_view

        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_names = classes

        self.file_paths = []
        self.labels = []
        for cls_idx, cls_name in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                fp = os.path.join(cls_dir, fname)
                if os.path.isfile(fp):
                    self.file_paths.append(fp)
                    self.labels.append(cls_idx)

        self.cls_num_list = None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = int(self.labels[idx])

        img = _safe_imread_rgb(img_path)

        if self.two_view:
            if self.transform is None:
                raise ValueError("two_view=True requires transform")
            v1 = self.transform(image=img)["image"]
            v2 = self.transform2(image=img)["image"]
            return [v1, v2], torch.tensor(label, dtype=torch.long)

        # single view
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            # minimal tensor
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, torch.tensor(label, dtype=torch.long)

def create_loaders(
    data_dir,
    batch_size=32,
    img_size=224,
    num_workers=4,
    test_size=0.3,
    val_size_in_test=1/3,
    seed=42,
    split_file=None,
    save_split=True,
):
    """
    Create train/val/test DataLoaders with stratified image-level split.

    If split_file is provided and exists, loads pre-saved split indices.
    Otherwise creates a new stratified split and optionally saves it.

    train_loader: (img, y)
    val_loader  : (img, y)
    test_loader : (img, y)
    """
    def get_random_resized_crop():
        try:
            return A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0)
        except TypeError:
            return A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0)

    train_transform = A.Compose([
        A.Resize(height=256, width=256),
        get_random_resized_crop(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # full dataset for stratified split
    full_dataset = MedicalDataset(data_dir, transform=None, two_view=False)

    # Try loading pre-saved split
    if split_file is not None and os.path.exists(split_file):
        with open(split_file, "r") as f:
            split_data = json.load(f)
        train_idx = split_data["train_idx"]
        val_idx = split_data["val_idx"]
        test_idx = split_data["test_idx"]
    else:
        train_idx, temp_idx = train_test_split(
            range(len(full_dataset)),
            test_size=float(test_size),
            stratify=full_dataset.labels,
            random_state=int(seed)
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=float(1.0 - val_size_in_test),
            stratify=[full_dataset.labels[i] for i in temp_idx],
            random_state=int(seed)
        )

        # Save split for reproducibility
        if save_split and split_file is not None:
            os.makedirs(os.path.dirname(split_file), exist_ok=True)
            split_data = {
                "train_idx": [int(i) for i in train_idx],
                "val_idx": [int(i) for i in val_idx],
                "test_idx": [int(i) for i in test_idx],
                "train_paths": [full_dataset.file_paths[i] for i in train_idx],
                "val_paths": [full_dataset.file_paths[i] for i in val_idx],
                "test_paths": [full_dataset.file_paths[i] for i in test_idx],
                "train_labels": [full_dataset.labels[i] for i in train_idx],
                "val_labels": [full_dataset.labels[i] for i in val_idx],
                "test_labels": [full_dataset.labels[i] for i in test_idx],
                "class_names": full_dataset.class_names,
                "seed": int(seed),
                "dataset_dir": str(data_dir),
            }
            with open(split_file, "w") as f:
                json.dump(split_data, f, indent=2)

    # train dataset
    train_dataset = MedicalDataset(
        data_dir,
        transform=train_transform,
        two_view=False
    )
    train_dataset.file_paths = [full_dataset.file_paths[i] for i in train_idx]
    train_dataset.labels = [full_dataset.labels[i] for i in train_idx]

    # cls_num_list
    num_classes = len(train_dataset.class_names)
    cls_counts = [0] * num_classes
    for y in train_dataset.labels:
        cls_counts[int(y)] += 1
    train_dataset.cls_num_list = cls_counts

    # val/test dataset
    val_dataset = MedicalDataset(data_dir, transform=val_transform, two_view=False)
    val_dataset.file_paths = [full_dataset.file_paths[i] for i in val_idx]
    val_dataset.labels = [full_dataset.labels[i] for i in val_idx]

    test_dataset = MedicalDataset(data_dir, transform=val_transform, two_view=False)
    test_dataset.file_paths = [full_dataset.file_paths[i] for i in test_idx]
    test_dataset.labels = [full_dataset.labels[i] for i in test_idx]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True
    )
    return train_loader, val_loader, test_loader

