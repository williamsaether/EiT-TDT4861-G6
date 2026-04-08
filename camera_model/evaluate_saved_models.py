from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


def normalize_label(label: str) -> Optional[str]:
    if "concrete" in label:
        return None
    return label.replace("_severe", "").replace("_slight", "").replace("_smooth", "")


def parse_label_from_filename(path: Path) -> str:
    stem = path.stem
    if "-" not in stem:
        return stem
    _, label = stem.split("-", 1)
    return label.replace("-", "_")


class RSCDFlatDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None, class_to_idx: Optional[dict[str, int]] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx or {}
        all_samples = sorted([p for p in root_dir.iterdir() if p.is_file()])
        self.samples = [
            p for p in all_samples if normalize_label(parse_label_from_filename(p)) in self.class_to_idx
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        label_name = normalize_label(parse_label_from_filename(path))
        label = self.class_to_idx[label_name]
        img = datasets.folder.default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


def maybe_subset(ds: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds))[:max_samples]
    return Subset(ds, indices)


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@dataclass
class EvalStats:
    loss: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    n_samples: int
    confusion: np.ndarray


def per_class_metrics(confusion: np.ndarray, idx_to_class: dict[int, str]) -> list[dict[str, float | str | int]]:
    tp = np.diag(confusion).astype(np.float64)
    pred_pos = confusion.sum(axis=0).astype(np.float64)
    actual_pos = confusion.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, actual_pos, out=np.zeros_like(tp), where=actual_pos > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    rows: list[dict[str, float | str | int]] = []
    for idx in range(confusion.shape[0]):
        rows.append(
            {
                "class_idx": int(idx),
                "class_name": idx_to_class[idx],
                "support": int(actual_pos[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            }
        )
    return rows


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalStats:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    loss_sum = 0.0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            loss_sum += loss.item() * labels.size(0)
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    num_classes = int(max(labels.max(), preds.max()) + 1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, preds), 1)

    tp = np.diag(confusion).astype(np.float64)
    pred_pos = confusion.sum(axis=0).astype(np.float64)
    actual_pos = confusion.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, actual_pos, out=np.zeros_like(tp), where=actual_pos > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    return EvalStats(
        loss=loss_sum / total,
        accuracy=float((preds == labels).mean()),
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        n_samples=total,
        confusion=confusion,
    )


def top_confusions(confusion: np.ndarray, idx_to_class: dict[int, str], top_k: int = 8) -> list[tuple[str, str, int]]:
    c = confusion.copy()
    np.fill_diagonal(c, 0)
    entries: list[tuple[str, str, int]] = []
    flat_idx = np.argsort(c, axis=None)[::-1]
    for idx in flat_idx:
        count = int(c.flat[idx])
        if count <= 0:
            break
        true_idx, pred_idx = np.unravel_index(idx, c.shape)
        entries.append((idx_to_class[true_idx], idx_to_class[pred_idx], count))
        if len(entries) >= top_k:
            break
    return entries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved RSCD checkpoints on val/test splits.")
    p.add_argument("--models-dir", type=Path, default=Path("models/checkpoints"))
    p.add_argument("--data-root", type=Path, default=Path("camera_model/data/RSCD dataset-1million"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-samples", type=int, default=None, help="Optional random subset size per split.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", type=Path, default=Path("models/eval/eval_summary.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_dir = args.data_root / "vali_20k"
    test_dir = args.data_root / "test_50k"

    outputs = []
    for checkpoint_path in sorted(args.models_dir.glob("*.pt")):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "model_state" not in ckpt or "class_to_idx" not in ckpt:
            print(f"Skipping {checkpoint_path.name}: missing model_state/class_to_idx.")
            continue

        class_to_idx = ckpt["class_to_idx"]
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        model = build_model(num_classes=len(class_to_idx)).to(device, memory_format=torch.channels_last)
        model.load_state_dict(ckpt["model_state"], strict=True)

        val_ds = maybe_subset(
            RSCDFlatDataset(val_dir, transform=eval_tfms, class_to_idx=class_to_idx),
            args.max_samples,
            args.seed,
        )
        test_ds = maybe_subset(
            RSCDFlatDataset(test_dir, transform=eval_tfms, class_to_idx=class_to_idx),
            args.max_samples,
            args.seed,
        )

        pin_memory = device.type == "cuda"
        loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory, shuffle=False)
        if args.num_workers > 0:
            loader_kwargs.update(persistent_workers=True)
        val_loader = DataLoader(val_ds, **loader_kwargs)
        test_loader = DataLoader(test_ds, **loader_kwargs)

        print(f"\n=== {checkpoint_path.name} ({len(class_to_idx)} classes) ===")
        val_stats = evaluate(model, val_loader, device)
        test_stats = evaluate(model, test_loader, device)
        print(
            f"Val:  acc={val_stats.accuracy:.4f} macro_f1={val_stats.macro_f1:.4f} "
            f"precision={val_stats.macro_precision:.4f} recall={val_stats.macro_recall:.4f} n={val_stats.n_samples}"
        )
        print(
            f"Test: acc={test_stats.accuracy:.4f} macro_f1={test_stats.macro_f1:.4f} "
            f"precision={test_stats.macro_precision:.4f} recall={test_stats.macro_recall:.4f} n={test_stats.n_samples}"
        )
        conf = top_confusions(test_stats.confusion, idx_to_class)
        if conf:
            print("Top confusions (true -> pred | count):")
            for true_name, pred_name, count in conf:
                print(f"  {true_name:24s} -> {pred_name:24s} | {count}")

        outputs.append(
            {
                "checkpoint": str(checkpoint_path),
                "n_classes": len(class_to_idx),
                "class_names": [idx_to_class[i] for i in range(len(idx_to_class))],
                "val": {
                    "loss": val_stats.loss,
                    "accuracy": val_stats.accuracy,
                    "macro_precision": val_stats.macro_precision,
                    "macro_recall": val_stats.macro_recall,
                    "macro_f1": val_stats.macro_f1,
                    "n_samples": val_stats.n_samples,
                    "confusion": val_stats.confusion.tolist(),
                    "per_class": per_class_metrics(val_stats.confusion, idx_to_class),
                },
                "test": {
                    "loss": test_stats.loss,
                    "accuracy": test_stats.accuracy,
                    "macro_precision": test_stats.macro_precision,
                    "macro_recall": test_stats.macro_recall,
                    "macro_f1": test_stats.macro_f1,
                    "n_samples": test_stats.n_samples,
                    "confusion": test_stats.confusion.tolist(),
                    "per_class": per_class_metrics(test_stats.confusion, idx_to_class),
                },
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {args.output_json}")


if __name__ == "__main__":
    main()
