from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms


@dataclass
class TrainConfig:
    data_root: Path = Path("data") / "RSCD dataset-1million"
    output_dir: Path = Path("runs") / "rscd_resnet18"
    batch_size: int = 256
    image_size: int = 224
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    use_pretrained: bool = False
    compile_model: bool = False
    num_workers: int = max(4, min(10, (os.cpu_count() or 8) - 2))
    prefetch_factor: int = 2
    log_every: int = 100
    aug_profile: str = "default"  # default | fast
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_rscd")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class MergedImageFolder(Dataset):
    def __init__(self, base_ds: datasets.ImageFolder, idx_to_orig_class: dict[int, str], class_to_idx: dict[str, int]):
        self.base_ds = base_ds
        self.indices: list[int] = []
        self.targets: list[int] = []
        for sample_idx, (_, orig_target) in enumerate(base_ds.samples):
            orig_label = idx_to_orig_class[orig_target]
            merged_label = normalize_label(orig_label)
            if merged_label is None:
                continue
            self.indices.append(sample_idx)
            self.targets.append(class_to_idx[merged_label])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_idx = self.indices[idx]
        img, _ = self.base_ds[sample_idx]
        return img, self.targets[idx]


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


def get_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        base_targets = np.asarray(dataset.dataset.targets, dtype=np.int64)
        idx = np.asarray(dataset.indices, dtype=np.int64)
        return base_targets[idx]
    return np.asarray(dataset.targets, dtype=np.int64)


def build_model(num_classes: int, use_pretrained: bool) -> nn.Module:
    if use_pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
) -> tuple[Optional[float], float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total else 0.0
    loss_avg = (running_loss / total) if (criterion is not None and total) else None
    return loss_avg, acc


def benchmark_dataloader(loader: DataLoader, logger: logging.Logger, warmup_batches: int = 2, measure_batches: int = 12) -> None:
    logger.info(
        "[timing] init iterator (workers=%s, warmup=%d, measure=%d)",
        loader.num_workers,
        warmup_batches,
        measure_batches,
    )
    it = iter(loader)
    t0 = time.perf_counter()
    next(it)
    first_batch_s = time.perf_counter() - t0
    logger.info("[timing] first batch %.3fs", first_batch_s)

    for _ in range(max(0, warmup_batches - 1)):
        next(it)
    t1 = time.perf_counter()
    got = 0
    for _ in range(measure_batches):
        try:
            next(it)
            got += 1
        except StopIteration:
            break
    steady_batch_s = (time.perf_counter() - t1) / max(1, got)
    logger.info("[timing] steady-state %.3fs/batch", steady_batch_s)
    if steady_batch_s > 1.0:
        logger.warning("Input pipeline is CPU-bound. Consider lower-cost augmentations or different worker/batch settings.")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    epochs: int,
    log_every: int,
    logger: logging.Logger,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    epoch_start = start_time
    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % log_every == 0 or batch_idx == total_batches:
            elapsed = time.time() - start_time
            bps = log_every / elapsed if batch_idx % log_every == 0 else 1.0 / max(elapsed, 1e-6)
            ips = bps * images.size(0)
            eta_s = ((total_batches - batch_idx) / max(bps, 1e-6))
            msg = (
                f"epoch {epoch}/{epochs} | batch {batch_idx}/{total_batches} | "
                f"loss {running_loss / total:.4f} acc {correct / total:.4f} | "
                f"{ips:.1f} img/s | eta {eta_s/60:.1f}m"
            )
            if device.type == "cuda":
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                msg += f" | gpu_mem {mem_gb:.2f}GB"
            logger.info(msg)
            start_time = time.time()

    epoch_elapsed = time.time() - epoch_start
    logger.info("epoch %d training finished in %.1f min", epoch, epoch_elapsed / 60.0)
    return running_loss / total, correct / total


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train RSCD ResNet18 with Windows-friendly logging.")
    p.add_argument("--data-root", type=Path, default=TrainConfig.data_root)
    p.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--prefetch-factor", type=int, default=TrainConfig.prefetch_factor)
    p.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    p.add_argument("--use-pretrained", action="store_true")
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--aug-profile", choices=["default", "fast"], default=TrainConfig.aug_profile)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    a = p.parse_args()
    return TrainConfig(
        data_root=a.data_root,
        output_dir=a.output_dir,
        batch_size=a.batch_size,
        image_size=a.image_size,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        seed=a.seed,
        use_pretrained=a.use_pretrained,
        compile_model=a.compile_model,
        num_workers=a.num_workers,
        prefetch_factor=a.prefetch_factor,
        log_every=a.log_every,
        aug_profile=a.aug_profile,
        max_train_samples=a.max_train_samples,
        max_val_samples=a.max_val_samples,
        max_test_samples=a.max_test_samples,
    )


def main() -> None:
    cfg = parse_args()
    logger = setup_logging(cfg.output_dir)
    logger.info("config: %s", json.dumps(asdict(cfg), default=str))

    train_dir = cfg.data_root / "train"
    val_dir = cfg.data_root / "vali_20k"
    test_dir = cfg.data_root / "test_50k"

    assert train_dir.exists(), f"Missing train directory: {train_dir}"
    assert val_dir.exists(), f"Missing val directory: {val_dir}"
    assert test_dir.exists(), f"Missing test directory: {test_dir}"

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info("gpu: %s | %.1f GB VRAM", props.name, props.total_memory / (1024 ** 3))
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if os.name == "nt":
        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)

    class_names = []
    for p in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
        merged = normalize_label(p.name)
        if merged is not None and merged not in class_names:
            class_names.append(merged)
    num_classes = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    logger.info("classes: %d", num_classes)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if cfg.aug_profile == "fast":
        train_tfms = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        train_tfms = transforms.Compose(
            [
                transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    logger.info("building datasets...")
    t_ds = time.perf_counter()
    base_train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    orig_idx_to_class = {v: k for k, v in base_train_ds.class_to_idx.items()}
    train_ds = MergedImageFolder(base_train_ds, orig_idx_to_class, class_to_idx)
    val_ds = RSCDFlatDataset(val_dir, transform=eval_tfms, class_to_idx=class_to_idx)
    test_ds = RSCDFlatDataset(test_dir, transform=eval_tfms, class_to_idx=class_to_idx)
    logger.info("datasets built in %.1fs", time.perf_counter() - t_ds)

    train_ds_sub = maybe_subset(train_ds, cfg.max_train_samples, cfg.seed)
    val_ds_sub = maybe_subset(val_ds, cfg.max_val_samples, cfg.seed)
    test_ds_sub = maybe_subset(test_ds, cfg.max_test_samples, cfg.seed)
    logger.info("sizes | train=%d val=%d test=%d", len(train_ds_sub), len(val_ds_sub), len(test_ds_sub))

    logger.info("building sampler...")
    t_sampler = time.perf_counter()
    train_targets = torch.as_tensor(get_targets(train_ds_sub), dtype=torch.long)
    class_counts = torch.bincount(train_targets, minlength=num_classes).clamp_min_(1)
    sample_weights = (1.0 / class_counts.float())[train_targets]
    train_sampler = WeightedRandomSampler(sample_weights, sample_weights.numel(), replacement=True)
    class_weights = (class_counts.sum() / class_counts).float()
    class_weights = (class_weights / class_weights.mean()).to(device)
    logger.info("sampler ready in %.1fs", time.perf_counter() - t_sampler)

    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    if cfg.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=cfg.prefetch_factor)
    train_loader = DataLoader(train_ds_sub, sampler=train_sampler, shuffle=False, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds_sub, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds_sub, shuffle=False, **loader_kwargs)
    logger.info(
        "dataloaders | batch=%d workers=%d prefetch=%d",
        cfg.batch_size,
        cfg.num_workers,
        cfg.prefetch_factor,
    )

    benchmark_dataloader(train_loader, logger)

    model = build_model(num_classes, cfg.use_pretrained).to(device, memory_format=torch.channels_last)
    if cfg.compile_model and device.type == "cuda" and hasattr(torch, "compile"):
        logger.info("compiling model...")
        model = torch.compile(model, mode="default")
    else:
        logger.info("model compile disabled")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_acc = 0.0
    metrics_path = cfg.output_dir / "metrics.jsonl"
    logger.info("starting training...")
    train_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg.epochs, cfg.log_every, logger
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "val_acc": float(val_acc),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_row) + "\n")

        logger.info(
            "epoch %d/%d summary | train loss %.4f acc %.4f | val loss %.4f acc %.4f",
            epoch,
            cfg.epochs,
            train_loss,
            train_acc,
            val_loss if val_loss is not None else float("nan"),
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {"model_state": model.state_dict(), "class_to_idx": class_to_idx, "config": asdict(cfg)}
            torch.save(ckpt, cfg.output_dir / "best.pt")
            logger.info("saved new best checkpoint (val_acc=%.4f)", best_val_acc)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info("final | best val acc %.4f | test loss %.4f acc %.4f", best_val_acc, test_loss, test_acc)
    torch.save({"model_state": model.state_dict(), "class_to_idx": class_to_idx, "config": asdict(cfg)}, cfg.output_dir / "last.pt")
    logger.info("training completed in %.1f min", (time.time() - train_start) / 60.0)


if __name__ == "__main__":
    main()
