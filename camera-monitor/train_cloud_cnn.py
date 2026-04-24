#!/usr/bin/env python3
"""
Train a small CNN on the dataset built by cloud_label_pipeline.py.

The model has two heads sharing a ResNet18 backbone:
  - classification head: predicts the cloud_class (CrossEntropy)
  - regression head:     predicts rain_pct in 0..100 (MSE on /100 target)

Usage:
    pip install torch torchvision pandas pillow
    python train_cloud_cnn.py --dataset-dir dataset --epochs 20
"""
import argparse
import json
import os
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class CloudDataset(Dataset):
    def __init__(self, root: str, split: str, classes):
        self.root = root
        self.classes = classes
        manifest = pd.read_csv(os.path.join(root, "manifest.csv"))
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)

        if split == "train":
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, float]:
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.root, row["image_path"])).convert("RGB")
        x = self.tf(img)
        y_cls = int(row["cloud_class_idx"])
        y_rain = float(row["rain_pct"]) / 100.0  # normalize to 0..1 for stable MSE
        return x, y_cls, y_rain


class CloudNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.cls_head = nn.Linear(in_feat, num_classes)
        self.rain_head = nn.Sequential(
            nn.Linear(in_feat, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.cls_head(f), self.rain_head(f).squeeze(-1)


def run_epoch(model, loader, opt, device, train: bool, w_rain: float):
    model.train(train)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    total, n_correct, n_seen, sum_mae = 0.0, 0, 0, 0.0
    for x, y_cls, y_rain in loader:
        x = x.to(device, non_blocking=True)
        y_cls = y_cls.to(device, non_blocking=True)
        y_rain = y_rain.float().to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits, rain_pred = model(x)
            loss = ce(logits, y_cls) + w_rain * mse(rain_pred, y_rain)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total += loss.item() * x.size(0)
        n_correct += (logits.argmax(1) == y_cls).sum().item()
        n_seen += x.size(0)
        sum_mae += (rain_pred - y_rain).abs().sum().item() * 100.0  # back to 0..100

    return total / n_seen, n_correct / n_seen, sum_mae / n_seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rain-weight", type=float, default=1.0,
                   help="Weight of the rain regression loss vs classification loss")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", default="cloud_cnn.pt")
    p.add_argument("--export-onnx", default="cloud_cnn.onnx",
                   help="Also export the best checkpoint to ONNX for RKNN conversion. Empty to skip.")
    args = p.parse_args()

    with open(os.path.join(args.dataset_dir, "classes.json")) as f:
        classes = json.load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}, classes: {classes}")

    train_ds = CloudDataset(args.dataset_dir, "train", classes)
    val_ds = CloudDataset(args.dataset_dir, "val", classes)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = CloudNet(num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_mae = run_epoch(model, train_dl, opt, device, True, args.rain_weight)
        va_loss, va_acc, va_mae = run_epoch(model, val_dl, opt, device, False, args.rain_weight)
        print(f"ep {ep:03d}  train loss {tr_loss:.3f} acc {tr_acc:.3f} rain_mae {tr_mae:.1f}%  "
              f"|  val loss {va_loss:.3f} acc {va_acc:.3f} rain_mae {va_mae:.1f}%")
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "classes": classes}, args.out)
            print(f"  saved -> {args.out}")

    if args.export_onnx:
        ckpt = torch.load(args.out, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval().to("cpu")
        dummy = torch.zeros(1, 3, 224, 224)
        torch.onnx.export(
            model, dummy, args.export_onnx,
            input_names=["input"], output_names=["logits", "rain_pct"],
            opset_version=13, do_constant_folding=True,
            dynamic_axes=None,  # RKNN prefers static shapes
        )
        print(f"exported -> {args.export_onnx}  (input 1x3x224x224, normalized ImageNet stats)")


if __name__ == "__main__":
    main()
