import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_Train"))

import dataset
from loss import CVALoss
from model import TokenHPE

BASE_DIR = Path(__file__).resolve().parent


def parse():
    p = argparse.ArgumentParser(description="Train TurtleNeck CVA model with Validation and Test.")
    p.add_argument("--gpu", dest="gpu_id", default=0, type=int)
    p.add_argument("--num_epochs", default=60, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--lr", default=0.0001, type=float)
    p.add_argument("--dataset", default="TurtleNeck", type=str)
    p.add_argument("--data_dir", default="./TurtleNeck_Images", type=str)
    p.add_argument("--train_json", default=str(BASE_DIR / "train_data.json"), type=str)
    p.add_argument("--val_json", default=str(BASE_DIR / "val_data.json"), type=str)
    p.add_argument("--test_json", default=str(BASE_DIR / "test_data.json"), type=str)
    p.add_argument("--weights", default="", type=str)
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    cudnn.enabled = True

    need = [args.train_json, args.val_json, args.test_json]
    if not os.path.isdir(args.data_dir):
        print(f"missing data_dir: {args.data_dir}")
        sys.exit(0)

    miss = [path for path in need if not os.path.isfile(path)]
    if miss:
        print("missing json files:")
        for path in miss:
            print(path)
        sys.exit(0)

    model = TokenHPE(num_ori_tokens=9, depth=3, heads=8, embedding="sine", ViT_weights=args.weights, dim=128)
    for p in model.feature_extractor.parameters():
        p.requires_grad = False

    model.to("cuda")

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tfm = transforms.Compose([
        transforms.Resize(240),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        norm,
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm,
    ])

    train_ds = dataset.getDataset(args.dataset, args.data_dir, args.train_json, train_tfm, train_mode=True)
    val_ds = dataset.getDataset(args.dataset, args.data_dir, args.val_json, val_tfm, train_mode=False)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    crit = CVALoss().cuda(args.gpu_id)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training with validation...")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_mae = 0.0

        for imgs, _, labels, _ in train_loader:
            imgs = imgs.cuda(args.gpu_id)
            labels = labels.cuda(args.gpu_id)
            pred, _ = model(imgs)

            out = crit(pred, labels)
            loss = out[0] if isinstance(out, tuple) else out

            opt.zero_grad()
            loss.backward()
            opt.step()

            a = pred.squeeze()
            b = labels.squeeze()
            if a.dim() == 0:
                a = a.unsqueeze(0)
                b = b.unsqueeze(0)

            train_loss += loss.item()
            train_mse += F.mse_loss(a, b).item()
            train_mae += F.l1_loss(a, b).item()

        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        train_mae /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for imgs, _, labels, _ in val_loader:
                imgs = imgs.cuda(args.gpu_id)
                labels = labels.cuda(args.gpu_id)
                pred, _ = model(imgs)

                out = crit(pred, labels)
                loss = out[0] if isinstance(out, tuple) else out

                a = pred.squeeze()
                b = labels.squeeze()
                if a.dim() == 0:
                    a = a.unsqueeze(0)
                    b = b.unsqueeze(0)

                val_loss += loss.item()
                val_mse += F.mse_loss(a, b).item()
                val_mae += F.l1_loss(a, b).item()

        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_mae /= len(val_loader)

        print(
            f"Epoch [{epoch + 1:02d}/{args.num_epochs:02d}] "
            f"| Train - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f} "
            f"| Val - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}"
        )

    print("Starting test...")
    test_ds = dataset.getDataset(args.dataset, args.data_dir, args.test_json, val_tfm, train_mode=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    model.eval()
    test_loss = 0.0
    test_mse = 0.0
    test_mae = 0.0

    with torch.no_grad():
        for i, (imgs, _, labels, paths) in enumerate(test_loader, 1):
            imgs = imgs.cuda(args.gpu_id)
            labels = labels.cuda(args.gpu_id)
            pred, _ = model(imgs)

            out = crit(pred, labels)
            loss = out[0] if isinstance(out, tuple) else out

            a = pred.squeeze()
            b = labels.squeeze()
            if a.dim() == 0:
                a = a.unsqueeze(0)
                b = b.unsqueeze(0)

            test_loss += loss.item()
            test_mse += F.mse_loss(a, b).item()
            test_mae += F.l1_loss(a, b).item()

            pred_angle = pred.item() * 100.0
            true_angle = labels.item() * 100.0
            name = os.path.basename(paths[0])
            print(f"Test [{i:02d}/{len(test_loader):02d}] {name} | True CVA: {true_angle:.2f}° | Pred CVA: {pred_angle:.2f}°")

    test_loss /= len(test_loader)
    test_mse /= len(test_loader)
    test_mae /= len(test_loader)
    print(f"\n[Final Test Average] Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
