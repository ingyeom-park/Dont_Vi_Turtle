import argparse
import os
import sys

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

import dataset
from loss import CVALoss
from model import TokenHPE


def parse():
    p = argparse.ArgumentParser(description="Train TurtleNeck CVA model.")
    p.add_argument("--gpu", dest="gpu_id", default=0, type=int)
    p.add_argument("--num_epochs", default=60, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--lr", default=0.00001, type=float)
    p.add_argument("--dataset", default="TurtleNeck", type=str)
    p.add_argument("--data_dir", default="./TurtleNeck_Images", type=str)
    p.add_argument("--filename_list", default="./master_annotations.json", type=str)
    p.add_argument("--snapshot", default="", type=str)
    p.add_argument("--weights", default="", type=str)
    p.add_argument("--describe", default="CVA_Predictor", type=str)
    p.add_argument("--output_string", default="TurtleNeck", type=str)
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    cudnn.enabled = True

    if not os.path.isdir(args.data_dir):
        print(f"missing data_dir: {args.data_dir}")
        sys.exit(0)

    if not os.path.isfile(args.filename_list):
        print(f"missing filename_list: {args.filename_list}")
        sys.exit(0)

    out_dir = os.path.join(
        "output",
        "snapshots",
        f"TokenHPE_{args.describe}_batch_size{args.batch_size}",
    )
    os.makedirs(out_dir, exist_ok=True)

    model = TokenHPE(
        num_ori_tokens=9,
        depth=3,
        heads=8,
        embedding="sine",
        ViT_weights=args.weights,
        dim=128,
    )

    state = None
    if args.snapshot:
        state = torch.load(args.snapshot)
        model.load_state_dict(state["model_state_dict"])
        print("Intermediate weights used!")

    model.to("cuda")
    print("Loading data and preprocessing...")

    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    tfm = transforms.Compose([
        transforms.Resize(240),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        norm,
    ])

    ds = dataset.getDataset(args.dataset, args.data_dir, args.filename_list, tfm)
    loader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
    )

    crit = CVALoss().cuda(args.gpu_id)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    if state is not None:
        opt.load_state_dict(state["optimizer_state_dict"])

    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 40], gamma=0.5)

    print("Starting training.")
    for epoch in range(args.num_epochs):
        for i, (imgs, _, labels, _) in enumerate(loader, 1):
            imgs = imgs.cuda(args.gpu_id)
            labels = labels.cuda(args.gpu_id)

            pred, _ = model(imgs)
            out = crit(pred, labels)
            loss = out[0] if isinstance(out, tuple) else out

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 2 == 0:
                total = len(ds) // args.batch_size + 1
                print(f"Epoch [{epoch + 1}/{args.num_epochs}], Iteration [{i}/{total}] MSE Loss: {loss.item():.4f}")

        sched.step()

        path = os.path.join(out_dir, f"{args.output_string}_epoch_{epoch + 1}.tar")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            },
            path,
        )
        print(f"Saved: {path}")
