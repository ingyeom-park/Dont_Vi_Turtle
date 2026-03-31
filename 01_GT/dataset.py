import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

JSON_PATH = "gt_annotations.json"

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

default_transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

REGION_NAMES = {
    0: "거북목+왼쪽",
    1: "거북목+정면",
    2: "거북목+오른쪽",
    3: "경미+왼쪽",
    4: "경미+정면",
    5: "경미+오른쪽",
    6: "정상+왼쪽",
    7: "정상+정면",
    8: "정상+오른쪽",
}


class TurtleNeckDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.data = [row for row in data if row["has_front"]]
        self.transform = transform or default_transform
        print(f"데이터셋 로드: {len(self.data)}장")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img = Image.open(row["front_image"]).convert("RGB")
        r = np.array(row["rotation_matrix"])
        labels = torch.FloatTensor([row["pitch_deg"], row["yaw_deg"], row["roll_deg"]])

        if self.transform:
            img = self.transform(img)

        return img, torch.FloatTensor(r), labels, row["front_image"]


def get_region(labels):
    pitch = labels[0].item() if isinstance(labels, torch.Tensor) else labels[0]
    yaw = labels[1].item() if isinstance(labels, torch.Tensor) else labels[1]

    if pitch > 0:
        if yaw > 15:
            return 0
        if yaw < -15:
            return 2
        return 1

    if pitch > -14:
        if yaw > 15:
            return 3
        if yaw < -15:
            return 5
        return 4

    if yaw > 15:
        return 6
    if yaw < -15:
        return 8
    return 7


def check(ds):
    cnt = {i: 0 for i in range(9)}

    for i in range(len(ds)):
        _, _, labels, path = ds[i]
        region = get_region(labels)
        cnt[region] += 1
        pitch = labels[0].item()
        yaw = labels[1].item()
        name = os.path.basename(path)
        print(f"[{i + 1:3d}] pitch={pitch:+6.1f} yaw={yaw:+5.1f} -> {region} ({REGION_NAMES[region]}) {name}")

    print("\n영역별 분포")
    for i in range(9):
        pct = cnt[i] / len(ds) * 100
        print(f"{i} {REGION_NAMES[i]:<14s} {cnt[i]:3d}장 {pct:5.1f}%")


if __name__ == "__main__":
    ds = TurtleNeckDataset(JSON_PATH)
    img, r, labels, path = ds[0]
    print(f"이미지: {img.shape}")
    print(f"회전행렬: {r.shape}")
    print(f"cont_labels: {labels}")
    print(f"파일: {os.path.basename(path)}")
    check(ds)