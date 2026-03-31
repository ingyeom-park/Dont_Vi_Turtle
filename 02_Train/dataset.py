import json
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data.dataset import Dataset


class TurtleNeckDataset(Dataset):
    def __init__(self, data_dir, json_path, transform=None, image_mode="RGB", train_mode=True):
        self.data_dir = data_dir
        self.image_mode = image_mode
        self.train_mode = train_mode

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        data = []
        miss = 0

        for row in raw:
            name = row.get("front", row.get("front_image", ""))
            if not name or "없음" in name:
                continue

            path = os.path.join(self.data_dir, os.path.basename(name))
            if os.path.exists(path):
                data.append(row)
            else:
                miss += 1
                if miss <= 5:
                    print(f"[경고] 파일을 찾을 수 없음: {path}")

        if miss:
            print(f"총 {miss}개의 이미지를 폴더에서 찾지 못해 제외되었습니다. 경로 확인 필요")

        self.data = data
        self.length = len(data)

        if train_mode:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform or T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        print(f"[필터링 완료] 유효 데이터: {self.length}개")

    def __getitem__(self, idx):
        row = self.data[idx]
        name = os.path.basename(row.get("front", row.get("front_image")))
        path = os.path.join(self.data_dir, name)

        img = Image.open(path).convert(self.image_mode)
        if self.transform:
            img = self.transform(img)

        angle = float(row["cva_angle"]) / 100.0
        label = torch.FloatTensor([angle])
        dummy = torch.eye(3).float()
        return img, dummy, label, path

    def __len__(self):
        return self.length


def getDataset(dataset, data_dir, filename_list, transformations, train_mode=True):
    return TurtleNeckDataset(data_dir, filename_list, transformations, train_mode=train_mode)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    json_paths = [
        base / "master_annotations.json",
        base.parent / "01_GT" / "GT_train.json",
        base.parent / "01_GT" / "gt_annotations.json",
    ]
    data_paths = [
        base / "TurtleNeck_Images",
        base.parent / "00_dataset",
    ]

    json_path = next((path for path in json_paths if path.exists()), None)
    data_dir = next((path for path in data_paths if path.exists()), None)

    if not json_path or not data_dir:
        print("dataset.py ready")
    else:
        ds = TurtleNeckDataset(str(data_dir), str(json_path), train_mode=False)
        print(f"dataset size: {len(ds)}")
