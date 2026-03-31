import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

JSON_PATH = 'gt_annotations.json'

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
    0: '거북목+왼쪽', 1: '거북목+정면', 2: '거북목+오른쪽',
    3: '경미+왼쪽',   4: '경미+정면',   5: '경미+오른쪽',
    6: '정상+왼쪽',   7: '정상+정면',   8: '정상+오른쪽',
}


class TurtleNeckDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        self.data      = [d for d in all_data if d['has_front']]
        self.transform = transform if transform is not None else default_transform
        print(f"데이터셋 로드: {len(self.data)}장")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        img         = Image.open(entry['front_image']).convert('RGB')
        R           = np.array(entry['rotation_matrix'])
        cont_labels = torch.FloatTensor([entry['pitch_deg'], entry['yaw_deg'], entry['roll_deg']])

        if self.transform:
            img = self.transform(img)

        return img, torch.FloatTensor(R), cont_labels, entry['front_image']


def Identity(cont_labels):
    pitch = cont_labels[0].item() if isinstance(cont_labels, torch.Tensor) else cont_labels[0]
    yaw   = cont_labels[1].item() if isinstance(cont_labels, torch.Tensor) else cont_labels[1]

    if pitch > 0:
        if yaw > 15:    return 0
        elif yaw < -15: return 2
        else:           return 1
    elif pitch > -14:
        if yaw > 15:    return 3
        elif yaw < -15: return 5
        else:           return 4
    else:
        if yaw > 15:    return 6
        elif yaw < -15: return 8
        else:           return 7


def validate_dataset(dataset):
    region_count = {i: 0 for i in range(9)}

    for i in range(len(dataset)):
        _, _, cont_labels, fname = dataset[i]
        region = Identity(cont_labels)
        region_count[region] += 1
        pitch = cont_labels[0].item()
        yaw   = cont_labels[1].item()
        print(f"  [{i+1:3d}] pitch={pitch:+6.1f}  yaw={yaw:+5.1f} -> {region} ({REGION_NAMES[region]})  {os.path.basename(fname)}")

    print(f"\n영역별 분포")
    for r in range(9):
        pct = region_count[r] / len(dataset) * 100
        print(f"  {r} {REGION_NAMES[r]:<14s}  {region_count[r]:3d}장  {pct:5.1f}%")


if __name__ == '__main__':
    dataset = TurtleNeckDataset(JSON_PATH)

    img, gt_R, cont_labels, filename = dataset[0]
    print(f"이미지: {img.shape}")
    print(f"회전행렬: {gt_R.shape}")
    print(f"cont_labels: {cont_labels}")
    print(f"파일: {os.path.basename(filename)}")

    validate_dataset(dataset)
