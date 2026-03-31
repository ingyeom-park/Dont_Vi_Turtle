import json
import os
import random

JSON_PATH = "gt_annotations.json"
TRAIN_PATH = "GT_train.json"
VAL_PATH = "GT_val.json"
TEST_PATH = "GT_test.json"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        row
        for row in data
        if row.get("has_front", True)
        and row.get("front_image", "")
        and "없음" not in row.get("front_image", "")
    ]


def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split(path=JSON_PATH, train_path=TRAIN_PATH, test_path=TEST_PATH, ratio=0.8, seed=42):
    data = load(path)
    random.seed(seed)
    random.shuffle(data)

    cut = int(len(data) * ratio)
    train = data[:cut]
    test = data[cut:]

    save(train_path, train)
    save(test_path, test)

    print(f"전체: {len(data)}장")
    print(f"train: {len(train)}장 -> {os.path.basename(train_path)}")
    print(f"test:  {len(test)}장 -> {os.path.basename(test_path)}")

    return train, test


def split3(
    path=JSON_PATH,
    train_path=TRAIN_PATH,
    val_path=VAL_PATH,
    test_path=TEST_PATH,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    total_ratio = round(train_ratio + val_ratio + test_ratio, 5)
    if total_ratio != 1.0:
        raise ValueError("train/val/test ratio must sum to 1.0")

    data = load(path)
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    cut1 = int(n * train_ratio)
    cut2 = cut1 + int(n * val_ratio)

    train = data[:cut1]
    val = data[cut1:cut2]
    test = data[cut2:]

    save(train_path, train)
    save(val_path, val)
    save(test_path, test)

    print(f"전체: {n}장")
    print(f"train: {len(train)}장 -> {os.path.basename(train_path)}")
    print(f"val:   {len(val)}장 -> {os.path.basename(val_path)}")
    print(f"test:  {len(test)}장 -> {os.path.basename(test_path)}")

    return train, val, test


if __name__ == "__main__":
    split()
