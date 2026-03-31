import json
import random
import os

JSON_PATH  = 'gt_annotations.json'
TRAIN_PATH = 'GT_train.json'
TEST_PATH  = 'GT_test.json'


def split_dataset(json_path, train_path, test_path, train_ratio=0.8, seed=42):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(seed)
    random.shuffle(data)

    split      = int(len(data) * train_ratio)
    train_data = data[:split]
    test_data  = data[split:]

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"전체: {len(data)}장")
    print(f"train: {len(train_data)}장 -> {os.path.basename(train_path)}")
    print(f"test:  {len(test_data)}장  -> {os.path.basename(test_path)}")

    return train_data, test_data


if __name__ == '__main__':
    split_dataset(JSON_PATH, TRAIN_PATH, TEST_PATH)
