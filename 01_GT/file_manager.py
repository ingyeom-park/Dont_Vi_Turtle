import os
import re
import shutil

DATA_DIR = "00_dataset"
PATTERN = re.compile(r"^([a-zA-Z]+)_(front|side)_(\d{3})\.jpg$")


def rename(path, code=11):
    rows = []
    for name in os.listdir(path):
        m = PATTERN.match(name)
        if not m:
            continue

        user, view, num = m.groups()
        new = f"{user}_{view}_{code}_{num}.jpg"
        os.rename(os.path.join(path, name), os.path.join(path, new))
        rows.append((name, new))

    print(f"변경 완료: {len(rows)}장")
    for old, new in rows:
        print(f"  {old} -> {new}")

    return rows


def count(path):
    n = len(os.listdir(path))
    print(f"총 {n}장")
    return n


def compare(a, b):
    left = set(os.listdir(a))
    right = set(os.listdir(b))

    only_left = left - right
    only_right = right - left

    print(f"base:   {len(left)}장")
    print(f"target: {len(right)}장")
    print(f"\nbase에만 있는 파일: {len(only_left)}장")
    for name in sorted(only_left):
        print(f"  {name}")

    print(f"\ntarget에만 있는 파일: {len(only_right)}장")
    for name in sorted(only_right):
        print(f"  {name}")

    return only_left, only_right


def sync(src, dst):
    src_names = set(os.listdir(src))
    dst_names = set(os.listdir(dst))
    miss = src_names - dst_names

    for name in sorted(miss):
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
        print(f"복사: {name}")

    print(f"완료: {dst} 총 {len(os.listdir(dst))}장")


def find_dups(a, b):
    rows = set(os.listdir(a)) & set(os.listdir(b))
    print(f"중복 파일: {len(rows)}장")
    for name in sorted(rows):
        print(f"  {name}")
    return rows


def find_only(a, b):
    rows = set(os.listdir(a)) - set(os.listdir(b))
    print(f"{a}에만 있는 파일: {len(rows)}장")
    for name in sorted(rows):
        print(f"  {name}")
    return rows


def find_top(path):
    rows = [name for name in os.listdir(path) if "_top_" in name]
    print(f"top 파일: {len(rows)}장")
    for name in sorted(rows):
        print(f"  {name}")
    return rows


def rm(path):
    os.remove(path)
    print(f"삭제: {path}")


if __name__ == "__main__":
    count(DATA_DIR)
    find_top(DATA_DIR)
