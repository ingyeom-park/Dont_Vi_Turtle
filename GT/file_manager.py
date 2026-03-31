import os
import re
import shutil

DATASET_FOLDER = 'dataset'


def rename_files(folder_path, code_number=11):
    pattern = re.compile(r'^([a-zA-Z]+)_(front|side)_(\d{3})\.jpg$')
    renamed = []

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            name, direction, number = match.groups()
            new_filename = f"{name}_{direction}_{code_number}_{number}.jpg"
            os.rename(
                os.path.join(folder_path, filename),
                os.path.join(folder_path, new_filename),
            )
            renamed.append(f"{filename} -> {new_filename}")

    print(f"변경 완료: {len(renamed)}장")
    for r in renamed:
        print(f"  {r}")


def count_files(folder_path):
    count = len(os.listdir(folder_path))
    print(f"총 {count}장")
    return count


def compare_folders(base_folder, target_folder):
    base_files   = set(os.listdir(base_folder))
    target_files = set(os.listdir(target_folder))

    only_in_base   = base_files - target_files
    only_in_target = target_files - base_files

    print(f"base:   {len(base_files)}장")
    print(f"target: {len(target_files)}장")
    print(f"\nbase에만 있는 파일: {len(only_in_base)}장")
    for f in sorted(only_in_base):
        print(f"  {f}")
    print(f"\ntarget에만 있는 파일: {len(only_in_target)}장")
    for f in sorted(only_in_target):
        print(f"  {f}")

    return only_in_base, only_in_target


def sync_folders(src_folder, dst_folder):
    src_files = set(os.listdir(src_folder))
    dst_files = set(os.listdir(dst_folder))
    to_copy   = src_files - dst_files

    for filename in sorted(to_copy):
        shutil.copy2(
            os.path.join(src_folder, filename),
            os.path.join(dst_folder, filename),
        )
        print(f"복사: {filename}")

    print(f"완료: {dst_folder} 총 {len(os.listdir(dst_folder))}장")


def find_duplicates(folder_a, folder_b):
    dups = set(os.listdir(folder_a)) & set(os.listdir(folder_b))
    print(f"중복 파일: {len(dups)}장")
    for f in sorted(dups):
        print(f"  {f}")
    return dups


def find_unique_files(folder_a, folder_b):
    unique = set(os.listdir(folder_a)) - set(os.listdir(folder_b))
    print(f"{folder_a}에만 있는 파일: {len(unique)}장")
    for f in sorted(unique):
        print(f"  {f}")
    return unique


def find_top_files(folder_path):
    top_files = [f for f in os.listdir(folder_path) if '_top_' in f]
    print(f"top 파일: {len(top_files)}장")
    for f in sorted(top_files):
        print(f"  {f}")
    return top_files


def delete_file(file_path):
    os.remove(file_path)
    print(f"삭제: {file_path}")


if __name__ == '__main__':
    count_files(DATASET_FOLDER)
    find_top_files(DATASET_FOLDER)