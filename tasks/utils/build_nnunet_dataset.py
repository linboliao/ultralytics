import os
import yaml
import shutil
from multiprocessing import Pool, cpu_count
from typing import List, Tuple


def collect_tasks(folders: List[str],
                  dst_images_path: str,
                  dst_masks_path: str) -> List[Tuple[str, str, str, str, str]]:
    """
    收集所有需要复制的文件任务
    Returns:
        [(src_img, src_mask, dst_img, dst_mask, case_id), ...]
    """
    tasks = []

    for folder in folders:
        folder = folder.replace('/images', '')
        folder_name = os.path.basename(folder.rstrip('/\\'))

        src_images = os.path.join(folder, "images")
        src_masks = os.path.join(folder, "masks")

        if not os.path.isdir(src_images) or not os.path.isdir(src_masks):
            continue

        for filename in sorted(os.listdir(src_images)):
            src_img = os.path.join(src_images, filename)
            src_mask = os.path.join(src_masks, filename)

            if not (os.path.isfile(src_img) and os.path.isfile(src_mask)):
                continue

            name, ext = os.path.splitext(filename)

            new_img_name = f"{folder_name}_{name}_0000{ext}"
            new_mask_name = f"{folder_name}_{name}{ext}"

            dst_img = os.path.join(dst_images_path, new_img_name)
            dst_mask = os.path.join(dst_masks_path, new_mask_name)

            # 防止覆盖
            if os.path.exists(dst_img) or os.path.exists(dst_mask):
                continue

            case_id = f"{folder_name}_{name}"
            tasks.append((src_img, src_mask, dst_img, dst_mask, case_id))

    return tasks


def copy_single_file(task: Tuple[str, str, str, str, str]) -> str:
    src_img, src_mask, dst_img, dst_mask, case_id = task

    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_mask, dst_mask)

    return case_id


def build_nnunet_dataset(yaml_path: str,
                         dst_base_path: str,
                         phase: str,
                         num_workers: int = None):
    """
    根据 YAML 构建 nnUNet 数据集

    Args:
        yaml_path: 配置文件路径
        dst_base_path: nnUNet Dataset 根目录
        phase: train / val / test
        num_workers: 并行进程数
    """

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    folders = config.get(phase, [])
    if not folders:
        print(f"No folders found for phase: {phase}")
        return

    # nnUNet 目录
    if phase == "train":
        dst_images = os.path.join(dst_base_path, "imagesTr")
        dst_masks = os.path.join(dst_base_path, "labelsTr")
    else:
        dst_images = os.path.join(dst_base_path, "imagesTs")
        dst_masks = os.path.join(dst_base_path, "labelsTs")

    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_masks, exist_ok=True)

    # 收集任务
    tasks = collect_tasks(folders, dst_images, dst_masks)

    print(f"Total files to copy: {len(tasks)}")

    if not tasks:
        return

    if num_workers is None:
        num_workers = min(cpu_count(), len(tasks))

    # 并行复制
    with Pool(processes=num_workers) as pool:
        case_ids = list(pool.imap_unordered(copy_single_file, tasks))

    # 保存索引文件
    case_ids = sorted(case_ids)
    with open(f"{phase}.txt", "w") as f:
        for cid in case_ids:
            f.write(f"{cid}\n")

    print(f"{phase}: copied {len(case_ids)} files successfully.")


if __name__ == "__main__":
    yaml_file = "tasks/cfg/datasets/PGlandSeg.yaml"
    destination_path = "/NAS145/liaolinbo/Data/GlandSeg/nnUnet/raw/Dataset005_PGlandSeg"

    build_nnunet_dataset(
        yaml_path=yaml_file,
        dst_base_path=destination_path,
        phase="train"
    )