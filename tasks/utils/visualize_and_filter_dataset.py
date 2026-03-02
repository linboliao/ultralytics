import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

base = '/NAS145/liaolinbo/Data/GlandSeg/gleason/dataset/train'
IMAGES_DIR = os.path.join(base, 'images')
MASKS_DIR = os.path.join(base, 'masks')
TARGET_FOLDER = '/NAS145/liaolinbo/Data/GlandSeg/gleason/dataset3/train'

# 创建目标文件夹
os.makedirs(os.path.join(TARGET_FOLDER, 'images'), exist_ok=True)
os.makedirs(os.path.join(TARGET_FOLDER, 'masks'), exist_ok=True)

# 类别颜色（RGBA）
CLASS_COLORS = {
    0: [0, 0, 0, 0],         # 背景
    1: [255, 0, 0, 128],     # benign
    2: [0, 255, 0, 128],     # grade 3
    3: [0, 0, 255, 128],     # grade 4
    4: [255, 255, 0, 128]    # grade 5
}

CLASS_LABELS = {
    0: 'bg',
    1: 'benign',
    2: 'grade 3',
    3: 'grade 4',
    4: 'grade 5'
}

# 读取图片文件（支持大小写）
image_files = [f for f in os.listdir(IMAGES_DIR)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

image_files.sort()

print(f"共找到 {len(image_files)} 张图片")

# 遍历图片
for i, filename in enumerate(image_files):

    print(f"\n[{i+1}/{len(image_files)}] 处理: {filename}")

    image_path = os.path.join(IMAGES_DIR, filename)
    mask_path = os.path.join(MASKS_DIR, filename)

    # 检查 mask 是否存在
    if not os.path.exists(mask_path):
        print(f"⚠ 未找到对应 mask: {filename}，跳过")
        continue

    # 读取图像
    img = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path).convert('L'))  # 强制单通道

    # 创建 overlay
    overlay = img.copy().astype(np.float32)

    # 叠加颜色
    for class_id, color in CLASS_COLORS.items():

        if class_id == 0:
            continue

        mask_indices = mask == class_id

        if np.any(mask_indices):
            rgba_color = np.array(color[:3]) / 255.0
            alpha = color[3] / 255.0

            overlay[mask_indices] = (
                overlay[mask_indices] * (1 - alpha)
                + rgba_color * 255 * alpha
            )

    overlay = overlay.astype(np.uint8)

    # 显示图像
    plt.figure(figsize=(12, 5))

    # 原图
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # 叠加图
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title('Image + Mask')
    plt.axis('off')

    # 添加图例
    legend_elements = []
    for class_id in range(1, 5):
        if np.any(mask == class_id):
            color = np.array(CLASS_COLORS[class_id][:3]) / 255.0
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1,
                              facecolor=color,
                              label=CLASS_LABELS[class_id])
            )

    if legend_elements:
        plt.legend(handles=legend_elements,
                   loc='upper center',
                   bbox_to_anchor=(0.5, -0.05),
                   ncol=2,
                   fontsize=8)

    plt.tight_layout()
    plt.show()

    # 用户输入
    key = input("按 1 复制到新数据集，其他键跳过: ").strip()

    # 按 1 才复制
    if key == "1":

        dst_image = os.path.join(TARGET_FOLDER, 'images', filename)
        dst_mask = os.path.join(TARGET_FOLDER, 'masks', filename)

        shutil.copy(image_path, dst_image)
        shutil.copy(mask_path, dst_mask)

        print(f"✅ 已复制 {filename} 到新数据集")

    else:
        print("跳过")

    plt.close()

print("\n处理完成！")