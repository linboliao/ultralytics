import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

base = '/NAS145/liaolinbo/Data/GlandSeg/gleason/dataset/train'
IMAGES_DIR = base + '/images'
MASKS_DIR = base + '/masks'
TARGET_FOLDER = '/NAS145/liaolinbo/Data/GlandSeg/gleason/dataset2/train'

# 确保目标文件夹存在
os.makedirs(os.path.join(TARGET_FOLDER, 'images'), exist_ok=True)
os.makedirs(os.path.join(TARGET_FOLDER, 'masks'), exist_ok=True)

# 定义不同类别的颜色和标签
CLASS_COLORS = {
    0: [0, 0, 0, 0],  # 背景：透明
    1: [255, 0, 0, 128],  # 类别1：红色半透明
    2: [0, 255, 0, 128],  # 类别2：绿色半透明
    3: [0, 0, 255, 128],  # 类别3：蓝色半透明
    4: [255, 255, 0, 128]  # 类别4：黄色半透明
}

CLASS_LABELS = {
    0: 'bg',
    1: 'benign',
    2: 'grade 3',
    3: 'grade 4',
    4: 'grade 5'
}

# 获取所有图片文件名
image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()

# 遍历每张图片
for i, filename in enumerate(image_files):
    # 加载图片和mask
    img = np.array(Image.open(os.path.join(IMAGES_DIR, filename)))
    mask = np.array(Image.open(os.path.join(MASKS_DIR, filename)))

    # 创建叠加图
    overlay = img.copy().astype(np.float32)

    # 为不同类别应用不同颜色
    for class_id, color in CLASS_COLORS.items():
        if class_id == 0:  # 背景跳过
            continue
        mask_indices = mask == class_id
        if np.any(mask_indices):
            # 将RGBA颜色转换为0-1范围
            rgba_color = np.array(color[:3]) / 255.0
            alpha = color[3] / 255.0

            # 应用颜色叠加
            overlay[mask_indices] = (overlay[mask_indices] * (1 - alpha) +
                                     rgba_color * 255 * alpha)

    # 显示图片
    plt.figure(figsize=(12, 5))

    # 左侧：原图
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('images')
    plt.axis('off')

    # 右侧：叠加图
    plt.subplot(1, 2, 2)
    plt.imshow(overlay.astype(np.uint8))
    plt.title('image + Mask')
    plt.axis('off')

    # 添加图例（放在右侧图片上方）
    legend_elements = []
    for class_id in range(1, 5):  # 只显示类别1-4，跳过背景
        if np.any(mask == class_id):  # 只显示当前图片中存在的类别
            color = np.array(CLASS_COLORS[class_id][:3]) / 255.0
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color,
                                                 label=CLASS_LABELS[class_id]))

    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right',
                   bbox_to_anchor=(1.0, -0.1), ncol=2, fontsize=8)

    # 显示提示信息
    # plt.suptitle(f'{filename}  ({i + 1}/{len(image_files)})\n按1: 移动到新数据集, 其他键: 跳过')
    plt.tight_layout()
    plt.show()

    # 等待键盘输入
    key = input().strip()

    # 如果输入1，移动文件
    if key != "1":
        # 移动mask文件
        src_mask = os.path.join(MASKS_DIR, filename)
        dst_mask = os.path.join(TARGET_FOLDER, 'masks', filename)
        shutil.copy(src_mask, dst_mask)

        # 移动图片文件
        src_image = os.path.join(IMAGES_DIR, filename)
        dst_image = os.path.join(TARGET_FOLDER, 'images', filename)
        shutil.copy(src_image, dst_image)

        print(f'已移动 {filename} 到新数据集')

    plt.close()  # 关闭当前图片，显示下一张

print('处理完成！')