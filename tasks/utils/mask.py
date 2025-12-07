import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology


def overlay_contours(image, mask, color=(0, 255, 0), thickness=2, min_area_ratio=0.002):
    # 获取图像尺寸
    h, w = image.shape[:2]
    total_area = h * w
    min_area = total_area * min_area_ratio

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = image.copy()

    contour_mask = np.zeros((h, w), dtype=np.uint8)

    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            filtered_contours.append(cnt)
            cv2.drawContours(contour_mask, [cnt], -1, 1, -1)

    cv2.drawContours(overlay, filtered_contours, -1, color, thickness)

    return overlay, contour_mask


def get_mask(image):
    if type(image) is str:
        image = cv2.imread(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (45, 45), 0)

    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = morphology.closing(otsu_mask, kernel)
    cleaned_mask = morphology.opening(cleaned_mask, kernel)
    cleaned_mask = cv2.bitwise_not(cleaned_mask)
    contour = overlay_contours(image_rgb, cleaned_mask)
    plt.figure(figsize=(16, 12))
    plt.imshow(contour)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    return contour


def get_mask2(image):
    if type(image) is str:
        image = cv2.imread(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    _, otsu_mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    combined_mask = cv2.bitwise_and(purple_mask, otsu_mask)

    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.GaussianBlur(cleaned_mask, (3, 3), 0)

    mask = overlay_contours(image_rgb, cleaned_mask)
    plt.imshow(mask)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    return mask


def get_mask3(image, output):
    """
    简化版的平滑分割
    """
    if type(image) is str:
        image = cv2.imread(image)
    st = time.time()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # HSV提取紫色
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([100, 40, 40])
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Lab + 大津法
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a, _ = cv2.split(lab)
    _, otsu_mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 结合结果
    combined = cv2.bitwise_and(purple_mask, otsu_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.medianBlur(smoothed, 25)
    # print(f'{time.time() - st}')
    contour_image, contour_mask = overlay_contours(image_rgb, smoothed)

    plt.figure(figsize=(16, 12))

    plt.subplot(1, 2, 1)
    plt.imshow(contour_image)
    plt.title('Image with Contours')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(contour_mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.tight_layout()  # 自动调整子图间距
    plt.show()
    # cv2.imwrite(output, contour_mask)
    return contour_mask


import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def process_file_wrapper(args):
    """包装函数供map使用"""
    file, img_dir, mask_dir = args
    try:
        img_path = os.path.join(img_dir, file)
        output_path = os.path.join(mask_dir, file)
        get_mask3(img_path, output_path)
        return True
    except Exception:
        return False


def parallel_process_map(img_dir, mask_dir, max_workers=20):
    """使用map方法的极简版本"""
    os.makedirs(mask_dir, exist_ok=True)

    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

    args_list = [(file, img_dir, mask_dir) for file in files]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_file_wrapper, args_list),
            total=len(files),
            desc="处理文件"
        ))

    successful = sum(results)
    print(f"完成: {successful}/{len(files)} 个文件处理成功")



if __name__ == '__main__':
    img_dir = [
        '/NAS2/Data1/lbliao/Data/MXB/segment/第一批/dataset/1024/train/images',
        '/NAS2/Data1/lbliao/Data/MXB/segment/第一批/dataset/1024/val/images',
        '/NAS2/Data1/lbliao/Data/MXB/segment/第一批/dataset/1024/test/images',
        '/NAS2/Data1/lbliao/Data/MXB/segment/YNZL修订/dataset/1024/train/images',
        '/NAS2/Data1/lbliao/Data/MXB/segment/YNZL修订/dataset/1024/val/images',
        '/NAS2/Data1/lbliao/Data/MXB/segment/YNZL修订/dataset/1024/test/images',
        '/NAS2/Data1/lbliao/Data/MXB/segment/补充神经节标注/dataset/1024/train/images',
        '/NAS2/Data1/lbliao/Data/MXB/segment/补充神经节标注/dataset/1024/val/images',
        '/NAS2/Data1/lbliao/Data/MXB/segment/补充神经节标注/dataset/1024/test/images',
    ]
    mask_dir = [
        '/NAS2/Data1/lbliao/Data/MXB/segment/第一批/dataset/1024/train/otsu',
        '/NAS2/Data1/lbliao/Data/MXB/segment/第一批/dataset/1024/val/otsu',
        '/NAS2/Data1/lbliao/Data/MXB/segment/第一批/dataset/1024/test/otsu',
        '/NAS2/Data1/lbliao/Data/MXB/segment/YNZL修订/dataset/1024/train/otsu',
        '/NAS2/Data1/lbliao/Data/MXB/segment/YNZL修订/dataset/1024/val/otsu',
        '/NAS2/Data1/lbliao/Data/MXB/segment/YNZL修订/dataset/1024/test/otsu',
        '/NAS2/Data1/lbliao/Data/MXB/segment/补充神经节标注/dataset/1024/train/otsu',
        '/NAS2/Data1/lbliao/Data/MXB/segment/补充神经节标注/dataset/1024/val/otsu',
        '/NAS2/Data1/lbliao/Data/MXB/segment/补充神经节标注/dataset/1024/test/otsu',
    ]

    # for img, mask in zip(img_dir, mask_dir):
    #     # os.makedirs(mask, exist_ok=True)
    #     parallel_process_map(img, mask)
    for file in os.listdir(img_dir[0]):
        img_path = os.path.join(img_dir[0], file)
        output_path = os.path.join(mask_dir[0], file)
        get_mask3(img_path, output_path)
