# 过滤相似度
import os
import random

import numpy as np
import shutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from matplotlib import image as mpimg, pyplot as plt
from skimage.metrics import structural_similarity as ssim


# 相似度过滤
def process_file(filename, he_dir, ihc_dir, dst_he_dir, dst_ihc_dir):
    he_path = os.path.join(he_dir, filename)
    ihc_path = os.path.join(ihc_dir, filename)

    img1 = Image.open(he_path)
    img2 = Image.open(ihc_path)
    img1_gray = img1.convert('L')  # 'L'模式表示灰度图
    img2_gray = img2.convert('L')
    img1_gray_np = np.array(img1_gray)
    img2_gray_np = np.array(img2_gray)
    ssim_value = ssim(img1_gray_np, img2_gray_np)

    if ssim_value > 0.11:
        dst_he_path = os.path.join(dst_he_dir, filename)
        dst_ihc_path = os.path.join(dst_ihc_dir, filename)
        shutil.copy(he_path, dst_he_path)
        shutil.copy(ihc_path, dst_ihc_path)

    return ssim_value, filename


def parallel_run():
    he_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/tmp/he/'
    ihc_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/tmp/ihc/'
    dst_he_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/dst/he/'
    dst_ihc_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/dst/ihc/'
    he_dir = r'/data2/lbliao/Data/MSI/pair/1024/MLH1/trainA-0/'
    ihc_dir = r'/data2/lbliao/Data/MSI/pair/1024/MLH1/trainB-0/'
    dst_he_dir = r'//data2/lbliao/Data/MSI/pair/1024/MLH1/trainA/'
    dst_ihc_dir = r'/data2/lbliao/Data/MSI/pair/1024/MLH1/trainB/'
    os.makedirs(dst_he_dir, exist_ok=True)
    os.makedirs(dst_ihc_dir, exist_ok=True)
    results = []

    # 创建一个线程池
    with ThreadPoolExecutor(max_workers=20) as executor:
        # 提交任务到线程池
        future_to_file = {executor.submit(process_file, filename, he_dir, ihc_dir, dst_he_dir, dst_ihc_dir): filename for filename in os.listdir(he_dir)}

        # 等待任务完成并处理结果
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                ssim_value, _ = future.result()
                results.append(ssim_value)
                logger.info(f"file: {filename}, ssim: {ssim_value}")
            except Exception as exc:
                logger.error(f"file: {filename} generated an exception: {exc}")


# parallel_run()


# 过滤没有腺体的图片
def ann_filter():
    he_dir = r'/data2/lbliao/Data/MXB/Segement/dataset/2048-3/val/images/'
    # ihc_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/dst/ihc/'
    ann = r'//data2/lbliao/Data/MXB/Segement/dataset/2048-3/val/labels/'
    hes = os.listdir(he_dir)
    anns = os.listdir(ann)
    for he in hes:
        base, ext = os.path.splitext(he)
        ann_path = os.path.join(ann, f'{base}.txt')
        if f'{base}.txt' not in anns:
            with open(ann_path, 'w') as file:
                pass
            os.remove(os.path.join(he_dir, he))
            # os.remove(os.path.join(ihc_dir, he))
            logger.info(f'{he}没有腺体')
        ann_path = os.path.join(ann, f'{base}.txt')
        if not os.path.getsize(ann_path) > 0 and random.random() > 0.1:
            os.remove(os.path.join(he_dir, he))
            os.remove(ann_path)
            logger.info(f'{he}没有腺体')
# ann_filter()

# 图像大小过滤
def size_filter():
    he_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/tmp/he/'
    ihc_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/tmp/ihc/'
    he_dir = r'/data2/lbliao/Data/MXB/Segement/pair/2048/CK/he/'
    ihc_dir = r'/data2/lbliao/Data/MXB/Segement/pair/2048/CK/ihc/'
    ann_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/tmp/label/'

    # 遍历目录
    for filename in os.listdir(he_dir):
        base, _ = os.path.splitext(filename)
        he_path = os.path.join(he_dir, filename)
        ihc_path = os.path.join(ihc_dir, filename)
        # ann_path = os.path.join(ann_dir, f"{base}.txt")

        if os.path.isfile(he_path) and os.path.isfile(ihc_path):
            filesize = os.path.getsize(ihc_path)

            # 如果文件大小小于100KB，则删除
            if filesize < 3500 * 1024:
                os.remove(he_path)
                os.remove(ihc_path)
                # if os.path.isfile(ann_path):
                #     os.remove(ann_path)
                logger.info(f"Deleted {filename} due to size less than 1100KB")


def size_filter1():
    file_dir = r'/data2/lbliao/Data/MXB/Segement/dataset/2048/val/images/'
    ann_dir = r'/data2/lbliao/Data/MXB/Segement/dataset/2048/val/labels/'
    for filename in os.listdir(file_dir):
        base, _ = os.path.splitext(filename)
        file_path = os.path.join(file_dir, filename)
        ann_path = os.path.join(ann_dir, f"{base}.txt")
        if os.path.isfile(file_path):
            filesize = os.path.getsize(file_path)
            if filesize < 1100 * 1024:
                os.remove(file_path)
                os.remove(ann_path)
                logger.info(f"Deleted {filename} due to size less than 1100KB")


# size_filter()


# 指定你的文件夹路径
def split_data():
    he_dir = r'/NAS2/Data1/lbliao/Data/MXB/Detection/cellvit+/dataset/images'
    ann_path = r'/NAS2/Data1/lbliao/Data/MXB/Detection/cellvit+/dataset/labels'

    img_dir = r'/NAS2/Data1/lbliao/Data/MXB/Detection/cellvit+/dataset/'
    os.makedirs(img_dir, exist_ok=True)

    he_imgs = os.listdir(he_dir)
    random.shuffle(he_imgs)
    length = len(he_imgs)
    train = int(length * 0.7)
    for i in range(train):
        base, ext = os.path.splitext(he_imgs[i])
        # img_low = os.path.join(img_dir, 'train/images_low')
        img = os.path.join(img_dir, 'train/images')
        # img_high = os.path.join(img_dir, 'train/images_high')
        label = os.path.join(img_dir, 'train/labels')
        # os.makedirs(img_low, exist_ok=True)
        os.makedirs(img, exist_ok=True)
        # os.makedirs(img_high, exist_ok=True)
        os.makedirs(label, exist_ok=True)
        if os.path.exists(os.path.join(img, he_imgs[i])):
            continue
        shutil.copy(os.path.join(he_dir, he_imgs[i]), os.path.join(img, he_imgs[i]))
        # shutil.copy(os.path.join(he_dir.replace('/images', '/images_low'), he_imgs[i]), os.path.join(img_low, he_imgs[i]))
        # shutil.copy(os.path.join(he_dir.replace('/images', '/images_high'), he_imgs[i]), os.path.join(img_high, he_imgs[i]))
        shutil.copy(os.path.join(ann_path, f'{base}.txt'), os.path.join(label, f'{base}.txt'))

    for i in range(train, length):
        base, ext = os.path.splitext(he_imgs[i])
        img = os.path.join(img_dir, 'val/images')
        # img_low = os.path.join(img_dir, 'val/images_low')
        # img_high = os.path.join(img_dir, 'val/images_high')
        label = os.path.join(img_dir, 'val/labels')
        os.makedirs(img, exist_ok=True)
        # os.makedirs(img_low, exist_ok=True)
        # os.makedirs(img_high, exist_ok=True)
        os.makedirs(label, exist_ok=True)
        shutil.copy(os.path.join(he_dir, he_imgs[i]), os.path.join(img, he_imgs[i]))
        # shutil.copy(os.path.join(he_dir.replace('/images', '/images_low'), he_imgs[i]), os.path.join(img_low, he_imgs[i]))
        # shutil.copy(os.path.join(he_dir.replace('/images', '/images_high'), he_imgs[i]), os.path.join(img_high, he_imgs[i]))
        shutil.copy(os.path.join(ann_path, f'{base}.txt'), os.path.join(label, f'{base}.txt'))


# size_filter()
# parallel_run()


def img_resize():
    path = f'/NAS2/Data1/lbliao/Data/MXB/Seg-Relabel/patch/2048/image/'
    imgs = os.listdir(path)
    for img in imgs:
        try:
            img_path = os.path.join(path, img)
            img = Image.open(img_path)
            img = img.resize((1024, 1024))
            img.save(img_path)
        except Exception as exc:
            os.remove(img_path)


# img_resize()



# split_data()


def deal():
    path = f'/data2/lbliao/Data/MXB/Segement/dataset/2048/test/labels/'
    new_path = f'/data2/lbliao/Data/MXB/Segement/dataset/2048/test/labels-n/'

    os.makedirs(new_path, exist_ok=True)
    labels = os.listdir(path)
    for label in labels:
        label_path = os.path.join(path, label)
        new_label_path = os.path.join(new_path, label)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            with open(new_label_path, 'w') as new_f:
                for line in lines:
                    data = line.split(' ')
                    clazz = data[0]
                    if int(clazz) == 2:
                        continue
                    coords = []
                    for i in range(1, len(data), 2):
                        if 0 < float(data[i]) < 1 and 0 < float(data[i + 1]) < 1:
                            coords.append(float(data[i]))
                            coords.append(float(data[i + 1]))
                    if len(coords) > 6 and len(coords) % 2 == 0:
                        contours_str = ' '.join(map(str, coords))
                        line = f'{clazz} {contours_str}'
                        new_f.write(line + '\n')


# deal()


def resize():
    path = f'/data2/lbliao/Data/MXB/Segement/dataset/2048/yolo-detect/test/image/'
    out_path = f'/data2/lbliao/Data/MXB/Segement/dataset/2048/yolo-detect/test1/image/'
    os.makedirs(out_path, exist_ok=True)
    imgs = os.listdir(path)
    for img in imgs:
        img_path = os.path.join(path, img)
        out_img_path = os.path.join(out_path, img)
        img = Image.open(img_path)
        img = img.resize((1024, 1024))
        img.save(out_img_path, 'JPEG', quality=95)


# resize()

# def test():
#     A = r'/data2/lbliao/Data/MSI/pair/1024/MLH1/trainA/'
#     B = r'/data2/lbliao/Data/MSI/pair/1024/MLH1/trainB/'
#     A0 = r'/data2/lbliao/Data/MSI/pair/1024/MLH1/trainA-0/'
#     B0 = r'/data2/lbliao/Data/MSI/pair/1024/MLH1/trainB-0/'
#     b0_images = os.listdir(B0)
#     b_images = os.listdir(B)
#     for img in b0_images:
#         if img not in b_images:
#             img1 = mpimg.imread(os.path.join(A0, img))  # 替换为你的图片文件路径
#             img2 = mpimg.imread(os.path.join(B0, img))  # 替换为你的图片文件路径
#
#             # 创建一个1行2列的子图布局
#             fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1行2列
#
#             # 在第一个子图中显示第一张图片
#             axs[0].imshow(img1)
#             axs[0].set_title('Image 1')  # 设置标题
#             axs[0].axis('off')  # 不显示坐标轴
#
#             # 在第二个子图中显示第二张图片
#             axs[1].imshow(img2)
#             axs[1].set_title('Image 2')  # 设置标题
#             axs[1].axis('off')  # 不显示坐标轴
#
#             # 调整子图之间的间距
#             plt.tight_layout()
#
#             # 显示图形
#             plt.show()
#
# dir = '/data2/lbliao/Data/MXB/Segement/annotation/'
# file = os.listdir(dir)
# random.shuffle(file)
# print(file)
def my_test():
    patch_dir = '/NAS2/Data1/lbliao/Data/MXB/Segement/dataset/2048-4/'
    img_dir = os.path.join(patch_dir, 'images')
    ann_dir = os.path.join(patch_dir, 'lm_annotations')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    files = os.listdir(patch_dir)
    for file in files:
        file_path = os.path.join(patch_dir, file)
        if file.endswith('.json'):
            os.rename(file_path, os.path.join(ann_dir, file))
        elif file.endswith('.png'):
            os.rename(file_path, os.path.join(img_dir, file))

split_data()
# my_test()