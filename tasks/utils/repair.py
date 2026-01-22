import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import argparse


def remove_white_background_from_foreground(original_image, mask, mthresh=13, sthresh=17, sthresh_up=255, close=4, use_otsu=False):
    """
    参考HSV处理流程，将mask前景区域中原本是白色背景的部分改为背景（mask值设为0）

    参数:
        original_image: 原始RGB图像
        mask: 分割掩码，0=背景，1=cls1，2=cls2
        mthresh: 中值滤波核大小
        sthresh: 饱和度阈值下限
        sthresh_up: 饱和度阈值上限
        close: 形态学闭运算核大小
        use_otsu: 是否使用大津法

    返回:
        处理后的mask
    """
    # 确保mask是整数类型
    mask = mask.astype(np.uint8)

    # 转换为RGB格式（如果输入是BGR）
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = original_image

    # 参考您提供的HSV处理流程
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)  # 转换为HSV空间
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # 对饱和度通道进行中值滤波

    # 阈值处理
    if use_otsu:
        _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

    # 形态学闭运算
    if close > 0:
        kernel = np.ones((close, close), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    # img_otsu中白色区域(255)表示有足够饱和度的区域（非白色背景）
    # 我们需要的是白色背景区域，所以取反
    non_white_background_mask = img_otsu > 0  # 非白色背景区域

    # 获取前景区域（mask中不为0的区域）
    foreground_mask = mask > 0

    # 找到前景区域中同时是白色背景的区域（非饱和区域）
    # 注意：这里我们取反，因为img_otsu检测的是非白色区域
    white_background_in_foreground = np.logical_and(foreground_mask, ~non_white_background_mask)

    # 将这些区域在mask中设为背景（0）
    processed_mask = mask.copy()
    processed_mask[white_background_in_foreground] = 0

    return processed_mask


def save_visualization_result(image, mask, save_path):
    """
    只保存叠加效果图（简化版）

    Args:
        image: 原始图片（BGR格式）
        mask: 生成的mask
        save_path: 保存路径
    """
    # 确保图片是uint8格式
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # 创建叠加图
    overlay_img = image.copy()

    # 直接混合颜色（50%原图 + 50%颜色）
    # 类别1：绿色（BGR: [0,255,0]）
    overlay_img[mask == 1] = (overlay_img[mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

    # 类别2：红色（BGR: [0,0,255]）
    overlay_img[mask == 2] = (overlay_img[mask == 2] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

    cv2.imwrite(save_path, overlay_img)


def process_single_image(img_name, image_folder, mask_folder, vis_folder, overwrite_mask=True):
    """
    处理单张图片的完整流程

    Args:
        img_name: 图片文件名
        image_folder: 图片文件夹路径
        mask_folder: mask文件夹路径
        vis_folder: 可视化文件夹路径
        overwrite_mask: 是否覆盖原mask文件

    Returns:
        tuple: (图片名称, 处理状态, 错误信息)
    """
    try:
        img_path = os.path.join(image_folder, img_name)
        mask_path = os.path.join(mask_folder, img_name)
        vis_path = os.path.join(vis_folder, img_name)

        # 检查文件是否存在
        if not os.path.exists(img_path):
            return img_name, "failed", f"图片文件不存在: {img_path}"
        if not os.path.exists(mask_path):
            return img_name, "failed", f"mask文件不存在: {mask_path}"

        # 读取图片和mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return img_name, "failed", f"无法读取图片: {img_path}"
        if mask is None:
            return img_name, "failed", f"无法读取mask: {mask_path}"

        # 处理mask
        processed_mask = remove_white_background_from_foreground(image, mask)

        # 确保可视化文件夹存在
        os.makedirs(vis_folder, exist_ok=True)

        # 保存可视化结果
        save_visualization_result(image, processed_mask, vis_path)

        # 覆盖原mask文件
        if overwrite_mask:
            cv2.imwrite(mask_path, processed_mask)

        return img_name, "success", None

    except Exception as e:
        return img_name, "failed", f"处理图片时出错: {str(e)}"


def process_images_in_parallel(image_folder, mask_folder, vis_folder, max_workers=None, use_processes=False, overwrite_mask=True):
    """
    并行处理所有图片

    Args:
        image_folder: 图片文件夹路径
        mask_folder: mask文件夹路径
        vis_folder: 可视化文件夹路径
        max_workers: 最大工作线程/进程数（None则自动检测）
        use_processes: 是否使用进程池（默认使用线程池）
        overwrite_mask: 是否覆盖原mask文件
    """
    # 获取图片列表
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not images:
        print("未找到图片文件")
        return

    print(f"找到 {len(images)} 张图片，开始并行处理...")

    # 选择执行器类型
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    with Executor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [
            executor.submit(process_single_image, img, image_folder, mask_folder, vis_folder, overwrite_mask)
            for img in images
        ]

        # 使用tqdm显示进度
        success_count = 0
        failed_count = 0

        for future in tqdm(futures, desc="处理图片", unit="张"):
            try:
                img_name, status, error_msg = future.result()
                if status == "success":
                    success_count += 1
                else:
                    failed_count += 1
                    print(f"\n处理失败: {img_name} - {error_msg}")
            except Exception as e:
                failed_count += 1
                print(f"\n处理异常: {e}")

    print(f"\n处理完成！成功: {success_count} 张，失败: {failed_count} 张")


if __name__ == "__main__":
    # 或者直接使用硬编码路径（注释掉上面的命令行代码，取消下面的注释）
    image_folder = '/NAS145/liaolinbo/Data/GlandSeg/maixin-2/dataset/test/images'
    mask_folder = '/NAS145/liaolinbo/Data/GlandSeg/maixin-2/dataset/test/masks'
    vis_folder = '/NAS145/liaolinbo/Data/GlandSeg/maixin-2/dataset/test/vis'
    max_workers = None
    use_processes = False
    overwrite_mask = True

    # 并行处理
    process_images_in_parallel(
        image_folder=image_folder,
        mask_folder=mask_folder,
        vis_folder=vis_folder,
        max_workers=max_workers,
        use_processes=use_processes,
        overwrite_mask=overwrite_mask
    )
