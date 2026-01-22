import os
import sys
from PIL import Image
import tkinter as tk
from tkinter import messagebox


def show_images_with_delete_option(folder_path):
    """
    显示文件夹下的所有图片，每显示一张：
    - 输入1：删除图片
    - 输入其他或不输入：进入下一张
    """

    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    # 获取文件夹下所有图片文件
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"在文件夹 '{folder_path}' 中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片")
    print("操作说明：")
    print("- 输入 '1' 然后按回车：删除当前图片")
    print("- 直接按回车或其他输入：保留图片并进入下一张")
    print("- 输入 'q' 或 'quit'：退出程序")
    print("-" * 50)

    # 创建临时Tkinter根窗口（用于图片显示，不显示主窗口）
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    deleted_count = 0
    processed_count = 0

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img_path = os.path.join(folder_path.replace('vis', 'images'), image_file)
        masks_path = os.path.join(folder_path.replace('vis', 'masks'), image_file)
        labels_path = os.path.join(folder_path.replace('vis', 'labels'), image_file.replace('png', 'txt'))
        try:
            # 打开图片
            img = Image.open(image_path)

            # 显示图片信息
            print(f"\n[{processed_count + 1}/{len(image_files)}] 当前图片: {image_file}")
            print(f"尺寸: {img.size[0]}x{img.size[1]}")

            # 显示图片（在新窗口中）
            img.show(title=f"图片预览: {image_file}")

            # 获取用户输入
            while True:
                user_input = input("请输入操作 (1=删除, 其他=保留, q=退出): ").strip().lower()

                if user_input == '1':
                    # 删除图片
                    os.remove(image_path)
                    os.remove(img_path)
                    os.remove(masks_path)
                    os.remove(labels_path)
                    print(f"✓ 已删除: {image_file}")
                    deleted_count += 1
                    break
                elif user_input in ('q', 'quit'):
                    print("用户主动退出程序")
                    # 关闭所有图片窗口
                    root.quit()
                    return
                elif user_input == '' or user_input:
                    print(f"✓ 保留: {image_file}")
                    break

            processed_count += 1

        except Exception as e:
            print(f"✗ 处理图片 {image_file} 时出错: {e}")
            continue

    # 关闭Tkinter
    root.quit()

    print(f"\n" + "=" * 50)
    print(f"处理完成！")
    print(f"总图片数: {len(image_files)}")
    print(f"已处理: {processed_count}")
    print(f"已删除: {deleted_count}")
    print(f"保留: {processed_count - deleted_count}")


if __name__ == "__main__":
    # 设置要浏览的文件夹路径
    folder_path = '/NAS145/liaolinbo/Data/GlandSeg/maixin-2/dataset/val/vis'

    show_images_with_delete_option(folder_path)
