import cv2
import numpy as np
import os
from tqdm import tqdm


def modify_image_pixels(input_dir,crop_pixels=40):
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    for file_path in tqdm(image_files, desc="Processing images"):
        img = cv2.imread(file_path)
        if img is None:
            print(f"图像加载失败：{file_path}")
            continue

        # 裁剪图像下边缘40个像素
        cropped_image = img[:-crop_pixels, :]

        cv2.imwrite(file_path, cropped_image)

if __name__ == '__main__':
    # 调用函数并传入路径
    input_directory = '/media/xin/data/data/seg_data/ours/origins/0725_wire/0725_select'
    modify_image_pixels(input_directory)