"""
将像素值不是0的改为【128, 64, 128】，是0的改为【128，128，128】
"""

import cv2
import numpy as np
import os
from tqdm import tqdm


def modify_image_pixels(input_dir):
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

        # 修改像素值
        modified_img = np.where(np.all(img == [0, 0, 0], axis=-1, keepdims=True),
                                [128, 128, 128],
                                [128, 64, 128])

        cv2.imwrite(file_path, modified_img)

if __name__ == '__main__':
    # 调用函数并传入路径
    input_directory = '/media/xin/data/data/seg_data/ours/train_data/gtFine/train'
    modify_image_pixels(input_directory)

