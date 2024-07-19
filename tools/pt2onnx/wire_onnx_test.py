import sys, os

import argparse

import matplotlib.pyplot as plt
import torch
import sys
import torch.nn.functional as F
from tqdm import tqdm

from tools.pt2onnx.onnxmodel import ONNXModel
import cv2
import numpy as np


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_data", type=str, default="/media/xin/data/data/seg_data/ours/test_data/0705_cam1/0714_test.txt",help="")
    parser.add_argument("--model", type=str, default="/media/xin/work/github_pro/seg_model/PIDNet/tools/pt2onnx/PIDNet_wire_256x640_p6.onnx",help="")
    parser.add_argument('--dataset', type=str, default='wire',help='dataset name (default: citys)')
    parser.add_argument('--show_rgb', action='store_true',default=False)
    args = parser.parse_args()
    return args


def img_resize(image,target_size):
    H,W,_ = image.shape
    t_h = target_size[0] * 2
    diff_h = H - t_h
    cropped_img = image[diff_h:H,:]
    resize_img = cv2.resize(cropped_img,(target_size[1],target_size[0]))
    return resize_img


def img_crop(image,target_size):
    H,W,_ = image.shape
    diff_h = H - target_size[0]
    cropped_img = image[diff_h:,:]
    return cropped_img


def test_onnx(img_path, model_file,show_rgb=True):
    color_map = [(128, 64, 128),
                 (128, 128, 128)]
    model = ONNXModel(model_file)
    img_org = cv2.imread(img_path)
    if "224" in model_file:
        img_res = img_resize(img_org,(224,320))
    elif "480" in model_file:
        img_res = img_org
    else:
        img_res = img_crop(img_org,(256,640))
    img = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
    # 将图像转换为 float32 类型并归一化到 [0, 1]
    image = img.astype(np.float32) / 255.0

    # 定义均值和标准差
    mean = [0.485, 0.456, 0.406] # 123.675  116.28  103.53
    std = [0.229, 0.224, 0.225] # 58.395  57.12  57.375

    # 归一化
    image -= mean
    image /= std
    img = image.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype("float32")

    output = model.forward(img)
    pred = output[0]
    if not show_rgb:
        cv_image = (pred * 255).astype(np.uint8)
        img_org = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        # new_array = np.full((224, 640), 255, dtype=np.uint8)
        # # 垂直拼接数组
        # result = np.vstack((new_array, cv_image))
        # res_path = os.path.join("/media/xin/work/github_pro/seg_model/PIDNet/data/test_data/res",os.path.basename(img_path))
        # cv2.imwrite(res_path,result)
    else:
        cv_image = np.zeros_like(image).astype(np.uint8)
        for i, color in enumerate(color_map):
            for j in range(3):
                cv_image[:, :, j][pred == i] = color_map[i][j]
        img_org = img_res
    # 水平方向合并两张图像
    merged_image = np.hstack((img_org, cv_image))
    return merged_image

def main():
    args = GetArgs()
    img_path = args.input_data
    show_rgb = args.show_rgb
    if img_path.endswith(".txt"):
        with open(img_path) as f:
            files = f.readlines()
            for img in tqdm(files,desc='Processing'):
                merged_image = test_onnx(img.strip(), args.model,show_rgb=show_rgb)
                root_path = "/media/xin/data/data/seg_data/ours/test_data/0705_cam1"
                img_name = os.path.basename(img.strip())
                save_img_path = os.path.join(root_path, "PIDNet_p6", img_name)
                if not os.path.exists(os.path.dirname(save_img_path)):
                    os.makedirs(os.path.dirname(save_img_path))
                cv2.imwrite(save_img_path, merged_image)
                # cv2.imshow('Converted Image', merged_image)
                # cv2.waitKey(500)
    else:
        merged_image = test_onnx(img_path,args.model,show_rgb=show_rgb)
        cv2.imshow('Converted Image', merged_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
