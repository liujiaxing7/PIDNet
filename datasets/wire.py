# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from .base_dataset import BaseDataset


class Wire(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_classes=2,
                 multi_scale=True,
                 flip=True,
                 ignore_label=255,
                 base_size=640,
                 crop_size=(480, 640),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4
                 ):

        super(Wire, self).__init__(ignore_label, base_size,
                                     crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip() for line in open(os.path.join(root,list_path))]

        self.files = self.read_files()

        self.ignore_label = ignore_label

        self.color_list = [[128, 128, 128],[128, 64, 128]]

        # self.class_weights = None
        self.class_weights = torch.FloatTensor([0.9, 1.1]).cuda()


        self.bd_dilate_size = bd_dilate_size

    def read_files(self):
        files = []

        for image_path in self.img_list:
            label_path = image_path.replace("leftImg8bit/train","gtFine/mask_color").replace("jpg","png")
            # label_path = image_path.replace("leftImg8bit","gtFine").replace("jpg","png")
            name = os.path.join("data/wire",self.list_path)
            if os.path.exists(label_path):
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })

        return files

    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2]) * self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2) == 3] = i

        return label.astype(np.uint8)

    def label2color(self, label):
        color_map = np.zeros(label.shape + (3,))
        for i, v in enumerate(self.color_list):
            color_map[label == i] = self.color_list[i]

        return color_map.astype(np.uint8)

    def get_crop_img(self, H,img,size=400):
        # 裁剪图像
        if H == 480:
            crop_box = (0, H-size, img.size[0], img.size[1])
            cropped_img = img.crop(crop_box)
            return cropped_img
        elif H == size:
            return img
        elif H == 360:
            expanded_img = ImageOps.expand(img, border=(0, 40, 0, 0), fill=(0, 0, 0))
            return expanded_img
        else:
            raise "图像尺寸不符合要求"


    def __getitem__(self, index):
        item = self.files[index]
        name = item["img"]
        image = Image.open(item["img"]).convert('RGB')
        image = self.get_crop_img(image.size[1],image)
        image = np.array(image)
        size = image.shape
        color_map = Image.open(item["label"]).convert('RGB')
        color_map = self.get_crop_img(color_map.size[1],color_map)
        color_map = np.array(color_map)
        label = self.color2label(color_map)
        image, label, edge = self.gen_sample(image, label,
                                             self.multi_scale, self.flip, edge_pad=False,
                                             edge_size=self.bd_dilate_size, city=False)
        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))



