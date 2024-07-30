# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm
import models
import torch
import torch.nn.functional as F
from PIL import Image

from models.pidnet import PIDNet

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 128, 128),
             (128, 64, 128)]


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-s', type=str)
    parser.add_argument('--c', help='cityscapes pretrained or not', type=bool, default=True)
    parser.add_argument('--p', help='dir for pretrained model',
                        default='/media/xin/work/github_pro/seg_model/PIDNet/runs/p0/wire/pidnet_small_wire/best.pt', type=str)
    # parser.add_argument('--p', help='dir for pretrained model', default='../pretrained_models/camvid/PIDNet_S_Camvid_Test.pt', type=str)
    parser.add_argument('--r', help='file for input images', default='/media/xin/data/data/seg_data/ours/test_data/cam1_test/test_0724/test_300.txt', type=str)
    args = parser.parse_args()

    return args


def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if
                       (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    return model


if __name__ == '__main__':
    args = parse_args()
    # sv_path = args.r+'outputs/'
    sv_path = "/media/xin/work/github_pro/seg_model/PIDNet/data/test_data/outputs_wire1"

    model = PIDNet(m=2, n=3, num_classes=2, planes=32, ppm_planes=96, head_planes=128, augment=False,is_test=True)
    model = load_pretrained(model, args.p).cuda()
    model.eval()
    with torch.no_grad():
        with open(args.r) as f:
            images_list = f.readlines()
        for img_path in tqdm(images_list):
            img = cv2.imread(img_path.strip(),
                             cv2.IMREAD_COLOR)
            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)
            # pred = F.interpolate(pred, size=img.size()[-2:],
            #                      mode='bilinear', align_corners=True)
            # pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            pred = pred.cpu().numpy()
            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:, :, j][pred == i] = color_map[i][j]
            sv_img = Image.fromarray(sv_img)

            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            img_name = os.path.basename(img_path.strip())
            sv_img.save(sv_path + "/" + img_name)
            break




