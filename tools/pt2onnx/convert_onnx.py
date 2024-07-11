import sys, os
import argparse
import torch
from models.pidnet import PIDNet


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weights", type=str, default='/media/xin/work/github_pro/seg_model/PIDNet/output/wire1/pidnet_small_wire/best.pt',help="model path")
    parser.add_argument("--output", type=str, default="PIDNet_wire_256x640.onnx",help="output model path")
    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-s', type=str)
    parser.add_argument('--c', help='cityscapes pretrained or not', type=str, default="wire1")

    args = parser.parse_args()
    return args


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


def main():
    # H, W = 1024, 2048
    H, W = 256, 640
    args = GetArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_type = {"cityscapes":19,"camvid":11,"wire1":2}
    model = PIDNet(m=2, n=3, num_classes=data_type[args.c], planes=32, ppm_planes=96, head_planes=128, augment=False,is_test=True)
    model = load_pretrained(model, args.weights).to(device)
    model.eval()
    # adaptive_avg_pool2d
    onnx_input = torch.rand(1, 3, H, W)
    onnx_input = onnx_input.to(device)
    torch.onnx.export(model,
                      onnx_input,
                      args.output,
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])


if __name__ == '__main__':
    main()
