import argparse
import copy
import glob
import json
import os
import shutil
import sys

import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms


def lr_process(args):
    input_dir = args.input_dir 
    output_dir = args.output_dir
    resolution = args.d_resolution
    target_resolution = args.target_resolution

    for obj_name in os.listdir(input_dir):
        if args.obj_name is not None and obj_name not in args.obj_name:
            continue

        obj_dir = os.path.join(input_dir, obj_name)
        image_dirs = os.path.join(obj_dir, 'images_8')
        image_output_dir = os.path.join(output_dir, obj_name, 'images_8')
        image_output_ori_dir = os.path.join(output_dir, obj_name, 'images')
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(image_output_ori_dir, exist_ok=True)

        # process image
        image_path_list = glob.glob(os.path.join(image_dirs, '*.png')) 
        for img_path in image_path_list:
            img = Image.open(img_path)
            # if args.centre_crop:
            img = transforms.CenterCrop(min(img.size))(img)
            img_lr = img.resize((resolution, resolution), Image.BICUBIC)
            img_lr_upsampled = img_lr.resize((target_resolution, target_resolution), Image.BICUBIC)
            img_ori = img_lr.resize((target_resolution * 8, target_resolution * 8), Image.BICUBIC)
            img_lr_upsampled.save(os.path.join(image_output_dir, os.path.basename(img_path)))
            img_ori.save(os.path.join(image_output_ori_dir, os.path.basename(img_path)))
        
        # process npy 
        data = np.load(os.path.join(obj_dir, 'poses_bounds.npy'))
        poses = data[:, :-2].reshape(-1, 3, 5)
        hwf = poses[0, :3, -1] / 8
        height, width, focal = hwf
        scale = min(height, width) / target_resolution
        focal /= scale
        height = target_resolution
        width = target_resolution
        new_hwf = np.array([height, width, focal])
        poses[:, :3, -1] = new_hwf * 8
        poses = poses.reshape(-1, 15)
        data[:, :-2] = poses
        np.save(os.path.join(output_dir, obj_name, 'poses_bounds.npy'), data)
        
            
def parse_args():
    parser = argparse.ArgumentParser(description='Blender LR image processor')
    parser.add_argument('--input_dir', required=True, type=str, help='input image directory')
    parser.add_argument('--output_dir', required=True, type=str, help='output image directory')
    parser.add_argument('--obj_name', nargs='+', default=None, type=str, help='object name')
    parser.add_argument('--d_resolution', default=64, type=int, help='downsample resolution')
    parser.add_argument('--target_resolution', default=512, type=int, help='target resolution')
    # parser.add_argument('--centre_crop', action='store_true', help='whether to centre crop')
    args = parser.parse_args()
    return args


"""
"""

if __name__ == '__main__':

    args = parse_args()
    lr_process(args)
    