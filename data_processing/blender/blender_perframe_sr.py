import argparse
import copy
import glob
import json
import os
import shutil
import sys

import numpy as np
import PIL.Image as Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline, IFPipeline, IFSuperResolutionPipeline
from diffusers.utils import pt_to_pil
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, T5Tokenizer


class blenderDataset(Dataset):
    def __init__(self, input_dir, args) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.args = args
        self.get_mate_data()

    def get_mate_data(self):
        self.image_paths = []
        self.obj_names = os.listdir(self.input_dir)
        self.mate_data = []
        for obj_name in self.obj_names:
            if obj_name not in self.args.obj_name:
                continue
            image_paths = glob.glob(os.path.join(self.input_dir, obj_name, self.args.split, '*.png'))
            image_paths = list(filter(lambda x: 'depth' not in os.path.basename(x) and 'normal' not in os.path.basename(x), image_paths))
            image_paths.sort()
            for i, img_p in enumerate(image_paths):
                print(img_p)
                data_dict = {
                    'image_path': img_p, 
                    'obj_name': obj_name,
                    'idx': i,
                }
                self.mate_data.append(data_dict)

    
    def __len__(self):
        return len(self.mate_data)
    
    def __getitem__(self, index):
        img = Image.open(self.mate_data[index]['image_path']).convert('RGBA')
        white_background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        white_background.paste(img, mask=img.split()[3]) 
        img = white_background.convert('RGB')
        img = transforms.Resize((64, 64), Image.BICUBIC)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
        data_dict = self.mate_data[index]
        data_dict['image'] = img
        return data_dict


def read_prompt_from_file(file_path):
    prompt_list = []
    with open(file_path, 'r') as f:
        for prompt_line in f.readlines():   
            prompt_line = prompt_line.lstrip().rstrip()
            prompt_list.append(prompt_line)

    return prompt_list


def image_sr(rank, world_size, args):

    print(f"rank is {rank}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='env://')

    stage_1_id = "DeepFloyd/IF-I-XL-v1.0"
    stage_2_id = "DeepFloyd/IF-II-L-v1.0"
    # tokenizer = DiffusionPipeline.from_pretrained(stage_1_id, subfolder='tokenizer', cache_dir='/data5/wuzhongkai/diffusers_models', torch_dtype=torch.float16, local_files_only=True)
    # encoder = DiffusionPipeline.from_pretrained(stage_1_id, subfolder='text_encoder', cache_dir='/data5/wuzhongkai/diffusers_models', torch_dtype=torch.float16, local_files_only=True)
    # stage_1 = IFPipeline(tokenizer=tokenizer, text_encoder=encoder).to(rank)
    stage_1 = IFPipeline.from_pretrained(stage_1_id, variant="fp16", torch_dtype=torch.float16, cache_dir='/data5/wuzhongkai/diffusers_models', local_files_only=True).to(rank)
    stage_1.enable_model_cpu_offload(gpu_id=rank)
    stage_2 = DiffusionPipeline.from_pretrained(stage_2_id, text_encoder=None, variant="fp16", torch_dtype=torch.float16, cache_dir='/data5/wuzhongkai/diffusers_models', local_files_only=True).to(rank)
    stage_2.enable_model_cpu_offload(gpu_id=rank)

    if args.prompt_mode == 'single':
        prompt = args.prompt
        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt) 
        print("prompt : {}".format(prompt))
    elif args.prompt_mode == 'file':
        prompt_list = read_prompt_from_file(args.prompt_file_path)
        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt_list)
        print(f"prompt list : {prompt_list}")
        print(f"prompt_embed.shape {prompt_embeds.shape}")

    del stage_1

    input_dir = args.input_dir 
    output_dir = args.output_dir

    train_dataset = blenderDataset(input_dir, args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=4)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4)

    dataloader_list = [
        train_dataloader,
        # test_dataloader,
        # val_dataloader,
    ]
    print(len(train_dataset))

    print(len(train_dataset))
    for dataloader in dataloader_list:
        for data in dataloader:
            image = data['image']
            image_path = data['image_path']
            obj_names = data['obj_name']
            if args.prompt_mode == 'single':
                prompt_embeds_batch = prompt_embeds.repeat(len(image), 1, 1)
                negative_embeds_batch = negative_embeds.repeat(len(image), 1, 1)
            # for per_view_idx in range(args.image_per_view):
                # if args.prompt_mode == 'file':
                #     prompt_embeds_batch = prompt_embeds[per_view_idx : per_view_idx + 1].repeat(len(image), 1, 1)
                #     negative_embeds_batch = negative_embeds[per_view_idx : per_view_idx + 1].repeat(len(image), 1, 1)
            upscaled_image = stage_2(
                image=image, prompt_embeds=prompt_embeds_batch, negative_prompt_embeds=negative_embeds_batch,  output_type="pt",
                guidance_scale=args.guidance_scale, noise_level=args.noise_level, num_inference_steps=args.num_inference_steps
            ).images
            for i in range(upscaled_image.shape[0]):
                single_upscaled_image = pt_to_pil(upscaled_image)[i]
                # os.makedirs(output_dir, exist_ok=True)
                os.makedirs(os.path.join(output_dir, obj_names[i], args.split), exist_ok=True)
                single_upscaled_image.save(os.path.join(output_dir, obj_names[i], args.split, f'{os.path.basename(image_path[i])}'))


# def copy_json(args):

#     input_dir = args.input_dir 
#     output_dir = args.output_dir

#     json_output_dict = {
#         "labels": {}
#     }

#     # cal intrinsic matrix 
#     camera_angle = 0.6911112070083618
#     fov = np.degrees(camera_angle)
#     intrinsic_matrix = np.zeros((3, 3))
#     intrinsic_matrix[2, 2] = 1.0
#     intrinsic_matrix[0, 0] = intrinsic_matrix[1, 1] = 1.0 / (2 * np.tan(fov * np.pi / 360))
#     intrinsic_matrix[0, 2] = intrinsic_matrix[1, 2] = 0.5
#     intrinsic_matrix = intrinsic_matrix.reshape(9).tolist()

#     json_path = os.path.join(input_dir, 'dataset_blender_r4.03.json')
#     with open(json_path, 'r') as f:
#         meta_data = json.load(f)
#     camera_paths = meta_data["camera_path"]
#     for i, cp in enumerate(camera_paths):
#         camera_params = cp['camera_to_world']
#         camera_params = np.array(camera_params).reshape(4, 4)
#         # flip the x and z axis to adapt to the eg3d camera coordinate system
#         # camera_params[1, :] = -camera_params[1, :]
#         # camera_params[2, :] = -camera_params[2, :]
#         camera_params[:, 1] = -camera_params[:, 1]
#         camera_params[:, 2] = -camera_params[:, 2]
#         camera_params = camera_params.reshape(16).tolist()
#         # add instrinsic matrix
#         camera_params.extend(intrinsic_matrix)

#         for per_view_idx in range(args.image_per_view):
#             # json_output_dict["labels"][f"{os.path.basename(cp['image_path']).rstrip('.png')}_{per_view_idx}.png"] \
#             #     = camera_params
#             json_output_dict['labels'][f'{i:05d}_{per_view_idx}.png'] = camera_params

#     # json_output_dict['cam_radius'] = meta_data['cam_radius']

#     for obj_name in os.listdir(input_dir):
#         if obj_name not in args.obj_name:
#             continue
#         obj_dir = os.path.join(output_dir, obj_name)
#         print(obj_dir)

#         os.makedirs(obj_dir, exist_ok=True)
#         with open(os.path.join(obj_dir, 'dataset.json'), 'w') as jfp:
#             json.dump(json_output_dict, jfp, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description='Blender 64x64 to 256x256 using floyd')
    parser.add_argument('--input_dir', required=True, type=str, help='input image directory')
    parser.add_argument('--output_dir', required=True, type=str, help='output image directory')
    parser.add_argument('--guidance_scale', default=0.0, type=float, help='guidance scale for floyd')
    parser.add_argument('--noise_level', default=0, type=int, help='noise level for floyd')
    parser.add_argument('--num_inference_steps', default=50, type=int, help='num inference steps for floyd')
    parser.add_argument('--image_per_view', type=int, default=50, help='number of images per view')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--world_size', type=int, default=4, help='world size use in ddp')
    parser.add_argument('--port', type=int, default=5678, help='world size use in ddp')
    parser.add_argument('--prompt_mode', type=str, default="single", choices=['file', 'single'], help="in which mode to get prompt, choise from 'file' or 'single'")
    parser.add_argument('--prompt', type=str, default="", help='prompt for floyd sr model')
    parser.add_argument('--prompt_file_path', type=str, help='when using prompt_mode, specify a txt file to read the prompt list')
    parser.add_argument('--obj_name', type=str, nargs='+', help='which obj to sr')
    parser.add_argument('--split', type=str, default='train')

    args = parser.parse_args()
    return args


def main(args):
    world_size = args.world_size
    print(world_size)
    # copy_json(args)
    mp.spawn(image_sr, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    
    args = parse_args()
    main(args)