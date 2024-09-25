from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image, ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier
from numpy import asarray

import random
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from shutil import copyfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import os

label_to_index={
    "background": 0,
    "hat": 1,
    "hair": 2,
    "glove": 3,
    "sunglasses": 4,
    "upperclothes": 5,
    "dress": 6,
    "coat": 7,
    "socks": 8,
    "pants": 9,
    "jumpsuits": 10,
    "scarf": 11,
    "skirt": 12,
    "face": 13,
    "leftArm": 14,
    "rightArm": 15,
    "leftLeg": 16,
    "rightLeg": 17,
    "leftShoe": 18,
    "rightShoe": 19
}

def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    return torchvision.transforms.Compose(transform_list)


class try_on_dataset_VITONHD(data.Dataset):
    def __init__(self, state, order: str = 'paired', pairs_file: str = None, **args):
        self.state = state
        self.args = args
        self.kernel = np.ones((1, 1), np.uint8)
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.boundingbox_as_inpainting_mask_rate = 0.4

        self.order = order
        self.pairs_file = pairs_file






        self.source_dir = []
        self.segment_map_dir = []
        self.ref_dir = []
        self.pose_dir = []
        

        



        if self.state == "train":
            dataroot = os.path.join(args["dataset_dir"], "train_frame_num.txt")
        elif self.state == "test":
            dataroot = os.path.join(args["dataset_dir"], "test_frame_num.txt")
        elif self.state == "val":
            dataroot = os.path.join(args["dataset_dir"], "val_frame_num.txt")
        

        # if self.order == 'paired':
        #     filename = os.path.join(args["dataset_dir"], f"VITONHD_{self.state}_paired.txt")
        # else:
        #     filename = os.path.join(args["dataset_dir"], f"VITONHD_{self.state}_unpaired.txt")
        


        if pairs_file:
            filename = pairs_file

        print('pathFile:',filename)

        with open(filename) as f:
            framesss = []
            segment = []
            ref_image = []
            pose = []
            items = f.readlines()
            for item in items:
                model, allframes = item.strip().split()
                numbers = list(range(allframes))
                selected_frames = random.sample(numbers, 180)
                selected_frames.sort()
                for frame in selected_frames:
                    frame = str(frame).zfill(3)
                    framesss.append(frame+'.png')
                    segment.append(frame+'.png_gray.png')
                    pose.append(frame+'.jpg')
                self.source_dir.append([model,framesss])
                self.segment_map_dir.append([model,segment])
                self.pose_dir.append([model,pose])
                random_number = random.randint(0, 100)
                str_random = str(random_number).zfill(3)

                self.ref_dir.append([str_random+'.png',str_random+'.png_gray.png'])
    
        self.length = len(self.source_dir)

    
    def getMasks(self, parse_array):
        human_mask = (parse_array==2).astype(np.uint8) + (parse_array==13).astype(np.uint8) 

        heaad_mask_with_arms = (parse_array==2).astype(np.uint8) + (parse_array==13).astype(np.uint8) + \
                    (parse_array==14).astype(np.uint8) + (parse_array==15).astype(np.uint8) + \
                        (parse_array==16).astype(np.uint8) + (parse_array==17).astype(np.uint8) + (parse_array==10).astype(np.uint8)

        heaad_mask_with_arms_bg = (parse_array==2).astype(np.uint8) + (parse_array==13).astype(np.uint8) + \
            (parse_array==14).astype(np.uint8) + (parse_array==15).astype(np.uint8) + \
                (parse_array==16).astype(np.uint8) + (parse_array==17).astype(np.uint8) + (parse_array==10).astype(np.uint8) + \
                (parse_array==0).astype(np.uint8)
        
        heaad_mask_with_arms_bg = 1 - heaad_mask_with_arms_bg

        epsilon_randomness = random.uniform(0.001, 0.005)
        randomness_range = random.choice([ 8, 9, 10])
        kernel_size = random.choice([ 8, 10, 13, 15])

        # predict mask GT, inpainting mask to be dilated 
        heaad_mask_with_arms = 1 - heaad_mask_with_arms.astype(np.float32)
        heaad_mask_with_arms[heaad_mask_with_arms < 0.5] = 0
        heaad_mask_with_arms[heaad_mask_with_arms >= 0.5] = 1
        heaad_mask_with_arms_resized = cv2.resize(heaad_mask_with_arms, (256,512), interpolation=cv2.INTER_NEAREST)

    
        contours, _ = cv2.findContours(((1 - heaad_mask_with_arms_resized) * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        total_area = np.ones((512, 256))
        for contour in contours:
            # max_contour = max(contours, key = cv2.contourArea)
            epsilon = epsilon_randomness * cv2.arcLength(contour, closed=True)  
            approx_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
            randomness = np.random.randint(-randomness_range, randomness_range, approx_contour.shape)
            approx_contour = approx_contour + randomness

            zero_mask = np.zeros((512, 384))
            contours = [approx_contour]

            cv2.drawContours(zero_mask, contours, -1, (255), thickness=cv2.FILLED)

            kernel = np.ones((kernel_size,kernel_size),np.uint8)
            head_mask_with_arms_inpainting = cv2.morphologyEx(zero_mask, cv2.MORPH_CLOSE, kernel)
            head_mask_with_arms_inpainting = head_mask_with_arms_inpainting.astype(np.float32) / 255.0
            head_mask_with_arms_inpainting[head_mask_with_arms_inpainting < 0.5] = 0
            head_mask_with_arms_inpainting[head_mask_with_arms_inpainting >= 0.5] = 1
            head_mask_with_arms_inpainting = heaad_mask_with_arms_resized * (1 - head_mask_with_arms_inpainting)
            total_area = total_area * head_mask_with_arms_inpainting

        total_area = cv2.erode(total_area, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=3)[None]

        total_area = total_area[0] + heaad_mask_with_arms_bg
        total_area[total_area < 0.5] = 0
        total_area[total_area >= 0.5] = 1

        head_mask_tensor = torch.from_numpy(human_mask)     # 头部mask
        total_area_tensor = torch.from_numpy(total_area)    # 全身mask

        return head_mask_tensor, total_area_tensor




    def __getitem__(self, index):
        source_path_list = []
        for im in self.source_dir[index][1]:
            source_path_list.append(os.path.join(self.args["dataset_dir"],'FashionDataset_frames_crop', self.source_dir[index][0], im))
        segment_map_path_list = []
        for im in self.segment_map_dir[index][1]:
            segment_map_path_list.append(os.path.join(self.args["dataset_dir"],'humanParsing', self.segment_map_dir[index][0], im))
        pose_path_list = []
        for im in self.pose_dir[index][1]:
            pose_path_list.append(os.path.join(self.args["dataset_dir"],'mmPose', self.pose_dir[index][0]+'_img', im))

        ref_image_name = self.ref_dir[index][0][0]
        ref_image_parse_name = self.ref_dir[index][0][1]
        ran_int = random.randint(0, 50)
        ref_path = os.path.join(self.args["dataset_dir"],'FashionDataset_frames_crop', self.source_dir[index][0], ref_image_name)
        ref_parse_path = os.path.join(self.args["dataset_dir"],'humanParsing', self.segment_map_dir[index][0], ref_image_parse_name)
        
        source_img_list = []     # 原始图像列表
        source_img_tensor_list = []
        for source_path in source_path_list:
            source_img = Image.open(source_path).convert("RGB")
            source_img = source_img.resize((384,512), Image.BILINEAR)
            source_img_tensor_list.append(get_tensor()(source_img))
            source_img_list.append(source_img)
        
        segment_map_list = []    # 语义分割图像列表
        segment_map_tensor_list = []
        for segment_map_path in segment_map_path_list:
            segment_map = Image.open(segment_map_path)
            segment_map = segment_map.resize((384,512), Image.NEAREST)
            segment_map_list.append(np.array(segment_map))
            segment_map_tensor_list.append(torch.from_numpy(segment_map))

        head_madk_list = []
        total_area_list = []
        inpainting_mask_list = []
        for parse_array in segment_map_list:
            head_mask_tensor, total_area_tensor = self.getMasks(parse_array)
            head_madk_list.append(head_mask_tensor)
            total_area_list.append(total_area_tensor)
            inpainting_mask_list.append(torch.from_numpy(parse_array)*total_area_tensor)
        
        pose_img_list = []       # 姿态图像列表
        pose_img_tensor_list = []
        for pose_path in pose_path_list:
            pose_img = Image.open(pose_path).convert("RGB")
            pose_img = pose_img.resize((384,512), Image.BILINEAR)
            pose_img_tensor_list.append(get_tensor()(pose_img))
            pose_img_list.append(pose_img)

        ref_img = Image.open(ref_path).convert("RGB")
        ref_parse = Image.open(ref_parse_path)
        

        ref_img = ref_img.resize((384,512), Image.BILINEAR)
        ref_img_tensor = get_tensor()(ref_img)
        ref_parse = ref_parse.resize((384,512), Image.NEAREST)
        ref_head_mask_tensor, ref_total_area_tensor = self.getMasks(np.array(ref_parse))

        ref_img_tensor = ref_img_tensor * ref_head_mask_tensor

        clothes_tensor_list = []
        for source_img, total_area in zip(source_img_tensor_list, total_area_list):
            clothes = source_img * total_area
            clothes_tensor_list.append(clothes)



#----------------------------------------------------------------------------------------------------------------

#         segment_map_path = self.segment_map_dir[index]
#         ref_path = self.ref_dir[index]
#         pose_path = self.pose_dir[index]
#         densepose_path = self.densepose_dir[index]

#         source_img = Image.open(source_path).convert("RGB")
#         source_img = source_img.resize((384,512), Image.BILINEAR)
#         image_tensor = get_tensor()(source_img)


#         segment_map = Image.open(segment_map_path)
#         segment_map = segment_map.resize((384,512), Image.NEAREST)
#         parse_array = np.array(segment_map)


#         garment_mask = (parse_array == 5).astype(np.float32) + \
#                         (parse_array == 7).astype(np.float32)

#         garment_mask_with_arms = (parse_array == 5).astype(np.float32) + \
#                         (parse_array == 7).astype(np.float32) + \
#                     (parse_array == 14).astype(np.float32) + \
#                     (parse_array == 15).astype(np.float32)


#         epsilon_randomness = random.uniform(0.001, 0.005)
#         randomness_range = random.choice([ 80, 90, 100])
#         kernel_size = random.choice([ 80, 100, 130, 150])


#         # predict mask GT, inpainting mask to be dilated 
#         garment_mask = 1 - garment_mask.astype(np.float32)
#         garment_mask[garment_mask < 0.5] = 0
#         garment_mask[garment_mask >= 0.5] = 1
#         garment_mask_resized = cv2.resize(garment_mask, (384,512), interpolation=cv2.INTER_NEAREST)

    
#         contours, _ = cv2.findContours(((1 - garment_mask_resized) * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         if len(contours) != 0:
#             max_contour = max(contours, key = cv2.contourArea)
#             epsilon = epsilon_randomness * cv2.arcLength(max_contour, closed=True)  
#             approx_contour = cv2.approxPolyDP(max_contour, epsilon, closed=True)
#             randomness = np.random.randint(-randomness_range, randomness_range, approx_contour.shape)
#             approx_contour = approx_contour + randomness

#             zero_mask = np.zeros((512, 384))
#             contours = [approx_contour]

#             cv2.drawContours(zero_mask, contours, -1, (255), thickness=cv2.FILLED)

#             kernel = np.ones((kernel_size,kernel_size),np.uint8)
#             garment_mask_inpainting = cv2.morphologyEx(zero_mask, cv2.MORPH_CLOSE, kernel)
#             garment_mask_inpainting = garment_mask_inpainting.astype(np.float32) / 255.0
#             garment_mask_inpainting[garment_mask_inpainting < 0.5] = 0
#             garment_mask_inpainting[garment_mask_inpainting >= 0.5] = 1
#             garment_mask_inpainting = garment_mask_resized * (1 - garment_mask_inpainting)
#         else:
#             garment_mask_inpainting = np.zeros((512, 384))

#         garment_mask_GT = cv2.erode(garment_mask_resized, self.kernel_dilate, iterations=3)[None]
#         garment_mask_inpainting = cv2.erode(garment_mask_inpainting, self.kernel_dilate, iterations=5)[None]

#         garment_mask_GT_tensor = torch.from_numpy(garment_mask_GT)
#         garment_mask_inpainting_tensor = torch.from_numpy(garment_mask_inpainting)

# #----------------------------------------------------------------------------------------
#         # generate inpainting boundingbox, inpainting mask to be dilated, 
#         garment_mask_with_arms = 1 - garment_mask_with_arms.astype(np.float32)
#         garment_mask_with_arms[garment_mask_with_arms < 0.5] = 0
#         garment_mask_with_arms[garment_mask_with_arms >= 0.5] = 1
#         garment_mask_with_arms_resized = cv2.resize(garment_mask_with_arms, (384,512), interpolation=cv2.INTER_NEAREST)

#         garment_mask_with_arms_boundingbox = cv2.erode(garment_mask_with_arms_resized, self.kernel_dilate, iterations=5)[None]


#         # boundingbox
#         _, y, x = np.where(garment_mask_with_arms_boundingbox == 0)
#         if x.size > 0 and y.size > 0:
#             x_min, x_max = np.min(x), np.max(x)
#             y_min, y_max = np.min(y), np.max(y)
#             boundingbox = np.ones_like(garment_mask_with_arms_boundingbox)
#             boundingbox[:, y_min:y_max, x_min:x_max] = 0
#         else:
#             boundingbox = np.zeros_like(garment_mask_with_arms_boundingbox)

#         boundingbox_tensor = torch.from_numpy(boundingbox)


#         # limit in the boundingbox
#         garment_mask_inpainting_tensor = torch.where((garment_mask_inpainting_tensor==0) & (boundingbox_tensor==0), torch.zeros_like(garment_mask_inpainting_tensor), torch.ones_like(garment_mask_inpainting_tensor))


#         # select inpainting mask
#         if self.state != "test":
#             mask_or_boundingbox = random.random()
#             if mask_or_boundingbox < 1 - self.boundingbox_as_inpainting_mask_rate:
#                 inpainting_mask_tensor = garment_mask_inpainting_tensor
#             else:
#                 inpainting_mask_tensor = boundingbox_tensor
#         else:
#             inpainting_mask_tensor = boundingbox_tensor




#         ref_img_combine = Image.open(ref_path).convert("RGB")
#         ref_img_combine = ref_img_combine.resize((384,512), Image.BILINEAR)
#         ref_img_combine_tensor = get_tensor()(ref_img_combine)

#         pose_img = Image.open(pose_path).convert("RGB")
#         pose_img = pose_img.resize((384,512), Image.BILINEAR)
#         poseimage_tensor = get_tensor()(pose_img)

#         densepose_img = Image.open(densepose_path).convert("RGB")
#         densepose_img = densepose_img.resize((384,512), Image.BILINEAR)
#         denseposeimage_tensor = get_tensor()(densepose_img)

#         inpaint_image = image_tensor * inpainting_mask_tensor

        # ref_tensors = [ref_img_combine_tensor]

        # # 768 * 512 GT_image
        # GT_image_combined = torch.cat((image_tensor, ref_img_combine_tensor), dim=2)

        # # 768 * 512 GT_mask
        # GT_mask_combined = torch.cat((garment_mask_GT_tensor, torch.ones((1, 512, 384), dtype=torch.float32)), dim=2)

        # # 768 * 512 inpaint_image
        # inpaint_image_combined = torch.cat((inpaint_image, ref_img_combine_tensor), dim=2)

        # # 768 * 512 inpainting mask
        # inpainting_mask_combined = torch.cat((inpainting_mask_tensor, torch.ones((1, 512, 384), dtype=torch.float32)), dim=2)

        # # 768 * 512 posemap
        # pose_combined = torch.cat((poseimage_tensor, ref_img_combine_tensor), dim=2)

        # # 768 * 512 densepose
        # densepose_combined = torch.cat((denseposeimage_tensor, ref_img_combine_tensor), dim=2)
        
        # # image_name
        # image_name = os.path.split(source_path)[-1]



        return {
            "Video_name": self.source_dir[index][0],
            "GT_video" : source_img_tensor_list,
            "GT_video_mask" : segment_map_tensor_list,
            "ref_img" : ref_img_tensor,
            "clothes_video": clothes_tensor_list,
            "pose_video": pose_img_tensor_list,
            "inpaint_mask_video": inpainting_mask_list
        }

    def __len__(self):
        return self.length