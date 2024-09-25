import os
import argparse

import argparse
import json

 
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Whole Pose Estimation')
parser.add_argument('--input', type=str, default='/image')
parser.add_argument('--output', type=str, default='/image')
args = parser.parse_args()

def shell_str(dirpath, outputpath_vis, outputpath_json):
    sh_str = 'python demo/inferencer_demo.py ' + dirpath + '\
    --pose2d configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py \
    --pose2d-weights Demo_model/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth --det-model yolox_l_8x8_300e_coco  --pred-out-dir '+outputpath_json +' --vis-out-dir '+outputpath_vis
    return sh_str

with open('mmpose.sh', 'w') as f:
    
    if os.path.isdir(args.input):
        # print('Input is a directory')
        f.write(shell_str(args.input, args.output+'_vis', args.output+'_json'))
        print(shell_str(args.input, args.output+'_vis', args.output+'_json'))
        f.write('\n')
        subfilelist = os.listdir(args.input)
        for subfile in subfilelist:
            subfilepath = os.path.join(args.input, subfile)
            suboutputpath = os.path.join(args.output, subfile)
            if os.path.isdir(subfilepath):
                print('Subfile is a directory')
                f.write(shell_str(subfilepath, suboutputpath+'_vis', suboutputpath+'_json'))
                f.write('\n')

            
os.system('sh mmpose.sh')


# python whole_pose_single.py --input /data/muxiangyu/TryonDatasets/datasets/FashionDataset_frames_crop --output /data/muxiangyu/TryonDatasets/datasets/mmPose

# 将文件夹下的所有json文件变成没有背景的pose图像
def blockPoseImg(jsonpath, outputpath):
    with open(jsonpath, 'r') as file:
        data = json.load(file)
    
    wholePoseList = data[0]['keypoints']
    poseList = []
    for i in range(17):
        poseList.append(wholePoseList[i])
    feetList = []
    for i in range(17,23):
        feetList.append(wholePoseList[i])
    faceList = []
    for i in range(23,91):
        faceList.append(wholePoseList[i])
    handList = []
    for i in range(91,133):
        handList.append(wholePoseList[i])

    width, height = 384,512

    black_image = np.zeros((height, width, 3), np.uint8)

    for pose in poseList:
        black_image = cv2.circle(black_image, (int(pose[0]), int(pose[1])), 4, (0, 255, 0), -1)

    for feet in feetList:
        black_image = cv2.circle(black_image, (int(feet[0]), int(feet[1])), 3, (0, 255, 255), -1)

    for face in faceList:
        black_image = cv2.circle(black_image, (int(face[0]), int(face[1])), 1, (255, 255, 0), -1)


    for hand in handList:
        black_image = cv2.circle(black_image, (int(hand[0]), int(hand[1])), 2, (255, 0, 255), -1)

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    cv2.imwrite(os.path.join(outputpath, os.path.basename(jsonpath).split('.')[0]+'.jpg'), black_image) 



folderList = os.listdir(args.output)
for folder in folderList:
    if '_json' in folder:
        jsonList = os.listdir(os.path.join(args.output, folder))
        for json_1 in jsonList:
            if '.json' in json_1:
                jsonpath = os.path.join(args.output, folder, json_1)
                outputpath = os.path.join(args.output,folder.split('_json')[0]+'_img')
                blockPoseImg(jsonpath,outputpath)
