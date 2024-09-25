# 目的
将一个素人模特或者模特假人的面部等非皮肤区域替换成模板模特（reference image），并根据一个姿势序列来生成一段视频。
* 输入：一个素人模特，一个模板模特，以及一个动作序列（5s内
* 输出：将素人模特替换成模板模特的并且符合动作序列的视频（5s内）

# 数据预处理
## mmpose
用来获取模特的姿势特征，包含身体、面部、手和脚。
其中在json文件中，0~16是身体pose、17~22是脚的pose、23~90是面部的pose以及91~133是手部pose。
### 如何生成图像mmpose
使用环境openmmlab

通过运行文件/data/muxiangyu/pythonPrograms/Mannequin2Real/Mannequin2Video/mmpose/whole_pose.py可以生成图像的mmpose，包含图片、json文件以及可视化的pose_image。

```python
python whole_pose_single.py --input /data/muxiangyu/TryonDatasets/datasets/FashionDataset_frames_crop --output /data/muxiangyu/TryonDatasets/datasets/mmPose
```
其中，input 可以是图片路径、一级文件夹或两级文件夹，output是输出路径，路径中包含img、json以及vis三个文件。

## Graphonomy
用于对模特图片进行人体分割。
目前已有的标签

0: 背景；
2：头发；
7：上衣；
6：连衣裙；
10：脖子；
13：面部；
14：右胳膊（包含手）；
15：左胳膊（包含手）；
16：右脚；
17：左脚；
18：左鞋子；
19：右鞋子；

### 如何使用Graphonomy分割算法
使用环境 human_parsing_graphonomy
如果是对单张图像进行分割，使用/data/muxiangyu/pythonPrograms/Mannequin2Real/Mannequin2Video/Graphonomy/exp/inference/inference.py文件，对文件夹中的图片进行分割，使用/data/muxiangyu/pythonPrograms/Mannequin2Real/Mannequin2Video/Graphonomy/exp/inference/inference_folder.py

对于单张图像：
```python
python exp/inference/inference.py  \
--loadmodel /path_to_inference_model \
--img_path ./img/messi.jpg \
--output_path ./img/ \
--output_name /output_file_name
```
对于文件夹中的多张图像：
如果是一级文件夹：
将下面这段代码进行解除注释
```python
    # for img_name in os.listdir(opts.imgfolder_path):
    #     img_path = os.path.join(opts.imgfolder_path, img_name)
    #     if not os.path.exists(opts.output_path):
    #         os.makedirs(opts.output_path)
    #     inference(net=net, img_path=img_path,output_path=opts.output_path , output_name=img_name, use_gpu=use_gpu)
```
并运行/data/muxiangyu/pythonPrograms/Mannequin2Real/Mannequin2Video/Graphonomy/inference_folder.sh

如果是二级文件夹
则将下面这段代码接触注释
```pyhton
    for imgfolder in os.listdir(opts.imgfolder_path):
        imgfolder_path = os.path.join(opts.imgfolder_path, imgfolder)
        output_folder = os.path.join(opts.output_path, imgfolder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for img_name in os.listdir(imgfolder_path):
            img_path = os.path.join(imgfolder_path, img_name)
            
            inference(net=net, img_path=img_path,output_path=output_folder, output_name=img_name, use_gpu=use_gpu)
```
并运行/data/muxiangyu/pythonPrograms/Mannequin2Real/Mannequin2Video/Graphonomy/inference_folder.sh

此外，文件/data/muxiangyu/pythonPrograms/Mannequin2Real/Mannequin2Video/Graphonomy/parsing_analysis.ipynb是用来查看分割图像标签的Demo。