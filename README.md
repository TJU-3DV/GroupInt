# GroupInt
# GroupInt: Group-Centric Crowd Reconstruction via Multi-Scale Interaction Analysis

The official code for paper "GroupInt: Group-Centric Crowd Reconstruction via Multi-Scale Interaction Analysis"<br>

![figure](/assets/Pipeline.png)

## Installation 
**Step1:** <br>

Create conda environment and install dependencies for CameraHMR, refer to [GitHub - pixelite1201/CameraHMR · GitHub](https://github.com/pixelite1201/CameraHMR). <br>

**Step2:** <br>

Create conda environment and install dependencies for GroupInt. <br>

```
conda create -n groupint python=3.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111 # install pytorch
pip install -r requirements.txt
```


## Getting Started

**Step1:** <br>
Download the official SMPL model from [SMPLify website](http://smplify.is.tuebingen.mpg.de/) and put it in `smpl/smpl/SMPL_NEUTRAL.pkl`. <br>

**Step2:** <br>
Download trained models from [Baidu Netdisk(pwd=gnx5)](https://pan.baidu.com/s/1qoA5MFF_w8uzYIp7RU8TsA) and put them in `pretrained`. You may also need to download trained models for camerahmr from their official websites. <br>

**Step3:** <br>

Run demo for camerahmr annots.

```
conda activate camerahmr
python Annot_CameraHMR.py 
```

Run demo for GroupInt.

```
conda activate groupint
python single_inference.py 
```

## Train

You can download the training data from [Baidu Netdisk(pwd=j7cj)](https://pan.baidu.com/s/1J-WPBxhT5GgzRfoamuFboA) and place it in the `data` directory. 

Train for group.

```
python main_group.py --config cfg_files/config_group.yaml
```

Train for flow_matching.

```
python main_rec.py --config cfg_files/config_rec.yaml
```

## Dataset Annotations

We provide annotations of the group_id for LargeCrowd dataset. You may also need to download image files from their official websites.

[[Annotations]()]

## Acknowledgments
Some of the code are based on the following works. We gratefully appreciate the impact it has on our work.<br>
[CameraHMR](https://github.com/pixelite1201/CameraHMR)<br>
[GroupRec](https://github.com/boycehbz/GroupRec)<br>
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)<br>
