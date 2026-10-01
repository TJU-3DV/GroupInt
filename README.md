# GroupInt: Group-Centric Crowd Reconstruction via Multi-Scale Interaction Analysis

The official code for paper "GroupInt: Group-Centric Crowd Reconstruction via Multi-Scale Interaction Analysis"<br>

[Buzhen Huang](https://www.buzhenhuang.com/), [Yanchi Jiang](), [Yuanbo Li](), [Yitao Xie](), [Kun Li](https://cic.tju.edu.cn/faculty/likun/)<br>

[\[Project\]](https://www.buzhenhuang.com/), [\[Paper\]]()<br><br>
![figure](/assets/Pipeline.png)

## Installation 
Create conda environment and install dependencies for GroupInt. <br>

```
conda create -n groupint python=3.8
conda activate groupint
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111 # install pytorch
pip install -r requirements.txt
```


## Getting Started

**Step1:** <br>
Download the official SMPL model from [SMPLify website](http://smplify.is.tuebingen.mpg.de/) and put it in `smpl/smpl/SMPL_NEUTRAL.pkl`. <br>

**Step2:** <br>
Download trained models from [Baidu Netdisk](https://pan.baidu.com/s/1qoA5MFF_w8uzYIp7RU8TsA?pwd=gnx5) and put them in `pretrained`. You may also need to download trained models for camerahmr from [CameraHMR website](https://camerahmr.is.tue.mpg.de/). <br>

**Step3:** <br>

Run demo.

```
python single_inference.py 
```

## Train

You can download the training data from [Baidu Netdisk](https://pan.baidu.com/s/1J-WPBxhT5GgzRfoamuFboA?pwd=j7cj) and place it in `data`. 

Training for grouping tasks.

```
python main_group.py --config cfg_files/config_group.yaml
```

Training for flow matching-based reconstruction.

```
python main_rec.py --config cfg_files/config_rec.yaml
```

## Dataset Annotations

We provide annotations of the group_id for LargeCrowd dataset. You may also need to download image files from their official websites.

[[Annotations](https://pan.baidu.com/s/1-P8ltTd6_i3Q46Oyumgifg?pwd=1kfb)]

![figure](/assets/annot.jpg)

## Visualization

![figure](/assets/vis.jpg)

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{huang2026groupint,
  title={GroupInt: Group-Centric Crowd Reconstruction via Multi-Scale Interaction Analysis},
  author={Huang, Buzhen and Jiang, Yanchi and Li, Yuanbo and Xie, Yitao and Li, Kun},
  year={2026}
}
```

## Acknowledgments
Some of the code are based on the following works. We gratefully appreciate the impact it has on our work.<br>
[CameraHMR](https://github.com/pixelite1201/CameraHMR)<br>
[GroupRec](https://github.com/boycehbz/GroupRec)<br>
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)<br>

