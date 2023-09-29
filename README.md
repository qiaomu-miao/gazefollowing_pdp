# Gazefollowing_PDP

This is the code implementation for the paper "Patch-level Gaze Distribution Prediction for Gaze Following" published in WACV 2023 ([paper](https://openaccess.thecvf.com/content/WACV2023/papers/Miao_Patch-Level_Gaze_Distribution_Prediction_for_Gaze_Following_WACV_2023_paper.pdf)).

## Environments:
python=3.7.13 <br>
pytorch=1.10.0 <br>
cudatoolkit=11.3.1 <br>

## Training

Download the images and annotations for [GazeFollow](https://www.dropbox.com/s/3ejt9pm57ht2ed4/gazefollow_extended.zip?dl=0) and [VideoAttentionTarget](https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=0).

Download the depth maps for GazeFollow and VideoAttentionTarget datasets [here](https://drive.google.com/drive/folders/1vA8Qks5hyjK-_ivxI8ocWOuyLxLg31Fq?usp=sharing).

Download the initial weights for training on GazeFollow and VideoAttentionTarget datasets [here](https://drive.google.com/drive/folders/14Oyh0aXFXbTJ9BjS919XGmvd54TBN4ig?usp=sharing).

Modify config_pdp.yaml with your datasets and depth map directories accordingly.

### Training on GazeFollow
For training on gazefollow dataset, run:
```
python train_gazefollow_patch.py --init_weights {initial_weights_for_spatial_training.pt}
```

For training the model without using depth images:
```
python train_gazefollow_patch.py --init_weights {initial_weights_for_spatial_training.pt} --not_use_depth --lambda_ 40
```

### Training on VideoAttentionTarget
For VideoAttentionTarget, we split the dataset into 5-frame sequences in training and test sets and stored the splits [here](https://drive.google.com/drive/folders/1Rt_Ejm918Et5qtqARvVfDUurkggF7tlw?usp=sharing).
For training on VideoAttenionTarget dataset, run:
```
python train_videoatt_patch.py --init_weights {initial_weights_for_temporal_training.pt}
```

for training the model without using depth images:
```
python train_videoatt_patch.py --init_weights {initial_weights_for_temporal_training_nodepth.pt} --not_use_depth 
```

## Pretrained Models

[Here](https://drive.google.com/drive/folders/1A9nqTvdGXB7F-rwU9g2bJItrSfT1EGob?usp=sharing) we provide the pretrained model weights on GazeFollow and VideoAttentionTarget datasets.

## Citation
If you find our code useful, please consider citing our paper:
```
@inproceedings{miao2023patch,
  title={Patch-level Gaze Distribution Prediction for Gaze Following},
  author={Miao, Qiaomu and Hoai, Minh and Samaras, Dimitris},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={880--889},
  year={2023}
}
```