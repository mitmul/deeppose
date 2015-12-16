DeepPose
========

NOTE: This is not official implementation. Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659).

# Requirements

- [Chainer 1.5+](https://github.com/pfnet/chainer) (Neural network framework)
- numpy 1.9+
- scipy 0.16+
- scikit-learn 0.15+
- OpenCV 2.4+

# Data preparation

```
bash shells/download.sh
python scripts/flic_dataset.py
python scripts/lsp_dataset.py
```

This script downloads FLIC-full dataset (http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC) and perform cropping regions of human and save poses as numpy files into FLIC-full directory.

## MPII Dataset

- [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#download)
- # of training images: 18079, # of test images: 6908
    - test images don't have any annotations
    - so we split trining imges into training/test joint set
    - each joint set has
- # of training joint set: 17928, # of test joint set: 1991

# Start training

## For FLIC Dataset

Just run:

```
nohup python scripts/train.py > AlexNet_flic.log 2>&1 < /dev/null &
```

For speed:

```
CHAINER_TYPE_CHECK=0 nohup python scripts/train.py > AlexNet_flic.log 2>&1 < /dev/null &
```

It is same as:

```
nohup python scripts/train.py \
--model models/AlexNet_flic.py \
--gpu 0 \
--epoch 1000 \
--batchsize 32 \
--prefix AlexNet_LCN_AdaGrad_lr-0.0005 \
--snapshot 10 \
--datadir data/FLIC-full \
--channel 3 \
--flip 1 \
--size 220 \
--crop_pad_inf 1.5 \
--crop_pad_sup 2.0 \
--shift 5 \
--lcn 1 \
--joint_num 7 \
> AlexNet_LCN_AdaGrad_lr-0.0005.log 2>&1 &
```

`--flip 1` means it performs LR flip augmentation, and `--flip 0` does nothing. `--lcn 1` means local(should be said "global"?) contrast normalization will be applied.

See the help messages with `--help` option for details.

### GPU memory requirement

- batchsize: 128 -> about 2870 MiB
- batchsize: 64 -> about 1890 MiB
- batchsize: 32 (default) -> 1374 MiB

# Visualize Filters of 1st conv layer

- Go to result dir of a model
-  `python ../../scripts/draw_filters.py`

# Visualize Prediction

## Example

### Prediction and visualize them and calc mean errors

```
python scripts/evaluate_flic.py \
--model results/AlexNet_2015/AlexNet.py \
--param results/AlexNet_2015/AlexNet_epoch_400.chainermodel \
--datadir data/FLIC-full
--gpu 0 \
--batchsize 128 \
--mode test
```

### Tile some randomly selected result images

```
python scripts/evaluate_flic.py \
--model results/AlexNet_2015/AlexNet_flic.py \
--param results/AlexNet_2015/AlexNet_epoch_450.chainermodel \
--mode tile \
--n_imgs 25
```

### Create animated GIF to intuitively compare predictions and labels

```
cd results/AlexNet_2015
bash ../../scripts/create_anime.sh test_450_tiled_pred.jpg test_450_tiled_label.jpg test_450.gif
```
