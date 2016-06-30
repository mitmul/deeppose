# DeepPose

NOTE: This is not official implementation. Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659).

# Requirements

- Python 2.7.11+

  - [Chainer 1.5+](https://github.com/pfnet/chainer) (Neural network framework)
  - numpy 1.9+
  - scipy 0.16+
  - scikit-learn 0.15+
  - OpenCV 2.4+

## Installation of dependencies

```
pip install chainer
pip install numpy
pip install scipy
pip install scikit-learn
# for python3
conda install -c https://conda.binstar.org/menpo opencv3
# for python2
conda install opencv
```

# Data preparation

```
bash shells/download.sh
python scripts/flic_dataset.py
python scripts/lsp_dataset.py
python scripts/mpii_dataset.py
```

This script downloads FLIC-full dataset (<http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC>) and perform cropping regions of human and save poses as numpy files into FLIC-full directory. Same processes are performed for LSP, MPII datasets.

## MPII Dataset

- [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#download)
- training images: 18079, test images: 6908

  - test images don't have any annotations
  - so we split trining imges into training/test joint set
  - each joint set has

- training joint set: 17928, test joint set: 1991

# Start training

## For FLIC Dataset

Starting with the prepared shells is the easiest way. If you want to run `train.py` with your own settings, please check the options first by `python scripts/train.py --help` and modify one of the following shells to customize training settings.

```
bash shells/train_flic.sh
```

## For LSP Dataset

```
bash shells/train_lsp.sh
```

## For MPII Dataset

```
bash shells/train_mpii.sh
```

### GPU memory requirement

- AlexNet

  - batchsize: 128 -> about 2870 MiB
  - batchsize: 64 -> about 1890 MiB
  - batchsize: 32 (default) -> 1374 MiB

- ResNet50

  - batchsize: 32 -> 6877 MiB

# Visualize Filters of 1st conv layer

- Go to result dir of a model
- `python ../../scripts/draw_filters.py`

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
