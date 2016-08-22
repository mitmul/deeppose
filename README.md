# DeepPose

NOTE: This is not official implementation. Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659).

# Requirements

- Python 3.5.1+
  - [Chainer 1.13.0+](https://github.com/pfnet/chainer)
  - numpy 1.9+
  - scikit-image 0.11.3+
  - OpenCV 3.1.0+

I strongly recommend to use Anaconda environment. This repo may be able to be used in Python 2.7 environment, but I haven't tested.

## Installation of dependencies

```
pip install chainer
pip install numpy
pip install scikit-image
# for python3
conda install -c https://conda.binstar.org/menpo opencv3
# for python2
conda install opencv
```

# Dataset preparation

```
bash datasets/download.sh
python datasets/flic_dataset.py
python datasets/lsp_dataset.py
python datasets/mpii_dataset.py
```

- [FLIC-full dataset](http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC)
- [LSP Extended dataset](http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip)
- **MPII dataset**
    - [Annotation](http://datasets.d2.mpi-inf.mpg.de/leonid14cvpr/mpii_human_pose_v1_u12_1.tar.gz)
    - [Images](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz)

## MPII Dataset

- [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#download)
- training images: 18079, test images: 6908
  - test images don't have any annotations
  - so we split trining imges into training/test joint set
  - each joint set has
- training joint set: 17928, test joint set: 1991

# Start training

Starting with the prepared shells is the easiest way. If you want to run `train.py` with your own settings, please check the options first by `python scripts/train.py --help` and modify one of the following shells to customize training settings.

## For FLIC Dataset

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

# Prediction

Will add some tools soon
