# DeepPose

**NOTE: This is NOT the official implementation.**

This is an unofficial implementation of [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659).

# Requirements

- Python 3.5.1+
  - [Chainer](https://chainer.org/)>=4.2.0
  - [CuPy](https://cupy.chainer.org/)>=4.2.0
  - [ChainerCV](http://chainercv.readthedocs.io/en/stable/index.html)>=0.10.0
  - [NumPy](http://numpy.org/)>=1.14.5
  - [opencv-python](https://pypi.org/project/opencv-python/)==3.4.5.20

# Download Datasets

```
bash datasets/download.sh
```

- [FLIC-full dataset](https://bensapp.github.io/flic-dataset.html)
- [LSP Extended dataset](http://sam.johnson.io/research/lspet.html)

# How to start training

Starting with the prepared shells is the easiest way. If you want to run `train.py` with your own settings, please check the options first by `python scripts/train.py --help` and modify one of the following shells to customize training settings.

## For FLIC Dataset

```
python scripts/train.py -o results/$(date "+%Y-%m-%d_%H-%M-%S")
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
