# DeepPose

NOTE: This is not official implementation. Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659).

# Requirements

- Python 3.5.1+
  - [Chainer](https://chainer.org/) 4.2.0
  - [CuPy](https://cupy.chainer.org/) 4.2.0
  - [ChainerCV](http://chainercv.readthedocs.io/en/stable/index.html) 0.10.0
  - [NumPy](http://numpy.org/) 1.14.5
  - [opencv-python](https://pypi.org/project/opencv-python/) 3.4.1.15

# Download Datasets

```
bash download.sh
```

- [FLIC-full dataset](https://bensapp.github.io/flic-dataset.html)
- [LSP Extended dataset](http://sam.johnson.io/research/lspet.html)

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
