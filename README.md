deeppose
========

# Requirements

- [Chainer](https://github.com/pfnet/chainer) (Neural network framework)
    - I'm using master version on GitHub, so
        - `$ git clone https://github.com/pfnet/chainer.git`
        - `$ cd chainer; python setup.py install`
    - and then, if you want to train networks with GPU,
        - `$ pip install chainer-cuda-deps`
- progressbar2
    - `$ pip install progressbar2`
    - NOTE: it's not `progressbar`!

# Data preparation

    $ bash scripts/downloader.sh
    $ python scripts/flic_dataset.py
    # python scripts/lsp_dataset.py

This script downloads FLIC-full dataset (http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC) and perform cropping regions of human and save poses as numpy files into FLIC-full directory.

## (MPII Dataset)[http://human-pose.mpi-inf.mpg.de/#download]

- # of training images: 18079, # of test images: 6908
    - test images don't have any annotations
    - so we split trining imges into training/test joint set
    - each joint set has
- # of training joint set: 17928, # of test joint set: 1991

# Start training

## For FLIC Dataset

```
    nohup python scripts/train.py \
    --model models/AlexNet.py \
    --gpu 0 \
    --epoch 1000 \
    --batchsize 128 \
    --prefix AlexNet_LCN_AdaGrad_lr-0.0005 \
    --snapshot 10 \
    --datadir data/FLIC-full \
    --channel 3 \
    --flip True \
    --size 220 \
    --crop_pad_inf 1.5 \
    --crop_pad_sup 2.0 \
    --shift 5 \
    --lcn True \
    --joint_num 7 \
    > AlexNet_LCN_AdaGrad_lr-0.0005.log 2>&1 &
```

## For LSP Dataset

```
    nohup python scripts/train.py \
    --model models/AlexNet_lsp.py \
    --gpu 0 \
    --epoch 1000 \
    --batchsize 128 \
    --prefix AlexNet_lsp \
    --snapshot 10 \
    --datadir data/lspet_dataset \
    --channel 3 \
    --flip True \
    --size 100 \
    --crop_pad_inf 1.5 \
    --crop_pad_sup 2.0 \
    --shift 5 \
    --lcn True \
    --joint_num 14 \
    --fname_index 0 \
    --joint_index 1 \
    > LSP_AlexNet_lsp_LCN_AdaGrad_lr-0.0005.log 2>&1 &
```

See the help messages with `--help` option for details.

# Visualize Filters of 1st conv layer

- Go to result dir of a model
-  `python ../../scripts/draw_filters.py`

# Visualize Prediction

## Example

```
    python scripts/predict_flic.py \
    --model results/AlexNet_2015-06-22_13-00-34_143494563481/AlexNet.py \
    --param results/AlexNet_2015-06-22_13-00-34_143494563481/AlexNetBN_LCN_AdaGrad_lr-0.0005_epoch_400.chainermodel \
    --gpu 8 \
    --datadir data/FLIC-full
```

run the above command from deeppose's root dir.
