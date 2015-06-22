deeppose
========

# Requirements

- [Chainer](https://github.com/pfnet/chainer) (Neural network framework)
    - Just do `$ pip install chainer`
    - and then, if you want to train networks with GPU, `$ pip install chainer-cuda-deps`

# Data preparation

    $ bash scripts/downloader.sh
    $ python scripts/flic_dataset.py

This script downloads FLIC-full dataset (http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC) and perform cropping regions of human and save poses as numpy files into FLIC-full directory.

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

See the help messages with `--help` option for details.

- 1 epoch takes 2.75 min on K80.
