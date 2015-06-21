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

    $ python scripts/train.py --model models/AlexNet.py --gpu 0

See the help messages with `--help` option for details.

- 1 epoch takes 2.75 min on K80.
