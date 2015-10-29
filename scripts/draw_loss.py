#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import matplotlib
if sys.platform in ['linux', 'linux2']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse


def draw_loss_curve(logfile, outfile):
    train_loss = []
    test_loss = []
    for line in open(logfile):
        line = line.strip()
        if not 'epoch:' in line:
            continue
        epoch = int(re.search(ur'epoch:([0-9]+)', line).groups()[0])
        if 'train' in line and 'inf' not in line:
            tr_l = float(re.search(ur'loss=([0-9\.]+)', line).groups()[0])
            train_loss.append([epoch, tr_l])
        if 'test' in line and 'inf' not in line:
            te_l = float(re.search(ur'loss=([0-9\.]+)', line).groups()[0])
            test_loss.append([epoch, te_l])

    train_loss = np.asarray(train_loss)[1:]
    test_loss = np.asarray(test_loss)[1:]

    if not len(train_loss) > 1:
        return

    plt.clf()
    fig, ax1 = plt.subplots()
    plt.plot(train_loss[:, 0], train_loss[:, 1], label='training loss')
    plt.plot(test_loss[:, 0], test_loss[:, 1], label='test loss')
    plt.xlim([2, len(train_loss)])
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.legend(loc='upper right')
    plt.savefig(outfile, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', '-f', type=str)
    parser.add_argument('--outfile', '-o', type=str)
    args = parser.parse_args()
    print(args)

    draw_loss_curve(args.logfile, args.outfile)
