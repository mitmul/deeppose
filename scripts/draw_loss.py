#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import re
import sys

if sys.platform in ['linux', 'linux2']:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib
    import matplotlib.pyplot as plt


def draw_loss_curve(logfile, outfile):
    try:
        train_loss = []
        test_loss = []
        for line in open(logfile):
            line = line.strip()
            if 'epoch:' not in line:
                continue
            epoch = int(re.search('epoch:\s*([0-9]+)', line).groups()[0])
            if 'training' in line and 'inf' not in line:
                print(line)
                tr_l = float(re.search('loss:\s*([0-9\.]+)', line).groups()[0])
                train_loss.append([epoch, tr_l])
            if 'test' in line and 'inf' not in line:
                te_l = float(re.search('loss:\s*([0-9\.]+)', line).groups()[0])
                test_loss.append([epoch, te_l])

        train_loss = np.asarray(train_loss)[1:]
        test_loss = np.asarray(test_loss)[1:]

        if not len(train_loss) > 1:
            return

        print(train_loss)
        print(test_loss)

        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(train_loss[:, 0], train_loss[:, 1],
                 label='training loss', c='r')
        ax1.set_xlim([2, len(train_loss)])
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('training loss')
        ax1.legend(bbox_to_anchor=(0.25, -0.1), loc=9)

        if len(test_loss) > 1:
            ax2 = ax1.twinx()
            ax2.plot(test_loss[:, 0], test_loss[:, 1], label='test loss',
                     c='b')
            ax2.set_ylabel('test loss')

            ax2.legend(bbox_to_anchor=(0.75, -0.1), loc=9)
            # ax2.set_ylim(ax1.get_ylim())

        plt.savefig(outfile, bbox_inches='tight')

    except Exception as e:
        print(str(type(e)), e, line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default='log.txt')
    parser.add_argument('--outfile', type=str, default='log.png')
    args = parser.parse_args()
    print(args)

    draw_loss_curve(args.logfile, args.outfile)
