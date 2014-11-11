#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import numpy as np
if sys.platform.startswith('linux'):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_loss_curve(fname):
    loss_iter = []
    loss = []
    test_iter = []
    test = []
    for line in open(fname):
        if 'Iteration' in line and 'loss' in line:
            txt = re.search(ur'Iteration\s([0-9]+)', line)
            loss_iter.append(int(txt.groups()[0]))
            txt = re.search(ur'loss\s=\s([0-9\.]+)\n', line)
            loss.append(float(txt.groups()[0]))
        if 'Testing net' in line:
            txt = re.search(ur'Iteration\s([0-9]+)', line)
            test_iter.append(int(txt.groups()[0]))
        if 'Test net output' in line and 'loss':
            txt = re.search(ur'=\s*([0-9\.]+)\s*loss\)', line)
            if txt:
                test.append(float(txt.groups()[0]))

    plt.clf()
    plt.plot(loss_iter, loss)
    plt.plot(test_iter, test)
    plt.savefig('loss_curve.png')


if __name__ == '__main__':
    save_loss_curve('nohup.out')
