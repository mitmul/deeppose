#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer.training import extensions
from chainer.training.extensions import log_report as log_report_module

import logging
import sys


class LogPrinter(extensions.PrintReport):

    def __init__(self, entries, log_report=str('LogReport'), out=sys.stdout):
        self._entries = entries
        self._log_report = log_report
        self._log_len = 0

    def __call__(self, trainer):
        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)  # update the log report
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

        log = log_report.log
        log_len = self._log_len
        while len(log) > log_len:
            self._print(log[log_len])
            log_len += 1
        self._log_len = log_len

    def _print(self, observation):
        msg = ''
        for i, entry in enumerate(self._entries):
            if entry in observation:
                msg += '{}:{}, '.format(entry, observation[entry])
        logging.info(msg)
