#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
build word2vec model from text

input format:
    .cut
    label id content

Usage:
    word2vec.py <infile> <modelfile>

"""

import string
import sys,time
import os, os.path
import random
import logging
import numpy as np
from optparse import OptionParser
from gensim.models.word2vec import Word2Vec
from sklearn.feature_selection import SelectKBest, chi2
from classifier import load_file, load_option, _categories 

class FeatureMon:
    def __init__(self):
        self.data = []

    def load_data(self, fname,opts):
        data = load_file(fname, opts)
        # features, label, hit
        # self.data.extend((data.data.split(), data.target))
        self.data = zip(data.data, data.target)
        logger.info('loading end at %s', len(self.data))

    def check(self, feature):
        """
            Check a feature on the loaded dataset
            output the coverage matrix
        """
        total = len(self.data)
        hit = np.zeros((len(self.data),2))
        for idx in range(len(self.data)):
            items, target = self.data[idx]
            hit[idx][0] = int(target)
            if feature in items:
                hit[idx][1] = 1

        #output statistics
        coverage = np.sum(hit[:,1]) *1.0/ total
        logger.info('Total Coverage: %.3f', coverage)
        maxtarget = int(np.max(hit[:,0]))
        for label in range(1, maxtarget+1):
            labeltotal = np.sum(hit[:,0] == label)
            hittotal = np.sum((hit[:,0] == label) & (hit[:,1] > 0))
            coverage = hittotal *1.0/ labeltotal

            logger.info('Label %s Coverage: %.3f', label, coverage)


if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 2:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)

    # cmd argument parser
    opts = load_option()

    fm = FeatureMon()

    fm.load_data(sys.argv[1], opts)

    fm.check(sys.argv[2])


