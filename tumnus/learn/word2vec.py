#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
build word2vec model from text

input format:
    .cut
    label id content

output files:
    save word2vec model file

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
from classifier import load_file 

class W2V:
    def __init__(self):
        self.data = []
        self.w2v = None

    def load_data(self, fname):
        data = load_file(fname)
        self.data.extend(data.data)

    def train(self, size=600, min_count=5, sg=1, iter=20):
        _data = [x.split() for x in self.data]
        #Initialize model and build vocab
        #self.w2v = Word2Vec(size=600, min_count=5, sg=1, iter=20)
        self.w2v = Word2Vec(size=size, min_count=min_count, sg=sg, iter=iter)
        self.w2v.build_vocab(_data)
        self.w2v.train(_data)

    def save(self, fname):
        self.w2v.save_word2vec_format(fname)

def load_option():
    op = OptionParser()
    op.add_option("--output",
                  action="store", type=str, default="word2vec.model", dest="output_fname",
                  help="Output model file name.")
 
    (opts, args) = op.parse_args()
    return opts, args

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
    opts, args = load_option()

    w2v = W2V()

    for infile in args:
        logger.info('loadint %s', infile)
        w2v.load_data(infile)

    w2v.train()
    w2v.save(opts.output_fname)


