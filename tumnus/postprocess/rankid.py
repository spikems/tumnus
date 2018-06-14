#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
analysis the rank of feature in .coef output by classifier

input format:
    .coef
    label   \t  feature \t  weight

output files:

Usage:
    rankid.py <infiles> 




"""

import string
import sys,time
import os, os.path
import random
import logging
import numpy as np
from optparse import OptionParser

class Feature:
    def __init__(self):
        self.weight = 0.
        self.rank = 0
        self.setcnt = 0

class featureset:
    def __init__(self):
        self.data = {}
        self.coefcnt = 0

    def load_coef(self, fname):
        with open(fname,'r') as inf:
            self.coefcnt += 1

            currank = 0
            lastlabel = ''
            for line in inf:
                items = line.strip().split('\t')
                # logger.info('items = %s', items)

                #label change?
                if items[0] != lastlabel:
                    currank = 0
                    lastlabel = items[0]

                if items[0] not in self.data:
                    self.data[items[0]] = {}
                    
                fdict = self.data[items[0]]
                if items[1] in fdict:
                    # update this feature
                    fdict[items[1]].weight += float(items[2])
                    fdict[items[1]].rank += currank
                    fdict[items[1]].setcnt += 1
                else:
                    fdict[items[1]] = Feature()
                    fdict[items[1]].weight = float(items[2])
                    fdict[items[1]].rank = currank
                    fdict[items[1]].setcnt = 1
 
                currank += 1

    def save(self, filename):
        wf = open(filename, 'w')
        for label in self.data:
            for feature in self.data[label]:
                fs = self.data[label][feature]
                wf.write('%s\t%s\t%s\t%s\t%s\n'%(label, feature, fs.setcnt, fs.rank, fs.weight))

    def save_filter(self, fname):
        """
            Save the combined features into a filter file

        """
        oset = []
        for label in self.data:
            fset = []
            for feature in self.data[label]:
                fs = self.data[label][feature]
                if fs.setcnt == self.coefcnt:
                    fset.append(feature)
            oset.append(set(fset))
    
        #get only intersection
        out = set.intersection(*oset)

        wf = open(fname, 'w')
        for feature in out:
            wf.write('%s\n'%(feature))


if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    # cmd argument parser

    if len(sys.argv) < 2:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)

    fs = featureset()

    for infile in sys.argv[1:]:
        logger.info('loadint %s', infile)
        fs.load_coef(infile)

    fs.save('coef_summary.txt')
    fs.save_filter('coef_filter.txt')
