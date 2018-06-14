#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
daysplit strate smpling to make the train dataset balance

input format:
    .cut    label date content

output files:
    xx-date.cut

Usage:
    daysplit.py <infile> <outfile>
"""

import string
import sys,time
import os, os.path
import random
import logging
import numpy as np
from optparse import OptionParser

def readfile(fname):
    with open(fname,'r') as inf:
        return [ line.strip() for line in inf]
    return []

def dosplit(train, out):
    '''
    '''
    traindata = readfile(train)
    totallines = len(traindata)
    logger.info('total lines = %d', totallines)

    traindata = sorted(traindata, key = lambda x:int(x.split()[1]) % 100000)
    sort_data = [x.split() for x in traindata]
    sort_data.append([0,-1,'']) #set sentinle

    startid = 0
    curdate = int(sort_data[0][1]) % 100000
    labels = {}  #label, cnt
    for id in xrange(len(sort_data)):
        # label status
        if curdate != int(sort_data[id][1]) % 100000:
            
            #split
            if out:
                outf = open(out + '-' + str(curdate) + '.cut', 'w')
                for line in xrange(startid, id):
                    outf.write('%s\n'%traindata[line])
                outf.close()

            #output label status
            logger.info('labels in day %s is: %s', curdate, labels)

            #go next
            startid = id
            curdate = int(sort_data[id][1]) % 100000
            labels = {}
        else:
            labels[sort_data[id][0]] = labels.get(sort_data[id][0],0) + 1

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

    inname = sys.argv[1]
    outname = sys.argv[2] if len(sys.argv) > 2 else ''

    dosplit(inname,outname)
