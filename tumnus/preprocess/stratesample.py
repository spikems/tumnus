#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
strate smpling to make the train dataset balance

input format:
    .txt

output files:
    train-[r-]outfile   ; r- means result after permutation
    test-[r-]outfile

Usage:
    sample.py <infile> <outfile> <number> <permutation>
        number: integer 
            <0      ;-1 means split by time, get last day for test
            [0,1)   ;0.1 means train-test split ratio
            0       ;0 means run only permutation
            >1      ;means record cnt to save
 
        permutation: bool
            True/False  ; do permutation or not
            
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

def dosample(train, out, number, dopermute = True):
    '''
        number: integer 
                float   ;0.1 means train-test split ratio
                float   ;0 means run only permutation
    '''
    traindata = readfile(train)

    totallines = len(traindata)
    logger.info('total lines = %d', totallines)

    #init permutation 
    if dopermute:
        logger.info('init permuation')
        permute = np.random.permutation(totallines)
        #set prefix to out file name
        out = 'r-' + out
    else:
        logger.info('skip permuation')
        permute = np.arange(totallines)
    
    if number < 0:
        # split by time
        # this is for special file format
        # label date content
        #traindata = [x.split() for x in traindata]
        #sort_data = sorted(traindata, key = lambda x:x[1])
        traindata = sorted(traindata, key = lambda x:x.split()[1])
        sort_data = [x.split() for x in traindata]


        datecnt = 0
        curdate = sort_data[-1][1]
        for id in xrange(-1, -len(sort_data), -1):
            if curdate != sort_data[id][1]:
                datecnt += 1
                if datecnt >= -number:
                    #finish
                    break
        trainnum = id + 1

    elif number > 0 and number <1 :
        # split by ratio
        trainnum =int(totallines*number)
    elif number == 0 or number > totallines:
        # permutation only
        trainnum = totallines
    elif number < totallines:
        # split by absolute number
        trainnum = int(number)

    logger.info('train number = %d, test number = %d', trainnum, len(traindata) - trainnum)
    #train
    outf = open('train-'+out, 'w')
    _permute = permute[:trainnum]
    for line in _permute:
        outf.write('%s\n'%traindata[line])
    outf.close()
    #test   
    outf = open('test-'+out, 'w')
    _permute = permute[trainnum:]
    for line in _permute:
        outf.write('%s\n'%traindata[line])
    outf.close()


if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    # cmd argument parser

    if len(sys.argv) < 4:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)

    inname = sys.argv[1]
    outname = sys.argv[2]
    num = float(sys.argv[3])
    dopermute = True

    if len(sys.argv)==5 and (sys.argv[4]=='false' or sys.argv[4] == 'False'):
        dopermute = False

    dosample(inname,outname, num, dopermute)
