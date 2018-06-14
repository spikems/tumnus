#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
undersampling to make the train dataset balance

input format:
    sent, id, content

Usage:
    undersample.py <input> <output> 

"""

import string
import sys,time
import os, os.path
import random
import logging
from optparse import OptionParser

def readfile(fname):
    with open(fname,'r') as inf:
        return [ line.strip() for line in inf]
    return []

def undersample(train, out, seed = 12345):
    '''
    '''
    traindata = readfile(train)
    outf = open('large_' + out, 'w')
    outf2 = open('small_' + out, 'w')

    classes={}
    for rec in traindata:
        if rec=='': break
        cat = rec[:rec.find(' ')]
        classes[cat] = classes[cat] + 1 if cat in classes else 1

    print('cat statistics as:%s'%classes)
    random.seed(seed)

    #under sampling by the largest one, here class 0
    ratio = (classes['1'] + classes['-1'])*1.0/classes['0']
    print('undersampling for class(-1,1) ratio as:%0.3f'%ratio)

    for rec in traindata:
        if rec=='': break

        cat = rec[:rec.find(' ')]
        if cat=='1' or cat == '-1':
            outf.write('%s\n'%rec)
        else:
            if random.random() <= ratio:
                outf.write('%s\n'%rec)

    # under sample by the smallest class, here class -1
    ratio1 = (classes['-1'])*1.0/classes['1']
    ratio2 = (classes['-1'])*1.0/classes['0']
    print('undersampling for class(-1) ratio as:%0.3f, %0.3f'%(ratio1, ratio2))

    for rec in traindata:
        if rec=='': break

        cat = rec[:rec.find(' ')]
        if cat == '-1':
            outf2.write('%s\n'%rec)
        elif cat == '1':
            if random.random() <= ratio1:
                outf2.write('%s\n'%rec)
        else:
            if random.random() <= ratio2:
                outf2.write('%s\n'%rec)
 

if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    # cmd argument parser

    if len(sys.argv) < 3:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)

    inname = sys.argv[1]
    outname = sys.argv[2]
    undersample(inname,outname, int(sys.argv[3]) if len(sys.argv)>3 else 12345)
