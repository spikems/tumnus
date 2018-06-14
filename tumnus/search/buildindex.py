#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Build index by Indri

Input:
    .cut
    label   id  content

buildindex.py --infile <input> --outfile <output>
"""

import logging
import numpy as np
import sys, os
from optparse import OptionParser
from time import time

def convert2trectext(cutfile, trecfile, recordid = True):
    """
        Convert from .cut into .trec
        TrecText Format:
        <DOC>
         <DOCNO>document_id</DOCNO>
         <TEXT>
         Index this document text.
         </TEXT>
        </DOC>

        recordid; True to use recordid, otherwise use id in .cut file
    """
    cutf = open(cutfile, 'r')
    trecf = open(trecfile, 'w')
    recid = 0
    for line in cutf:
        line = line.strip()
    
        pos1 = line.find(' ')
        #put the id into data, extract into X_id in the future
        pos2 = line.find(' ', pos1+1)
        #pos2 = pos1
        if recordid == True:
            id = str(recid)
            recid += 1
        else:
            id = line[pos1+1:pos2]
    
        words = line[pos2+1:]

        #output
        trecf.write("<DOC>\n<DOCNO>%s</DOCNO>\n<TEXT>\n%s\n</TEXT>\n</DOC>\n"%(id, words))
    
    cutf.close()
    trecf.close()


def buildindex(trecfile, indexpath):
    """
    """
    #create the parameter file
    parameters = """
        <parameters>
        <corpus>
        <path>INPATH</path>
        <class>trectext</class>
        </corpus>
        <index>OUTPATH</index>
        <memory>256M</memory>
        </parameters>
    """
    param_file = 'build.param'
    param = parameters.replace('INPATH', trecfile)
    param = param.replace('OUTPATH', indexpath)
    with open(param_file ,'w') as pf:
        pf.write('%s'%param)

    # run indri
    ret = 0
    cmd = 'IndriBuildIndex %s'% param_file
    outexpect = indexpath
    if not os.path.exists(outexpect):
        ret = os.system(cmd)
    else:
        logger.info('index: %s exists, skip build index', outexpect)

    if ret:
        logger.info('run build index failed, quit')
        sys.exit(-1)

# parse commandline arguments
def load_option():
    op = OptionParser()
    op.add_option("--infile",
                  action="store", type=str, 
                  help="input slsx file to predict")
    op.add_option("--outfile",
                  action="store", type=str,
                  help="output xlsx file with the prediction filled.")
    op.add_option("--recordid",
                  action="store_true", 
                  help="use record id or the id in content.")
    op.add_option("--h",
                  action="store_true", dest="print_help",
                  help="Show help info.")
    
    (opts, args) = op.parse_args()
   
    if opts.print_help:
        print(__doc__)
        op.print_help()
        print()
        sys.exit(0)

    return opts

if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    basedir = os.path.dirname(sys.argv[0])
    basedir = os.path.dirname(basedir)
    logger.info('basedir detect as %s', basedir)

    opts = load_option()
    # convert into TREC_TEXT
    cutfile = opts.infile
    trecfile = opts.infile.replace('.cut', '.trec')
    convert2trectext(cutfile, trecfile, recordid = True)

    # build index
    buildindex(trecfile, opts.outfile)
