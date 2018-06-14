#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Predictor for 小米 Project.

Task Descriptions:
--------------------
小米品牌文章识别 

Data Format
---------------------
Input file are xls file, with such columns: postid	title	摘要	评级	hidden	domain	url	分类	posttime
Predictor use 3 columns: 序号，标题，摘要
Output file has 4 columns: 序号, 预测分类, prob, content,
Output file are sorted by possiblility of classify error, thus mannual modification can be done by scan the outputfile from the begining.

Usage: 
---------------------
predict_xiaomi.py --data <jichu|ziran> --model <model name> --infile <input xlsx> --outfile <output filename>
    --data  ; define the task type
    --model ; the model file name, trained and provide by mining group
    --infile ; input xlsx file to predict
    --outfile ; output xlsx file with the prediction filled
"""

from __future__ import print_function

import logging
import numpy as np
import sys, os
from optparse import OptionParser
from time import time
import pickle
import xlsxwriter
from tumnus.learn import Learner
from tumnus.postprocess import ProbFile

_categories = {
                'jichu':['alwayson','campaign','sales'],
                'ziran':['购买', '问询', '使用感受','新闻','水军','其他'],
                'xiaomi':['zaoyin', 'brand']
              }


# parse commandline arguments
def load_option():
    op = OptionParser()
    op.add_option("--data",
                  action="store", type=str, default="jichu",
                  help="define the task type, jichu by default.")
    op.add_option("--model",
                  action="store", type=str, 
                  help="the model file name, trained and provide by mining group.")
    op.add_option("--infile",
                  action="store", type=str, 
                  help="input slsx file to predict")
    op.add_option("--outfile",
                  action="store", type=str,
                  help="output xlsx file with the prediction filled.")
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
    ret = 0
    # dump xls, opts.infile -> dump_$data_0.txt
    cmd = 'python %s/xlsdumper.py --dataset %s --use_idcol %s'%(basedir + '/preprocess', opts.data, opts.infile)
    outexpect = 'dump_%s_0.txt'%opts.data
    if not os.path.exists(outexpect):
        ret = os.system(cmd)

    if ret:
        logger.info('run dumper failed, quit')
        sys.exit(-1)

    # preprocess, -> dump_$data_0.cut -> $outfile.cut/.log/.grp
    cmd = 'python %s/cut.py -f %s  --template %s/preprocess/template/template  --brands %s/../data/xiaomi.brands'%(basedir + '/preprocess', outexpect, basedir, basedir)
    outexpect = 'dump_%s_0.cut'%opts.data
    if not os.path.exists(outexpect):
        ret = os.system(cmd)

    if ret:
        logger.info('run cut failed, quit')
        sys.exit(-1)

    cmd = 'python %s/dedup.py -k 3 -i %s -o %s'%(basedir + '/preprocess', outexpect, opts.outfile + '.cut')
    outexpect = '%s.cut'%opts.outfile
    if not os.path.exists(outexpect):
        ret = os.system(cmd)

    if ret:
        logger.info('run dedup failed, quit')
        sys.exit(-1)

    # predict
    predictor = Learner(train = False)
    predictor.load_model(opts.model)
    predictor.load_dataset(opts.outfile + '.cut', recordid = False, category_names = _categories[opts.data])
    predictor.transform()
    predictor.feature_select()
    y_test, pred, pred_prob = predictor.predict(savename = opts.outfile)

    # post process
    probfile = ProbFile(opts.outfile + '.prob')
    probfile.load_data()
    ndata, idx = probfile.reorder_bydiff()

    # output the .xls file
    # predictor.pred_prob, .pred, .dataset.data, .dataset.ids
    schema = [u'序号', u'预测分类', u'概率', u'标题与摘要']
    workbook = xlsxwriter.Workbook(opts.outfile + '.xlsx')
    sheet = workbook.add_worksheet('predict')
    
    #write schema
    for col in range(len(schema)):
        sheet.write(0, col, schema[col])
    
    #idx = [0,1,2]
    for row, indexval in enumerate(idx):
        # row, indexval -> point to the original dataset
        sheet.write(row + 1, 0, predictor.dataset.ids[indexval])
        sheet.write(row + 1, 1, predictor.dataset.target_names[pred[indexval]].decode('utf-8'))
        sheet.write(row + 1, 2, ' '.join(['%.03f'%p for p in pred_prob[indexval]]))
        sheet.write(row + 1, 3, predictor.dataset.data[indexval].decode('utf-8'))

    workbook.close()

