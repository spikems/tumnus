#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Predictor for 美赞臣 Project.

Task Descriptions:
--------------------
1. 基础分类是指对抓取到的基础数据做话题分类，分为Sales、Campaign和Always-on三类；  
    Sales主要指代购、转置等信息，及网友网购评价；
    Campaign指品牌线上线下活动；
    Always-on指自然讨论数据。

2. 自然分类是指对基础分类中的Always-on进行再次分话题，分为使用感受、问询、购买、新闻、水军及其他六类话题。  
    新闻是指品牌及行业新闻；
    话题优先级为：购买>问询>使用感受（一条数据同时符合2个话题的话，按照这个优先级进行分类），但是如果摘要既包含问询又包含回答，则算为使用感受。

Data Format
---------------------
Input file are xls file, with such columns: 序号    监控对象    分类    标题    链接    日期    来源    点击    回复   摘要    作者    事件    评级    文章字数    分类
Predictor use 3 columns: 序号，标题，摘要
Output file has 4 columns: 序号, 预测分类, prob, content,
Output file are sorted by possiblility of classify error, thus mannual modification can be done by scan the outputfile from the begining.

Usage: 
---------------------
predict_mzc.py --data <jichu|ziran> --model <model name> --infile <input xlsx> --outfile <output filename>
    --data  ; define the task type, jichu by default
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
import xlwt
from tumnus.learn import Learner
from tumnus.postprocess import ProbFile

_categories = {'jichu':['alwayson','campaign','sales'],
                'ziran':['购买', '问询', '使用感受','新闻','水军','其他']}


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
    cmd = 'python %s/xlsdumper.py --dataset %s %s'%(basedir + '/preprocess', opts.data, opts.infile)
    outexpect = 'dump_%s_0.txt'%opts.data
    if not os.path.exists(outexpect):
        ret = os.system(cmd)

    if ret:
        logger.info('run dumper failed, quit')
        sys.exit(-1)

    # preprocess, -> dump_$data_0.cut -> $outfile.cut/.log/.grp
    cmd = 'python %s/cut.py -f %s'%(basedir + '/preprocess', outexpect)
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
    y_test, pred, pred_prob = predictor.predict(savename = opts.outfile)

    # post process
    probfile = ProbFile(opts.outfile + '.prob')
    probfile.load_data()
    ndata, idx = probfile.reorder_bydiff()

    # output the .xls file
    # predictor.pred_prob, .pred, .dataset.data, .dataset.ids
    schema = [u'序号', u'预测分类', u'概率', u'标题与摘要']
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('predict')
    
    #write schema
    for col in range(len(schema)):
        sheet.write(0, col, schema[col])
    
    # idx = [0,1,2]
    for row, indexval in enumerate(idx):
        # row, indexval -> point to the original dataset
        sheet.write(row + 1, 0, predictor.dataset.ids[indexval])
        sheet.write(row + 1, 1, predictor.dataset.target_names[pred[indexval]].decode('utf-8'))
        sheet.write(row + 1, 2, ['%.03f '%p for p in pred_prob[indexval]])
        sheet.write(row + 1, 3, predictor.dataset.data[indexval].decode('utf-8'))

    workbook.save(opts.outfile + '.xls')
