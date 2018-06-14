#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
extract content from the xlsx file

input:
    #jichu&ziran data
    序号    监控对象    分类    标题    链接    日期    来源    点击    回复   摘要    作者    事件    评级    文章字数    基础分类
    #xiaomi data
    postid title abstract 评级 hidden domain url 分类 posttime
    
output: 
    label id  title content

usage: xlsdump.py --dataset <dataset name> <xls file>

"""

from __future__ import print_function
from os.path import join, dirname, abspath
import os
import logging
import xlrd
import sys
import numpy as np
from optparse import OptionParser


def trim(line):
    """
        remove all \n\r\t in line
    """
    return line.replace('\n','').replace('\r','').replace('\t','')

class XLSDumper():
    """
    Dump data from definite columns in a xls/xlsx file.

    """
    _labels={
            'jichu':{'Always-on':0,'Campaign':1,'Sales':2},
            'ziran':{u'购买':0, u'问询':1, u'使用感受':2,u'新闻':3,u'水军':4,u'其他':5},
            'xiaomi':{'-2':0, '-1':0, '0':0, '1':1, '3':1, '4':1},
            #'xiaomi':{-2:'删除后恢复', -1:'已处理', 1:'已删除', 2:'正文噪音', 3:'歧义噪音', 4:'列表页噪音'},
            'xiaomi-sem':{u'负面':0, u'中立':1, u'正面':2}
        }

    _dumpcol = {'jichu':[14,3,9,0,5,2], # type, title, abstract, id, post timestamp, post page type
                'ziran':[14,3,9,0,5,2], # type, title, abstract, id, post timestamp, post page type
                'xiaomi':[4, 1, 2, 0, 8, 3], #hidden, title, abstract, id, post timestamp, semtype
                'xiaomi-sem':[3, 1, 2, 0, 8, 4] #semtype, title, abstract, id, post timestamp, hidden
            }

    def __init__(self):
        self.xl_workbook = None

    def loadbook(self, fname):
        self.xl_workbook = xlrd.open_workbook(fname)
    
        return self.xl_workbook
    
 
# 0,'',ziran,True
    def dumpsheet(self, sheet_id, outfname, dataset, use_idcol = True):
        """
        Dump configured by dataset refer to dumpcol and labels arrays.

        use_idcol = False ; return col postdate as id for training
                 = True  ; return col id for prediction

        """

        logger.info('dumping sheet %s'%self.xl_workbook.sheet_names()[sheet_id])
        if not outfname:
            outfname = 'dump_' + dataset + '_%d.txt'%sheet_id
        txtf = open(outfname,'w')
        xl_sheet = self.xl_workbook.sheet_by_index(sheet_id)

        # all values, iterating through rows and columns
        nullcnt = 0
        sheet = []
        num_cols = xl_sheet.ncols   # Number of columns
#	print ('num_cols:',num_cols,xl_sheet.nrows)
        for row_idx in range(0, xl_sheet.nrows):    # Iterate through rows
            arow = []
            for getid in self._dumpcol[dataset]:
                cell_obj = xl_sheet.cell_value(row_idx, getid)  # type选取需要的单元格
                if type(cell_obj) not in [unicode, str]:
                    arow.append(str(int(cell_obj)))		#int的作用是将浮点转为整型
                else:
                    arow.append(cell_obj)
            #check the record
            if arow[0] not in self._labels[dataset]:	#检查给的标签是否符合
                nullcnt += 1
                continue
            else:
                if dataset == 'xiaomi' and arow[0] == -2:
                    arow[0] = -1

            sheet.append(arow)
        # sort the data by timestamp
        errcnt = 0
        delcnt = 0
        print ("use_idcol:",use_idcol) 
        print ("length",len(sheet)) 
        if not use_idcol:	
            sheet = sorted(sheet, key = lambda arow: arow[4]) #按照时间排序，一般对嵌套的列表使用key
        for arow in sheet:
            label = self._labels[dataset][arow[0]] #将标签转成数字
            #patch here 
            if dataset == 'xiaomi-sem':
                if arow[5] > 0:  #delete
                    delcnt += 1
                    continue
            try:
                # return col.id or col.postdate
                if use_idcol:
                    txtf.write('%s %s %s %s\n'%(label, int(arow[3]) , trim(arow[1].lower().encode('utf-8')), trim(arow[2].lower().encode('utf-8')))) #label(int),id(int),title,abstract
                else:
                    txtf.write('%s %s%s %s %s\n'%(label, int(arow[3]), arow[4], trim(arow[1].lower().encode('utf-8')), trim(arow[2].lower().encode('utf-8')))) #label,id,time,title,abstract
            except:
                #known exceptions,  
                #title is number, encode() will raise type error
                errcnt += 1
                pass

        txtf.close()
        logger.info('nullcnt = %d, errorcnt = %d, delcnt = %d'%(nullcnt, errcnt, delcnt))

def load_option():
    op = OptionParser()
    op.add_option("--dataset",
                  action="store", type=str, dest="dataset", default='jichu',
                  help="Define the input file format.")
    op.add_option("--use_idcol",
                  action="store_true", 
                  help="define use postdate or id column in dataset, false by default means use postdate for training.")
    op.add_option("--h",
                  action="store_true", dest="print_help",
                  help="Show help info.")
 
    (opts, args) = op.parse_args()
#check file is or not ;if not print help ,else accep file
    if len(args) == 0 or opts.print_help:
        print(__doc__)
        op.print_help()
        print()
        sys.exit(0)
    else:
        opts.infile = args[0]
   # print ('opts:%s'%opts)
   # print ('args:%s'%args)
    return opts, args

if __name__=='__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    # cmd argument parser
    opts, args = load_option()      
#opts= opts:{'print_help': None, 'infile': '\xe5\xae\x9d\xe9\xa9\xac\xe5\x99\xaa\xe9\x9f\xb3\xe6\xa0\x87\xe8\xae\xb0.xlsx', 'use_idcol': None, 'dataset': 'xiaomi'}
#args:['\xe5\xae\x9d\xe9\xa9\xac\xe5\x99\xaa\xe9\x9f\xb3\xe6\xa0\x87\xe8\xae\xb0.xlsx']
    
    # Open the workbook
    workbook = XLSDumper()
    workbook.loadbook(opts.infile)
    workbook.dumpsheet(0, '', opts.dataset, opts.use_idcol)
