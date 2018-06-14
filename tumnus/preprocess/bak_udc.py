#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Cutword for sentiment dataset

input format:
    id sentitype text

Usage:
    cut -t [pos] -c [hidden]  -f <file name> --template <template path> --brands <brand path or single brand name>
        -t  ; save pos tag
        -c  ; output <hidden, text cut>, otherwise output <senti,  text cut> by default
        -f  ; input file name
        --template ; template file path
        --brands ; brand list , if brand is a brand name, we will just deal the single word.but a path we will read the path text;
"""

import string
import sys,time
import os, os.path
import logging
import re
from jieba.norm import norm_cut, norm_seg
from optparse import OptionParser
import chardet
reload(sys)
sys.setdefaultencoding('utf8')


special_dict={
"contryProfile" : ["人口","农业","气温","政局","平方公里","宪法"],
"investCase" : ["工业园","项目","合作","投产"],
"investEnv": ["投资环境","物价水平","宏观经济","银行和保险行业","融资服务"],
"investPolicy" : ["政策","对外贸易","外资","税率","税负","税收"], 
"infoRepost" : ["报道","新闻网"]}



def wordInDict(sentence,lists,label):
 
    special_word = [] 
    num = 0
    for i in lists :
        if i in sentence:
            special_word.append('%s_%s'%(label,num))
            num += 1

    return ' '.join(special_word)



def special(sentence):
 
     wordLabel ="%s %s %s %s %s" \
     %(wordInDict(sentence,special_dict["contryProfile"],"contryProfile"),
     wordInDict(sentence,special_dict["investCase"],"investCase"), 
     wordInDict(sentence,special_dict["investEnv"],"investEnv"), 
     wordInDict(sentence,special_dict["investPolicy"],"investPolicy"), 
     wordInDict(sentence,special_dict["infoRepost"],"infoRepost")) 
     
     return wordLabel 

def cut_input(input):
    '''
    cut a input string, return utf-8 string
    '''

    result = norm_seg(input)
    wordsList = []
    for w in result:
        if w.word.strip() == '' or w.flag.strip() == '':
            continue
        wordsList.append(w.word)

    words = " ".join(wordsList)

    return words.encode('utf-8')
     


def cut_file(fileName,posFlag):
    '''
    cut from file and output to filename.cut
    '''
    dir, name = os.path.splitext(fileName)
    middle = '_pos' if posFlag else ''
    writer = open( dir + middle + '.cut', 'w')
    reader = open(fileName, 'rb')
 
    reccnt = 0
    #
    # parse the records
    # id label hidden text
    #
    for line in reader:
        #add a id
        try:
            line = line.strip().split('\t') 
            label = int(re.sub(r'\s','',line[0]))
            stype = int(re.sub(r'\s','',line[1]))
            content = re.sub(r'\s','',line[2])

            result = cut_input(content)
            result = '%s %s'%(result,special(content))

            writer.write('%d %d '%(label, stype) + result  + '\n')
        except:
            continue
    reccnt += 1

    reader.close()
    writer.close()


if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    #logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    # cmd argument parser
    usage = 'cut -t [pos] -f <file path> --template <template path> --brands <brand path or single brand name>'
    parser = OptionParser(usage)
    parser.add_option("-f", dest="pathName")
    parser.add_option("-t", dest="type", action="store_true")

    opt, args = parser.parse_args()
    if opt.pathName is None:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)

    posFlag = False
    if not (opt.type is None):
        posFlag = True
    

    arg_name = opt.pathName 
    cut_file(arg_name,posFlag)
