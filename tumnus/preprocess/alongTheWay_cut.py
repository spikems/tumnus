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
"contryProfile" : ["人口","农业","气温","政局","平方公里","宪法","国土","小镇","特色","贫穷","面貌","交通运输","位于","语言","宗教","总统","节日","国庆节","面积","气候","政治","政局","立法","政府","内阁","州","市","法院","检察院","工业","农业","旅游","教育","新闻出版"],
"investCase" : ["工业园","中国经验","助","由中国企业","样本","东方工业园","援助","与中国","中资参与","入股","成功","里程碑","突破",
"新的篇章","深入落地","重要意义","将为","融资支持","资助","项目投资","确定对","实施","开工","新成果","投产","收购","股权","重要进展",
"打破","提升了","以来","例","引领者","承建","第一次","进展","完成","升级","建成后","历史上","投资规模","签署","建设项目","合同"],
"investEnv": ["投资吸引力","外资","近几年的经济","优势","竞争力","环境","经济自由度","经济增长率","GDP","收入","支出","膨胀率","赤字",
"失业率","外汇储备","债务","外债","信用","特色产业","投资环境","物价水平","宏观经济","银行和保险行业","融资服务","增长","出口","规划",
"国内市场","销售总额","生活支出","基础设施","公路","铁路","总里程","客运公司","线路","时速","火车头","车厢","机场","航班","货运量",
"航空公司","","船","港口","旅客","出港","进港","邮政","信件","包裹","固话","移动电话","宽带","电力","基础设施","贸易","商品","外国援助","金融环境","当地货币","外汇管理","银行","保险公司","融资服务","信用卡","证券","商务成本","水价","电价","气价","劳动力","工薪",
"土地价格","成本"],
"investPolicy" : ["法规","贸易","法律","规定","在进口","检验","检疫","海关","规章","制度","细则","规定","税收","体系","税","优惠",
"措施","法","劳动合同","工作时间","政策","对外贸易","外资","税率","税赋","税收"], 
"infoRepost" : ["报道","新闻","网","参加","记者","新华社","讲座","论坛","议题","与会","倡议","研讨会","会议","媒体","座谈会",
"专家学者","报告称","统计","数据显示","报道","编辑","采访","致辞","出席","年会","人民日报","主持人","焦点","报","消息"]}



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
            content = re.sub(r'\s','',line[2])[:300]

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
