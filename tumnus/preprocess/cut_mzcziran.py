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
from jieba.norm import norm_cut, norm_seg
from optparse import OptionParser
import chardet



def load_template_feature(result_with_flag, posList, dBrand, lFeatureTemplate):
    '''
        add template feature to vector
    '''
    lWords_Flag = result_with_flag.split(" ")
    lAddFeature = []
    lAfterQueue = posList
    lBeforeQueue = {'all':[]}
    
    # key word dict 
    dKeyWords = {}
    iPos = 0
    for sWord_Flag in lWords_Flag:
        if len(sWord_Flag.split('_')) < 2:
            continue
        sWord = sWord_Flag.split('_')[0]
        sFlag = sWord_Flag.split('_')[-1]

        if sFlag == '':
            continue

        #delete first word from lAfterQueue
        lAfterQueue[sFlag] = lAfterQueue[sFlag][1:]
        lAfterQueue['all'] = lAfterQueue['all'][1:]
        if not sFlag in lBeforeQueue:
            lBeforeQueue[sFlag] = []

        if dBrand.has_key(sWord):

            dKeyWords[sWord] = 1

            for sItem in lFeatureTemplate:
                lFlagNum = sItem.split("_")
                sPosFlag = lFlagNum[0]
                if not sPosFlag in lAfterQueue:
                    lAfterQueue[sPosFlag] = []
                    
                if not sPosFlag in lBeforeQueue:
                    lBeforeQueue[sPosFlag] = []

                if len(lFlagNum) == 2:
                    iNum = int(lFlagNum[1])
                    if iNum > 0:
                        sFeature = 'local_brand_%s' % "_".join(lAfterQueue[sPosFlag][:iNum])
                    else:
                        sFeature = 'local_%s_brand' % "_".join(lBeforeQueue[sPosFlag][iNum:])
                else:
                    iMax, iSmall = (int(lFlagNum[1]), int(lFlagNum[2])) if int(lFlagNum[1]) > int(lFlagNum[2]) else (int(lFlagNum[2]), int(lFlagNum[1]))
                    if iMax * iSmall > 0:
                        if iMax > 0:
                            sFeature = 'local_brand_%s' % "_".join(lAfterQueue[sPosFlag][iSmall - 1:iMax])
                        else:
                            if iMax == -1:
                                sFeature = 'local_%s_brand' % "_".join(lBeforeQueue[sPosFlag][iSmall : ])
                            else:
                                sFeature = 'local_%s_brand' % "_".join(lBeforeQueue[sPosFlag][iSmall : iMax + 1])
                    else:
                        sFeature = 'local_%s_brand_%s' % ("_".join(lBeforeQueue[sPosFlag][iSmall:]), "_".join(lAfterQueue[sPosFlag][:iMax]))
                lAddFeature.append(sFeature)
        
        #add word to lBeforeQueue
        lBeforeQueue[sFlag].append(sWord)
        lBeforeQueue['all'].append(sWord)
        iPos = iPos + 1
     
    #add keyword count feature
    if len(dKeyWords) > 2:
        lAddFeature.append("local_many_keyword")

    return  (" ").join(lAddFeature)         

def ziran_class(self,result):
    '''
    for shopping string
    '''
    dic1 = {'渠道': 1, '价格': 1, '供应': 1, '有效期': 1, '保质期': 1, '哪里买': 1, '促销': 1, '降价': 1, '价格': 1, '降价': 1, '便宜': 1, '贵': 1,'实惠': 1, '划算': 1, '低价': 1, '性价比': 1, '超市': 1, '药房': 1, '药店': 1, '卖场': 1, '代购': 1, '海淘': 1, '直邮': 1, '发货': 1,'晒单': 1, '万宁': 1, '有卖': 1, '哪卖': 1, '哪买': 1, '哪里买': 1, '那里卖': 1, '哪有': 1, '多少钱': 1, '哪个地方买': 1, '平台': 1}
    dic2 = {'上火': 1, '便秘': 1, '拉肚子': 1, '绿便': 1, '长肉': 1, '觉得': 1, '感觉': 1, '溶': 1, '甜': 1, '苦': 1, '腥': 1, '腹泻': 1,'口感': 1, '味道': 1, '新鲜': 1, '建议': 1, '适应': 1, '心得': 1, '绿屎': 1, '便秘': 1, '便便': 1, '很硬': 1, '拉不出': 1, '拉稀': 1,'拉西': 1, '闹肚子': 1, '拉肚子': 1, '稀便': 1, '腹泻': 1, '吐奶': 1, '胀气': 1, '上火': 1, '烦躁': 1, '红肿': 1, '消化': 1, '肠道': 1,'肠胃': 1, '过敏': 1, '湿疹': 1, '麻疹': 1, '腥': 1, '甜': 1, '口感': 1, '味': 1}
    for i in result.split(''):
	if i in dic1 :
	    return True
	


def cut_input(input):
    '''
    cut a input string, return utf-8 string
    '''

    result = norm_seg(input)
    wordsList = []
    wordsPosList = []
    posDict = {'all':[]}
    for w in result:
        if w.word.strip() == '' or w.flag.strip() == '':
            continue
        wordsList.append(w.word)
        wordsPosList.append(w.word + '_' + w.flag)
        if not w.flag in posDict:
            posDict[w.flag] = []
        posDict[w.flag].append(w.word.encode('utf-8'))
        posDict['all'].append(w.word.encode('utf-8'))

    words_posFlag = " ".join(wordsPosList)
    words = " ".join(wordsList)

    return words.encode('utf-8'), words_posFlag.encode('utf-8'), posDict
     
def cut_file(fileName, posFlag, lFeatureTemplate, dBrand):
    '''
    cut from file and output to filename.cut
    '''
    dir, name = os.path.splitext(fileName)
    middle = '_pos' if posFlag else ''
    writer = open( dir + middle + '.cut', 'w')
    reader = open(fileName, 'rb')
    shopping_writer = open(dir + middle+ '.shopping','w')
    else_writer =open('ziran_else.cut','w')

    reccnt = 0
    #
    # parse the records
    # id label hidden text
    #
    for line in reader:
        #add a id 
        if line[0] == ' ':
            pos0 = line.find(' ')+1
        else:
            pos0 = 0
        pos1 = line.find(' ',pos0)+1
        pos2 = line.find(' ',pos1)+1

        if pos2 > 0:
            label = int(float(line[pos0:pos1-1]))
            stype = int(float(line[pos1:pos2-1]))
            content = line[pos2:-1]

            result, result_with_flag, posList = cut_input(content)
            template_feature = ''
            if len(lFeatureTemplate) and len(dBrand):
                template_feature = load_template_feature(result_with_flag, posList, dBrand, lFeatureTemplate)
            if posFlag:
                result = result_with_flag
	    result = zr_feature_add(result)					
    #        if ret == 2:
     #           shopping_writer.write('%d %d '%(label, stype) + result + template_feature + '\n')
#	    elif ret == 1
 #               else_writer.write('%d %d '%(label, stype) + result + template_feature + '\n')
#	    else:		
            writer.write('%d %d '%(label, stype) + result + template_feature + '\n')

        reccnt += 1

    reader.close()
    writer.close()
    shopping_writer.close()
    else_writer.close()
def zr_feature_add(result):
    dict_news= {'新浪':1,'报道':1,'报导':1,'路透':1,'新华':1,'业内':1,'政策':1,'企业':1,'十大':1,'新闻':1,'日报':1,'晚报':1,'政府':1,'限*令':1,'肉毒':1,'恒天然':1,'垄断':1,'公司':1,'贿赂':1,'第一口奶':1,'亚硝酸盐':1,'结石门':1,'信任危机':1,'南京妇幼':1,'工信部':1,'发改委':1,'销毁':1,'股':1,'曝':1,'财报':1,'工厂':1,'记者':1}
    dict_ask={'怎么样':1,'哪个好':1,'想问':1,'不知道':1,'怎么回事':1,'怎么办':1,'好不好':1,'为什么':1,'要不要':1,'急急':1,'求问':1,'跪求':1,'急问':1,'吗':1,'怎样':1,'哪个':1,'咋样':1,'多少钱':1,'哪里卖':1,'请问':1,'如何':1}
    dict_feeling={'推荐':1,'建议':1,'不错':1,'很好':1,'还可以':1,'挺好':1,'还行':1,'上火':1,'有火':1,'便秘':1,'便便':1,'大便':1,'拉肚子':1,'绿便':1,'长肉':1,'觉得':1,'感觉':1,'溶':1,'甜':1,'苦':1,'腥':1,'香':1,'腹泻':1,'口感':1,'味道':1,'口味':1,'新鲜':1,'建议':1,'适应':1,'绿屎':1,'眼屎':1,'便秘':1,'便便':1,'很硬':1,'拉不出':1,'拉稀':1,'拉西':1,'闹肚子':1,'拉肚子':1,'稀便':1,'腹泻':1,'吐奶':1,'胀气':1,'上火':1,'烦躁':1,'红肿':1,'消化':1,'肠道':1,'肠胃':1,'过敏':1,'疹':1,'腥':1,'甜':1,'口感':1,'味道':1,'粘':1,'苦':1,'溶解':1,'好喝':1,'喜欢喝':1,'爱喝':1,'难喝':1,'营养':1,'DHA':1,'ARA':1,'叶酸':1,'蛋白质':1,'钙':1,'小安素':1,'原装':1,'进口':1,'外国':1,'国外':1,'原罐':1,'海外':1,'产地':1,'英国':1,'荷兰':1,'新西兰':1,'纽4西兰':1,'澳洲':1,'澳大利亚':1,'德国':1,'美国':1,'美版':1,'爱尔兰':1,'香港':1,'港版':1,'安全':1,'放心':1,'可信':1,'相信':1,'信赖':1,'信任':1,'一直喝':1,'一直买':1,'大品牌':1,'有名品牌':1,'知名品牌':1,'脑部发育':1,'大脑发育':1,'脑发育':1,'大脑生长':1,'聪明':1,'机灵':1,'益智':1,'智力':1,'智力发育':1,'免疫力':1,'免疫能力':1,'抵抗力':1,'保护力':1,'预防':1,'疾病':1,'学习能力':1,'好学':1,'观察力':1,'思考力':1,'活动力':1,'学习':1,'学习基础':1,'呼吸系统':1,'感冒':1,'咳嗽':1,'气管炎':1,'支气管炎病':1,'过敏':1,'消化不良':1,'烦躁':1,'胀气':1,'肚胀':1,'涨肚':1,'胀肚':1,'拉稀':1,'拉西':1,'闹肚子':1,'绿便':1,'绿色便便':1,'热气':1,'心疼':1,'血':1,'乳清蛋白':1,'水解':1,'糖':1,'不拉':1,'现在喝的':1,'过敏':1,'湿疹':1,'红肿':1,'疹':1,'痘':1,'甜':1,'疙瘩':1,'肺炎':1,'咳嗽':1,'溶解':1,'颗粒':1,'细腻':1,'母乳':1,'粑粑':1,'消化':1,'吸收':1,'肚肚':1,'肚子':1}
    dict_else={'排行':1,'奶粉地图':1,'奶粉世界地图':1,'排名':1,'奶粉选购常识':1,'奶粉选择不用愁':1,'奶粉科普':1,'鉴别真假':1,'怎么辨别':1,'如何辨别':1,'十强':1,'10强':1}
    dict_shopping={'买':1,'卖':1,'渠道':1,'价格':1,'供应':1,'有效期':1,'保质期':1,'哪里*买':1,'促销':1,'降价':1,'价格':1,'降价':1,'便宜':1,'贵':1,'实惠':1,'划算':1,'低价':1,'性价比':1,'超市':1,'药房':1,'药店':1,'卖场':1,'代购':1,'海淘':1,'直邮':1,'发货':1,'晒单':1,'万宁':1,'有卖':1,'哪卖':1,'哪买':1,'哪里买':1,'那里卖':1,'哪有':1,'多少钱':1,'哪个地方买':1,'平台':1,'钱':1}
    num_word = [0,0,0,0,0]
    add_words = ''
    for word in result.split(' '):
        if word in dict_shopping:
            num_word[4]+=1
        if word in dict_news:
	    num_word[0]+=1
	if word in dict_ask:
	    num_word[1]+=1
	if word in dict_feeling:
	    num_word[2]+=1
	if word in dict_else:
	    num_word[3]+=1
	    
    if num_word[0] == 1:
	add_words=' new_special_1'
    elif num_word[0] > 1:
	add_words=' new_special_2 new_special_1'
    if num_word[1] == 1:
	add_words=add_words + ' ask_special_1'
    elif num_word[1] > 1:
	add_words=add_words + ' ask_special_2 ask_special_1'
    if num_word[2] == 1:
	add_words=add_words + ' feeling_special_1'
    elif num_word[2] > 1:
	add_words=add_words + ' feeling_special_2 feeling_special_1'
    if num_word[3] == 1:
	add_words=add_words + ' else_special_1'
    elif num_word[3] > 1:
	add_words=add_words + ' else_special_2 else_special_1'
    if num_word[4] == 1:
	add_words=add_words + ' shopping_special_1'
    elif num_word[4] > 1:
	add_words=add_words + ' shopping_special_2 shopping_special_1'
    add_word = add_words.strip()
    result = '%s %s'%(result,add_word)
    return  result 

if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    # cmd argument parser
    usage = 'cut -t [pos] -f <file path> --template <template path> --brands <brand path or single brand name>'
    parser = OptionParser(usage)
    parser.add_option("-f", dest="pathName")
    parser.add_option("-t", dest="type", action="store_true")
    parser.add_option("--template", dest = "templatePath")
    parser.add_option("--brands", dest = "brands")

    opt, args = parser.parse_args()
    if opt.pathName is None:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)

    posFlag = False
    if not (opt.type is None):
        posFlag = True
    
    dBrand = {}
    if not opt.brands is None:
       if os.path.exists(opt.brands):
           oReader = open(opt.brands, 'rb')
           for sLine in oReader.readlines():
               dBrand[sLine.strip().lower()] = 1 
       else:
           dBrand[opt.brands.strip().lower()] = 1
 
    lFeatureTemplate = []
    if not opt.templatePath is None:
        if os.path.exists(opt.templatePath):
            oReader = open(opt.templatePath, 'rb')
            for sLine in oReader.readlines():
                lFeatureTemplate.append(sLine.strip())

    arg_name = opt.pathName 
    if os.path.isdir(arg_name):
        for root, dirs, files in os.walk(arg_name):
            #print root, dirs, files
            for file_name in files:
                cut_file( root + '/' + file_name, posFlag, lFeatureTemplate, dBrand)
            break    
    else:
        cut_file( arg_name, posFlag, lFeatureTemplate, dBrand)
