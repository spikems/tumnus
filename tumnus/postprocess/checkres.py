#coding=utf-8
#!/usr/bin/python

"""
create a html view from .res, add highlight for features
    input:
        class, predict, content

    Useage:
        checkres <.res> <.coef>

"""

import sys,os
import csv
import logging
from jieba.norm import norm_cut, norm_seg

def cut_input(input, posFlag):
    '''
    cut a input string, return utf-8 string
    '''

    if posFlag == True:
        result = norm_seg(input)
        wordsList = []
        for w in result:
            wordsList.append(w.word + '_' + w.flag)
        words = " ".join(wordsList)
    else:
        words = " ".join(norm_cut(input))
    #return words.encode('utf-8')
    return words

def read_res(resfile):
    """
    Read in res file and return a list of records

    """
    f = open(resfile,'r')
    resData = []
    for l in f:
        pos1 = l.find(' ')
        pos2 = l.find(' ', pos1+1)
        resData.append(l[pos2+1:].strip())

    print 'read_res readin reccnt = ', len(resData)
    return resData

def loadfeatures(coeffile):
    """
    coef file:
    neg\t车存\t-9.74207363139
    """
    features = {}
    with open(coeffile) as cf:
        for line in cf:
            items = line.strip().split('\t')
            #check if it has pos tag
            pos =items[1].find('_')
            if pos > 0:
               items[1] = items[1][:pos] 
            weight = float(items[2])
            #use only neg features
            if weight != 0:
                features[items[1]] = weight
    return features

def highlight(text, features):
    """
    output highlight <>word<> with feature match
    """

    # words = [x.encode('utf-8') for x in norm_cut(text)]
    words = text.split()
    output = []
    #pick colors, red1, red2/blue1, blue2
    colors=['#FF0000','#FF8000','#0000FF','#0080FF']
    flip = [0,0]
    for w in words:
        if w in features.keys():
            #TODO: get weight to make a color

            # just add a color
            if features[w] > 0:
                #red
                output.append('<span style="color:' + colors[flip[0]%2] + '">' + w + '</span>')
                flip[0] += 1
            else:
                output.append('<span style="color:' + colors[2 + flip[1]%2] + '">' + w + '</span>')
                flip[1] += 1
        else:
            output.append(w)

    return " ".join(output)



def items2html(items_fn, coeffile):
    logger.info('Start items2html, items_fn= %s, filter = %s', items_fn, coeffile)

    html_fn = os.path.splitext(os.path.basename(items_fn))[0] + '.html'

    items = read_res(items_fn)
    features = loadfeatures(coeffile)
    if not features:
        logger.info('load empty feature file, quit', coeffile)

    # sort the data by f1 score
    #items = sorted(items[1:], key = lambda x: float(x[3])*float(x[4])/(float(x[3]) + float(x[4]) + 1e-12))

    htmlfile = open(html_fn,"w")# Create the HTML file for output
    # resultData = []

    #colidx = {'id':0,'stype':9, 'title':3, 'text':10}
    colidx = [0,9,3,10]
    colname = ['id','stype', 'title', 'text']
    # colwidth = ["6", "2", "30", "30", "30", "2"]
    # colwidth = ["60", "200", "200", "20", "20", "100","20","100","20","20"]
    #colwidth = ["6%", "6%", "20%", "68%"]
    colwidth = ["6%", "6%", "6%", "82%"]
    # write <table> tag
    #       table {border-collapse:collapse; table-layout:fixed; width:1800px;}
    htmlfile.write('''<html><head><meta charset="utf-8">
            <style>
            table {border-collapse:collapse; table-layout:fixed; }
            table td {border:solid 1px #fab;word-wrap:break-word;vertical-align:top}
            </style></head><table width="100%" >''')
    #width="100%"

    htmlfile.write('<tr>')# write <tr> tag
    for index in range(len(colname)):
        htmlfile.write('<th width="' + colwidth[index] + '%">' + colname[index] + '</th>')

    htmlfile.write('</tr>')
    rownum = 0

    # generate table contents
    i=0
    for row in items: # Read a single row from the CSV file
        # print row[3],row[4]
        htmlfile.write('<tr>')
        for rid in range(len(colidx)):
            #if rid == 0:
            #    #url
            #    htmlfile.write('<td width="' + colwidth[rid] + '%"><a href="' + str(row[rid]) + '", target="_blank">' + str(row[rid]) + '</a></td>')

            #else:
            # if rid == 0 or rid == 1:
            #     htmlfile.write('<td width="' + colwidth[rid] + '%">' + str(row[colidx[rid]]) + '</td>')
            # else:
            #     htmlfile.write('<td width="' + colwidth[rid] + '%">' + highlight(str(row[colidx[rid]]), features) + '</td>')
            if rid <3:
                 htmlfile.write('<td width="' + colwidth[rid] + '%">' + ' ' + '</td>')
            else:
                 htmlfile.write('<td width="' + colwidth[rid] + '%">' + highlight(str(row), features) + '</td>')

        htmlfile.write('</tr>')
        rownum += 1
    # write </table> tag
    htmlfile.write('</table>')
    # print results to shell
    print "Created " + str(rownum) + " row table."

if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 3:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)


    items2html(sys.argv[1], sys.argv[2])

