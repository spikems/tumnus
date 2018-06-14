#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
near duplication remove for txt dataset

use simhash-py package, refer to: https://github.com/seomoz/simhash-py

input format:
    .txt
        txt     ; content each line
    .cut
        label id tokens...  ; skip the first two column
output files:
    outfile     ; deduped output file
    outfile.log ; log for each group of records
    outfile.grp ; duplicated id file, one group each line

Usage:
    dedup.py -i <in file> -o <out file> -b <blocknum> -k <diffbit number>
        -i, -infile     ;
        -o, -outfile    ;
        -b, -blocknum   ; 64 bits fingerprint dividen blocks, parameter for simhash package
        -k, -kbits      ; 0 by default means exact duplication, otherwise for near-duplication
"""

import string
import sys,time
import re, os, os.path
import logging
from optparse import OptionParser
import simhash


def getStartColumn(filename):
    """
        get the start column number according to different file format

        return:
            0   ; all columns
            >0  ; start column id
    """
    if filename.find('.cut') > 0:
        return 3
    return 0

def getStartPos(line, colid):
    """
        get the pos of the start column by the column id
        return:
            -1  ; no colid exist
            >=0 ; start position
    """
    if colid > 0:
        pos = line.find(' ')
        while pos >= 0:
            colid -= 1
            if colid == 0:
                return pos
            pos = line.find(' ', pos+1)

        return -1
    else:
        return 0

def compute(text):
    """
        compute hash for a document by shingles
    """
    #tokens = re.split(r'\W+', text)
    tokens = text.split()

    #logger.debug('%s', ''.join(tokens[:5]))

    phrases = (' '.join(phrase) for phrase in simhash.shingle(tokens, 4))
    #logger.debug('%s', [x for x in phrases])

    hashes = map(simhash.unsigned_hash, phrases)
    return simhash.compute(hashes)

def dedup_near(infile, outfile, b, k, debug = False):
    """
    """
    #
    removelist = []
    grplist = []

    #
    writer = open(outfile, 'w')
    reader = open(infile, 'rb')
 
    startColid = getStartColumn(infile)
    
    # 
    duphash = {}   #hash -> set(lineid) 

    #
    linecnt = 0
    data_h = []   #list of hash val 
    index = {}  # hash val -> lineid
    data_v = {}  # lineid -> data
    for line in reader:
        #apos = line.find(' ')
        apos = getStartPos(line, startColid)
        
        if apos >= 0:
            hash = compute(line[apos:])
            data_h.append(hash) 
            #here duplicate hash exist
            if hash in index:
                #add the same line into the same group
                #set grpid to the grpid of the last lineid with equal hash value
                if hash in duphash:
                    duphash[hash].append(linecnt)
                else:
                    #init with the first lineid
                    duphash[hash] = [index[hash]]
                    duphash[hash].append(linecnt)
            else:
                index[hash] = linecnt

            data_v[linecnt] = line
            #data_v[linecnt] = line[apos:]
        linecnt += 1

    #logger.info('lines=%s', '\n'.join([data_v[x] for x in range(5)]))
    #    logger.info('hash=%s', data_h[:5])
    if debug:
        with open('hash.txt', 'w') as hashf:
            for h in data_h:
                hashf.write('%s\n'%h)
        with open('hash_full.txt', 'w') as hashf:
            for idx in range(len(data_h)):
                hashf.write('%s %s'%(data_h[idx], data_v[idx]))

    # output the match group to .log
    grpwriter = open(outfile + '.log','w')
    for key in duphash.keys():
        ids = duphash[key]
        #only the first one reserved
        removelist.extend(ids[1:])
        grplist.append(ids)

        grpwriter.write('ids:%s\n'%' '.join([str(x) for x in ids]))
        #write the group of match
        for lineid in ids:
            grpwriter.write('%s'%data_v[lineid])

        grpwriter.write('==================\n')

    logger.info('duphash removecnt=%d, linecnt = %s', len(removelist), linecnt)
    #find all pairs of match
    matches = simhash.find_all(data_h, b, k)

    marks = {} #lineid -> groupid
    grpindex = {} # groupid -> [lineids]
    groupid = 0
    
    for A, B in matches:
        grpidA, grpidB = -1, -1
        if index[A] in marks:
            grpidA = marks[index[A]]
        if index[B] in marks:
            grpidB = marks[index[B]]
        if grpidA == -1 and grpidB == -1:
            #new pair
            marks[index[A]] = groupid
            marks[index[B]] = groupid
            grpindex[groupid] = set([index[A], index[B]])

            groupid += 1
        elif grpidA == -1:
            #add B to group A
            marks[index[A]] = grpidB
            grpindex[grpidB].add(index[A])
        elif grpidB == -1:
            marks[index[B]] = grpidA
            grpindex[grpidA].add(index[B])
        else:
            #merge two old groups
            for lid in grpindex[grpidB]:
                marks[lid] = grpidA
                grpindex[grpidA].add(lid)
            grpindex[grpidB].clear()

    # output the groups
    #grpwriter = open(outfile + '.log', 'w')
    linecntx = 0
    for grp in grpindex.keys():
        if grpindex[grp]:
            ids = [lid for lid in grpindex[grp]]
            ids = sorted(ids, reverse = True)

            linecntx += len(ids[1:])
            #output the first one
            removelist.extend(ids[1:])
            grplist.append(ids)

            #output all
            grpwriter.write('ids:%s\n'%ids)
            #write the group of match
            for lineid in ids:
                grpwriter.write('%s'%data_v[lineid])

            grpwriter.write('==================\n')
    
    logger.info('total removecnt=%d, linecntx = %s, grpcnt=%d', len(removelist), linecntx, len(grpindex.keys()))

    #out put final result
    remove = set(removelist)
    for lid in range(linecnt):
        if lid not in remove and lid in data_v:
            writer.write('%s'%data_v[lid])
    
    # output the grplist
    with open(outfile + '.grp','w') as grpf:
        for grp in grplist:
            if len(grp) > 1:
                grpf.write('%s\n'%' '.join([str(x) for x in grp]))
            else:
                grpf.write('%s\n'%grp[0])

    reader.close()
    writer.close()

def dedup_exact(infile, outfile):
    '''
        exact duplication remove by content dictionary
    '''
    writer = open(outfile, 'w')
    reader = open(infile, 'rb')
 
    startColid = getStartColumn(infile)

    duplicates = {}
    dupcnt = 0
    linecnt = 0
    # parse the records
    # label hidden text
    #
    for line in reader:
        #apos = line.find(' ')
        apos = getStartPos(line, startColid)

        if apos >=0 and line[apos:] in duplicates:
            dupcnt += 1
            continue
        if apos >= 0:
            duplicates[line[apos:]] = 1

            writer.write(line)
        linecnt += 1

    reader.close()
    writer.close()
    print('%d lines written, skip %s lines duplicate'%(linecnt, dupcnt))

# parse commandline arguments
def load_option():
    op = OptionParser()
    op.add_option("-b",
                  action="store", type=int, default=6, 
                  help="64 bits fingerprint dividen blocks, parameter for simhash package.")
    op.add_option("-k",
                  action="store", type=int, default=0,
                  help="0 by default means exact duplication, otherwise for near-duplication.")
    op.add_option("-i",
                  action="store", type=str, dest="infile",
                  help="input filename.")
    op.add_option("-o",
                  action="store", type=str, dest="outfile",
                  help="input filename.")

    (opts, args) = op.parse_args()

    return opts

if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    opts = load_option()

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    #logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    #logger.info('option:%s, %s, %s, %s', opts.infile, opts.outfile, opts.b, opts.k)
    if opts.infile is None or opts.outfile is None:
        print(__doc__)
        sys.exit(0)


    if opts.k == 0:
        dedup_exact(opts.infile, opts.outfile)
    else:
        dedup_near(opts.infile, opts.outfile, opts.b, opts.k)

