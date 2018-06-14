#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
label process optimization

How to optimize the labeling process on the output sequence by classifier?
How about focus on the blur section other than label all the instances?
The blur setection is the instances than the predict probability values among different classes have small vairance. e.g., max-next < 0.15

input format:
    .prob
    label predict probabilities

Usage:
    processopt.py --mode <0|1|2> --threshold <thres> <probfile> 
        mode 0; check the diff value smaller than thres
        mode 1; check the performance score value above thres
        mode 2; check the labeling labor as the thres

        threshold: range 0-1 if not set


"""

import string
import sys,time
import os, os.path
import random
import logging
import numpy as np
from optparse import OptionParser
from sklearn import metrics

logger = logging.getLogger(__name__)

#global
CHECK_DIFFTHRESHOLD = 0
CHECK_SCORETHRESHOLD= 1
CHECK_LABORRATIO = 2

class ProbFile:
    """
       define three different check mode 
    """
    def __init__(self, filename = '', mode = 0, threshold = 0):
        self.data = []
        self.classnum = 0
        self.num = 0

        # parameters
        self.filename = filename
        self.mode = mode
        self.threshold = threshold

    def load_data(self):
        """
            input:
                .prob
                label predict probabilities
        """
        self.data = np.loadtxt(self.filename)
        self.classnum = self.data.shape[1] - 2
        self.num = self.data.shape[0]
        logger.info('mode %s, threshold %s, loading data end at %s', self.mode, self.threshold, self.data.shape)


    def score(self, y_test, pred, quiet = True):
        precision = metrics.precision_score(y_test, pred, average=None)
        recall = metrics.recall_score(y_test, pred, average=None)
        score = metrics.accuracy_score(y_test, pred)
        cmstr = metrics.confusion_matrix(y_test, pred)
        if not quiet:
            logger.info("accuracy :   %0.3f" , score)
            logger.info("precision:   %s" , precision)
            logger.info("recall   :   %s" , recall)
            logger.info("\n%s", cmstr)
        
        return precision, recall, cmstr

    def reorder_bydiff(self):
        """
        Output a order of predictions, set the most difficult ones in the beginning, i.e., sort by diff
        Return:
            ndata   ; prob matrix, diff, error
            idx     ; sorted index
        """
        ndata = self.calc_diff()
        # sort by diff
        idx = np.argsort(ndata[:,2+self.classnum])
        return ndata, idx

    def check(self):
        """
            Check a threshold of the blur area to select out a subset for mannual labeling
            output the processing performance
        """
        ndata = self.calc_diff()

        # sort by diff
        idx = np.argsort(ndata[:,2+self.classnum])
        sdata = ndata[idx]
 
        # get the error items 
        erroridx = np.where(sdata[:,2 + self.classnum + 1] == 0)
        error = sdata[erroridx]

        if self.mode == CHECK_DIFFTHRESHOLD:
            #
            # checktype 0: here test for diff spans
            #
            for span in np.arange(0.1, 1,0.1):
                spanidx = np.where(sdata[:, 2+self.classnum] < span)
                logger.info('fix span with diff < %s, total %s instances', span, spanidx[0].shape)
                #calc performance after label fix for this span
                pred = np.copy(sdata[:,:2])
                #fix all the spanidx items
                #fixdata = pred[spanidx]
                #fixdata[:,1] = fixdata[:,0]
                for fixid in spanidx[0]:
                    fixline = pred[fixid]
                    fixline[1] = fixline[0]

                #calc score
                self.score(pred[:,0], pred[:,1], quiet = False)

        elif self.mode == CHECK_SCORETHRESHOLD:
            #
            # checktype 1: check for precision/recall above thres
            # 
            pred = np.copy(sdata[:,:2])
            #fix one by one
            labelcnt = 0
            modifycnt = 0
            maxnomodifyspan = 0
            nomodifyspan = 0

            precision, recall, cmstr = self.score(pred[:,0], pred[:,1], quiet = True)
            logger.info('eval-start:%5d %0.2f %5d %0.2f %3d %d %s %s', 0, 0, 0,0, 0, self.num, precision, recall) 

            for fixline in pred:
                if fixline[1] != fixline[0]:
                    modifycnt += 1
                    
                    maxnomodifyspan = max(maxnomodifyspan, nomodifyspan)
                    # reset span
                    nomodifyspan = 0
                else:
                    nomodifyspan += 1

                fixline[1] = fixline[0]
                labelcnt += 1

                #calc score
                precision, recall, cmstr = self.score(pred[:,0], pred[:,1], quiet = True)
                
                # if precision.all() >= self.threshold and recall.all() >= self.threshold:
                if np.sum(precision >= self.threshold) == self.classnum and np.sum(recall >= self.threshold) == self.classnum:
                   #logger.info('eval-stop at labelcnt=%d, ratio = %0.2f in total %d records, precision %s, recall %s', labelcnt, labelcnt*1.0 / self.num, self.num, precision, recall) 
                   logger.info('eval-stop :%5d %0.2f %5d %0.2f %3d %d %s %s', labelcnt, labelcnt*1.0 / self.num, 
                           modifycnt, modifycnt *1.0/ labelcnt, maxnomodifyspan, self.num, precision, recall) 
                   logger.info('stop at labelcnt=%d, ratio = %0.2f in total %d records', labelcnt, labelcnt*1.0 / self.num, self.num) 
                   logger.info("precision:   %s" , precision)
                   logger.info("recall   :   %s" , recall)
                   logger.info("\n%s", cmstr)

                   break
 
    
        else:
            #
            # checktype 2: check for labor count
            # 
            pred = np.copy(sdata[:,:2])
            #fix one by one
            labelcnt = 0
            modifycnt = 0
            maxnomodifyspan = 0
            nomodifyspan = 0

            precision, recall, cmstr = self.score(pred[:,0], pred[:,1], quiet = True)
            logger.info('eval-start:%5d %0.2f %5d %0.2f %3d %d %s %s', 0, 0, 0,0, 0, self.num, precision, recall) 

            for fixline in pred:
                if fixline[1] != fixline[0]:
                    modifycnt += 1
                    
                    maxnomodifyspan = max(maxnomodifyspan, nomodifyspan)
                    # reset span
                    nomodifyspan = 0
                else:
                    nomodifyspan += 1

                    if nomodifyspan >= self.threshold:
                        # stop here
                        #calc score
                        precision, recall, cmstr = self.score(pred[:,0], pred[:,1], quiet = True)
                        logger.info('eval-stop :%5d %0.2f %5d %0.2f %3d %d %s %s', labelcnt, labelcnt*1.0 / self.num, 
                                modifycnt, modifycnt *1.0/ labelcnt, nomodifyspan, self.num, precision, recall) 
                        break
 

                fixline[1] = fixline[0]
                labelcnt += 1


    def calc_diff(self):
        """
            calculate the probdiff = max - second max
            return: 
                data diff error
        """
        #calc the blurness for each instance
        # max-second max < threshold
        ind = np.argsort(self.data[:,3:])[:,-2:]
        #maxv = self.data[np.transpose(np.vstack(np.arange(self.num), ind[:,-1]))]
        maxv = [np.arange(self.num), ind[:,-1] + 3]
        #logger.info('maxvidx=%s', maxv)
        maxv = self.data[maxv]
        #logger.info('maxv=%s', maxv)

        maxv2 = self.data[np.arange(self.num), ind[:,-2] + 3]
        diff = maxv- maxv2

        error = self.data[:,1] - self.data[:,2]
        # set to 1/0
        error = error==0

        ndata = np.hstack((self.data, diff.reshape((self.num, 1))))
        ndata = np.hstack((ndata, error.reshape((self.num, 1))))
        with open('add-diff.txt', 'w') as outf:
            for idx in range(len(ndata)):
                outf.write('%s %s\n' % (" ".join(['%d' % p for p in ndata[idx, :3]]), " ".join(['%.04f'%p for p in ndata[idx, 3:]])))
        
        return ndata


def load_option():
    op = OptionParser()
    op.add_option("--report",
                  action="store_true", dest="print_report",
                  help="Print a detailed classification report.")
    op.add_option("--mode",
                  action="store", type = int, default=0,
                  help="set the check mode, 0 by default to check the diff.")
    op.add_option("--threshold",
                  action="store", type = float, default=0,
                  help="set the threshold of check.")
 
    (opts, args) = op.parse_args()

    if len(args) >= 1:
        opts.filename = args[0]
    else:
        opts.filename = 'model.prob'
    return opts

if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 2:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)

    # cmd argument parser
    opts = load_option()

    pf = ProbFile(opts.filename, opts.mode, opts.threshold)

    pf.load_data()

    pf.check()


