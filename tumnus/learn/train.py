#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
A Classifier Application
"""
import logging
import numpy as np
from optparse import OptionParser
import sys, os
from time import time
import matplotlib.pyplot as plt
from tumnus.learn import Learner

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
#from sklearn.preprocessing import scale
#from gensim.models.word2vec import Word2Vec
#parse commandline arguments
def load_option():
    op = OptionParser()
    op.add_option("--fs_type",
                  action="store", type=str, default="no",
                  help="Define the feature selection type, such as chi2.")
    op.add_option("--n_features",
                  action="store", type=int, default=0,
                  help="Define the feature number of the feature selection.")
    op.add_option("--use_word2vec",
                  action="store", type=str, default="", 
                  help="Use a word2vec vectorizer, otherwise a tfidf vectorizer. Provide the model file name")
    op.add_option("--vectorizer_type",
                  action="store", type=str, default='tfidf',
                  help="Define the vectorizer type, tfidf by default. cout, template as other choices.")
    op.add_option("--ngram_range",
                  action="store", type=str, default="1,1",
                  help="Define the ngram_range for vectorizer.")
    op.add_option("--bin_class",
                  action="store", type="int", default=-999,
                  help="Build a binary classifier on this specific label.")
    op.add_option("--recordid",
                  action="store_true", 
                  help="define use recordid or id in dataset.")
    op.add_option("--balanced",
                  action="store_true", 
                  help="define wether balance the class weight.")
    op.add_option("--classifier",
                  action="store", type=str, default="lsvc",
                  help="define the classifier, lsvc as linearsvc, rf as random forest.")
    op.add_option("--vocabulary",
                  action="store", type=str, default="",
                  help="define the pre feature filter file name, by default pre_filter.dict.")
    op.add_option("--testmodel",
                  action="store", type=str, default="", 
                  help="load a pre-trained model file, use it for later prediction and test.")
    op.add_option("--trainfile",
                  action="store", type=str, default="", 
                  help="define the train dataset file name, train.cut by default.")
    op.add_option("--testfile",
                  action="store", type=str, default="", 
                  help="define the test dataset file name, test.cut by default.")
    op.add_option("--appname",
                  action="store", type=str, default="", 
                  help="define the application name of this run, null by default.")
    op.add_option("--debug",
                  action="store_true", 
                  help="Show debug info.")
    op.add_option("--h",
                  action="store_true", dest="print_help",
                  help="Show help info.")
    
    (opts, args) = op.parse_args()
   
    #type convert
    items = opts.ngram_range.split(',')
    opts.ngram_range = (int(items[0]),int(items[1]))
    if not opts.appname:
        opts.appname = 'test'

    #set default
    
    if opts.print_help:
        print(__doc__)
        op.print_help()
        print()
        sys.exit(0)

    logger.info('Start Classifier:[%s], features:[fs:%s, vector:%s, vocabulary:%s, ngram:%s]'%( opts.classifier, opts.fs_type, opts.vectorizer_type, opts.vocabulary, opts.ngram_range))


    return opts

def get_classifier(classifier = 'lr', balanced = True):
    """
        Input:
            clfname, class weight type
        Return:
            fullname, clf object
    """
    fullname = classifier + ('_balanced' if balanced else '')

    if balanced:
        class_weight = 'balanced' if classifier != 'rf' else 'balanced_subsample'
    else:
        class_weight = 'auto'
    
    #
    if classifier == "lsvc":
        clf = LinearSVC(penalty='l1',dual=False, tol=1e-3, class_weight=class_weight)
    elif classifier == "lsvcl2":
        clf = LinearSVC(penalty='l2', tol=1e-4, class_weight=class_weight)
    elif classifier == 'rf':
        clf = RandomForestClassifier(n_estimators=100, n_jobs=4,criterion='entropy', min_samples_split=1,class_weight = class_weight)
    elif classifier == 'lr':
        clf = LogisticRegression(class_weight = class_weight)
    elif classifier == 'lrl1':
        clf = LogisticRegression(class_weight = class_weight, penalty='l1')
    elif classifier == 'mulnb':
        clf = MultinomialNB() 
        # clfset.append(("LR_Balanced",LogisticRegression(class_weight = class_weight), penalty='l1'))
        #clfset.append(('LinearSVC l1 penality' ,LinearSVC(C=100, penalty='l1',dual=False, tol=1e-3, class_weight=class_weight)))
        #clfset.append(('LinearSVC l1 penality' ,LinearSVC(C=0.1, penalty='l1',dual=False, tol=1e-3, class_weight=class_weight)))
        #clfset.append(('SGD l1 penality' , SGDClassifier(alpha=.0001, n_iter=50, penalty='l1',class_weight=class_weight)))
        #clfset.append(('Ridge Classifier', RidgeClassifier(tol=1e-2, solver="lsqr")))
        #clfset.append(('SVM Classifier(rbf)', SVC(C=0.1, tol=1e-3, kernel="sigmoid", coef0=0.1)))
        #clfset.append(('SVM Classifier(poly)', SVC(tol=1e-3, kernel="poly")))
        
        #clfset.append(("Random forest",RandomForestClassifier(n_estimators=100)))
        #clfset.append(("BernoulliNB", BernoulliNB(alpha=.01)))
        #clfset.append(("linearSVC pipeline", Pipeline([
        #  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
        #  ('classification', LinearSVC())
        #])))
 
    return fullname, clf

if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    opts = load_option()


    print  'type(opts)' , type(opts)

    # this is the training app
    if opts.testmodel:
        #test model, no train process
        if not opts.testfile:
            logger.info('testfile not defined, quit...')
            sys.exit(-1)

        # use the model name, and the appname
        fullname = opts.testmodel + opts.appname

        predictor = Learner(train = False)
        predictor.load_model(opts.testmodel)
        predictor.load_dataset(opts.testfile, recordid = opts.recordid, bin_class = opts.bin_class)
        predictor.transform()
        predictor.feature_select()
        predictor.predict(savename = fullname)

    else:
        if not opts.trainfile:
            logger.info('trainfile not defined, quit...')
            sys.exit(-1)

        # get the classifier
        # specification for the full name
        # appname_dataset_classifier_balanced_fstype_vectorizertype_nfeatures_ngram
        fullname, clf = get_classifier(opts.classifier, opts.balanced)
        fullname = '%s_%s_%s_%s_%s%s'%(
                opts.appname, fullname, 
                opts.fs_type, opts.vectorizer_type, 
                opts.n_features if opts.fs_type!='no' else opts.vocabulary if opts.vocabulary else 0, 
                '_%s-%s'%opts.ngram_range if opts.ngram_range != (1,1) else '')

        # setup the learner
        trainer = Learner(train = True)
        trainer.load_dataset(opts.trainfile, recordid = opts.recordid, bin_class = opts.bin_class)
        trainer.init_vocabulary(opts.vocabulary)
        trainer.init_vectorizer(type = opts.vectorizer_type, ngram_range=opts.ngram_range)
        trainer.init_featureselector(type = opts.fs_type, selectCnt = opts.n_features)
        trainer.transform()
        trainer.feature_select(savename = fullname)
        trainer.train_model(clf, fullname)
    
        if opts.debug:
            logger.info(str(trainer).replace('\n',' '))

        #check if test
        if opts.testfile:
            predictor = Learner(train = False)
            predictor.copy_model(trainer)
            predictor.load_dataset(opts.testfile, recordid = opts.recordid, bin_class = opts.bin_class)
            predictor.transform()
            predictor.feature_select()
            predictor.predict(savename = fullname)

