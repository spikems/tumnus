#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Classification of text documents using sparse features
Code based on the tutorial code example of scikit-learn library.
"""

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys, os
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
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
from sklearn.preprocessing import scale
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib

#Global Variables
#_categories = ['alwayson','campaign','sales']
#_categories = {u'购买':0, u'问询':1, u'使用感受':2,u'新闻':3,u'水军':4,u'其他':5}
#_categories = {'购买':0, '问询':1, '使用感受':2,'新闻':3,'水军':4,'其他':5}
_categories = {'jichu':['alwayson','campaign','sales'],
                'ziran':['购买', '问询', '使用感受','新闻','水军','其他']}


#===================
# Class DataSet
#===================
class Bunch(dict):
    """Container object for datasets
    Dictionary-like object that exposes its keys as attributes.
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass

def run_filter(x_data, filter):
    """
        x_data: train or test dataset
        [strings]

    """
    if not filter:
        return x_data
    data=[]
    for line in x_data:
        words = line.strip().split(' ')
        features = []
        for w in words:
            #add special comb feature pass-through
            #if w.find('#') > 0:
            #    features.append(w)
            # elif filter:
            if w in filter:
                #check the filter
                features.append(w)
        data.append(" ".join(features))
    return data

def load_file(fname, opts, filter={}):
    """Load and return the dataset (classification).
       -1 23456 content...
        
       if opts.bin_class in the range of len(_categories[])
        then setup a binary classifier for this special label(one-vs-rest)

    """
    bin_class = opts.bin_class
    categories = _categories[opts.dataname]

    data = []
    target = []
    ids = []
    with open(fname, 'r') as inf:
        for line in inf:
            line = line.strip()

            pos1 = line.find(' ')
            #put the id into data, extract into X_id in the future
            pos2 = line.find(' ', pos1+1)
            #pos2 = pos1
            ids.append(line[pos1+1:pos2])

            #if filter:
            #    features = []
            #    words = line[pos2+1:].split(' ')
            #    for w in words:
            #        #add special comb feature pass-through
            #        if w.find('#') > 0:
            #            features.append(w)
            #        elif w in filter:
            #            #check the filter
            #            features.append(w)
            #    data.append(" ".join(features))
            #else:
            #    data.append(line[pos2+1:])
            features = []
            words = line[pos2+1:].split(' ')
            for w in words:
                #add special comb feature pass-through
                if w.find('#') > 0:
                    features.append(w)
                elif filter:
                    if w in filter:
                        #check the filter
                        features.append(w)
                else:
                    features.append(w)
            data.append(" ".join(features))

            #if bin_class != -999 and bin_class < len(categories):
            if bin_class != -999:
                target.append(0 if int(line[:pos1]) == bin_class else 1)
            else:
                #target.append(int(line[:pos1]) + 1)
                target.append(int(line[:pos1]))

    return Bunch(data=data, target=target, ids=ids, target_names= categories)


#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(w2v, text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
#if count != 0:
#        vec /= count
    return vec


class DataSet():
    """
        class of load dataset and preprocess
        opts settings:
           select_chi2
           all_categories
           use_tf
           n_features
           negonly
    """
    def __init__(self):
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_names = self.categories = None
        self.X_id = None
        self.data_test = None
        self.data_train = None
        self.name = ''


    def preprocess(self, opts):
        """
            opts
        """
        trainfile = opts.trainfile
        testfile = opts.testfile
        filterfile = opts.filterfile

        #get name as (name-chi2-vectortype)
        if opts.testmodel:
            self.name = opts.testmodel + '-Testmodel'
        else:
            self.name = '%s-%s'%(opts.select_chi2, 'count-%s'%opts.n_features if opts.use_tf else 'word2vec' if opts.use_word2vec else 'tfidf')

        if opts.expname:
            self.name += '-' + opts.expname

        # Load some categories from the training set
        if opts.all_categories:
            categories = None
        else:
            categories = _categories
        
        if opts.filtered:
            remove = ('headers', 'footers', 'quotes')
        else:
            remove = ()
        
        print("Loading senti dataset for categories:")
        print(categories if categories else "all")
        
        print("load filter")    
        data_filter = load_filter(filterfile)
        print("filter size = %s"%len(data_filter.keys()))
        
        #data_train = load_file(trainfile, opts.negonly, data_filter)
        #data_test = load_file(testfile, opts.negonly, data_filter)
        # run filter later, for word 2-gram
        data_train = load_file(trainfile, opts)

#if testfile == 'auto':
#           #auto split the train data file 

        data_test = load_file(testfile, opts)
        print('data loaded')
        
        categories = data_train.target_names    # for case categories == None
        
        
        def size_mb(docs):
            return sum(len(s) for s in docs) / 1e6
        
        data_train_size_mb = size_mb(data_train.data)
        data_test_size_mb = size_mb(data_test.data)
        
        print("%d documents - %0.3fMB (training set)" % (
            len(data_train.data), data_train_size_mb))
        print("%d documents - %0.3fMB (test set)" % (
            len(data_test.data), data_test_size_mb))
        print("%d categories" % len(categories))
        print()
        
        # split a training set and a test set
        y_train, y_test = data_train.target, data_test.target
        
        print("Extracting features from the training/test data using a sparse vectorizer")
        t0 = time()
        ##################
        ##################
        if opts.use_tf:
        #    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
        #                                   n_features=opts.n_features)
            if opts.use_word:
                _train_data = []
                if opts.n_features > 1:
                    #build word-based 2-gram
                    _data = [x.split() for x in data_train.data]
                    for idx in range(len(_data)):
                        for idx_word in range(len(_data[idx])-1):
                            _data[idx].append(_data[idx][idx_word] + _data[idx][idx_word+1])
                    data_train.data = [" ".join(x) for x in _data]
                    data_train.data = run_filter(data_train.data, data_filter)

                    _data = [x.split() for x in data_test.data]
                    for idx in range(len(_data)):
                        for idx_word in range(len(_data[idx])-1):
                            _data[idx].append(_data[idx][idx_word] + _data[idx][idx_word+1])
                    data_test.data = [" ".join(x) for x in _data]
                    data_test.data = run_filter(data_test.data, data_filter)
                else:
                    # no n-gram, run filter here
                    if data_filter:
                        _train_data = run_filter(data_train.data, data_filter)
                        _test_data = run_filter(data_test.data, data_filter)

                # use CountVectorizer for words
                vectorizer = CountVectorizer(analyzer=str.split)
                if _train_data:
                    X_train = vectorizer.fit_transform(_train_data)
                    X_test = vectorizer.transform(_test_data)
                else:
                    X_train = vectorizer.fit_transform(data_train.data)
                    X_test = vectorizer.transform(data_test.data)

            else:
                # vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, opts.n_features), max_features=100000)
                vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, opts.n_features), max_features=100000)
                #by default, the input is cut file, sep with ' '
                #remove all the seps
                _data = [''.join(x.split()) for x in data_train.data]
                X_train = vectorizer.fit_transform(_data)
                X_test = vectorizer.transform(data_test.data)

        elif opts.use_word2vec:
            _data = [x.split() for x in data_train.data]
            n_dim = 600
            ##Initialize model and build vocab
            #load the model file
            logger.info('loading the word2vec model from %s', opts.use_word2vec)
            _w2v = Word2Vec.load_word2vec_format(opts.use_word2vec)

            #make vectors
            X_train = np.concatenate([buildWordVector(_w2v, x, n_dim) for x in _data])

            _data = [x.split() for x in data_test.data]
            #make vectors
            X_test = np.concatenate([buildWordVector(_w2v, x, n_dim) for x in _data])

        else:
            #tfidf by default
            vectorizer = TfidfVectorizer(analyzer=str.split, sublinear_tf=True, max_df=0.9)
        
            if data_filter:
                _train_data = run_filter(data_train.data, data_filter)
                _test_data = run_filter(data_test.data, data_filter)
            else:
                _train_data = data_train.data
                _test_data = data_test.data

            X_train = vectorizer.fit_transform(_train_data)
            X_test = vectorizer.transform(_test_data)


        ####################
        X_id = data_test.ids

        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
        print("trainset n_samples: %d, n_features: %d" % X_train.shape)
        print("testset n_samples: %d, n_features: %d" % X_test.shape)
        print()
        
        # mapping from integer feature name to original token string
        if opts.use_word2vec:
            feature_names = None
            print("no feature names")
        else:
            feature_names = vectorizer.get_feature_names()
            print("feature_names shape=%s"%len(feature_names))

        #if opts.use_tf:
            #using scikit's tokenizer, output is unicode
        #    feature_names = [ x.encode('utf-8') for x in feature_names]
        
        if opts.select_chi2:
            print("Extracting %d best features by a chi-squared test" %
                  opts.select_chi2)
            t0 = time()
            ch2 = SelectKBest(chi2, k=opts.select_chi2)
            X_train = ch2.fit_transform(X_train, y_train)
            # run transform
            X_test = ch2.transform(X_test)
            print("done in %fs" % (time() - t0))
            
            if feature_names:
                # keep selected feature names
                indices =  ch2.get_support(indices=True)
                scores = [(feature_names[i], ch2.scores_[i]) for i in indices]

                #run pos filtering
                if opts.pos_filter:
                    pos_filter = load_filter(opts.pos_filter)
                    if pos_filter:
                        logger.info('run pos filtering on the model')
                        #colmask = [feature_names[i].encode('utf-8') in pos_filter for i in indices]
                        colmask = [feature_names[i] in pos_filter for i in indices]
                        #logger.info('colmask = %s', colmask)

                        # for idx in range(len(indices)):
                        #    if feature_names[indices[idx]] not in pos_filter:
                        #         X_test[:, idx] = 0
                        col_idx = X_test.indices
                        row_idx = X_test.indptr
                        for rowid in range(0, len(row_idx)-1):
                            for idx in range(row_idx[rowid] , row_idx[rowid + 1]):
                                if colmask[col_idx[idx]] == False:
                                    X_test[rowid, col_idx[idx]] = 0
 
                feature_names = [feature_names[i] for i in indices]
                scores = sorted(scores, key=lambda x:x[1], reverse=True)
                
               
                #save the chi2 features
                # with open('chi2_features','w') as chi2f:
                with open(self.name + '.chi2','w') as chi2f:
                    for sc in scores:
                        #if opts.use_tf or opts.use_word2vec:
                        if type(sc[0]) == unicode:
                            chi2f.write("%s\t%s\n"%(sc[0].encode('utf-8'), sc[1]))
                        else:
                            chi2f.write("%s\t%s\n"%(sc[0], sc[1]))

            print()
        
        if feature_names:
            feature_names = np.asarray(feature_names)

        #proxy
#self.X_train = scale(X_train)
#        self.X_test  = scale(X_test )

        self.X_train = X_train
        self.X_test  = X_test 
        self.y_train = y_train
        self.y_test  = y_test 
        self.X_id  = X_id 
        self.feature_names = feature_names
        self.categories =    categories  
        self.data_test = data_test
        self.data_train = data_train


# parse commandline arguments
def load_option():
    op = OptionParser()
    op.add_option("--report",
                  action="store_true", dest="print_report",
                  help="Print a detailed classification report.")
    op.add_option("--chi2_select",
                  action="store", type="int", dest="select_chi2",
                  help="Select some number of features using a chi-squared test")
    op.add_option("--no_confusion_matrix",
                  action="store_false", dest="print_cm",default=True,
                  help="Print the confusion matrix.")
    op.add_option("--no_top10",
                  action="store_false", dest="print_top10",default=True,
                  help="Print ten most discriminative terms per class"
                       " for every classifier.")
    op.add_option("--all_categories",
                  action="store_true", dest="all_categories",
                  help="Whether to use all categories or not.")
    op.add_option("--use_word2vec",
                  action="store", type=str, default="", dest="use_word2vec",
                  help="Use a word2vec vectorizer, otherwise a tfidf vectorizer. Provide the model file name")
    op.add_option("--use_word",
                  action="store_true",
                  help="Use a char/word based tokenizer.")
    op.add_option("--use_tf",
                  action="store_true",
                  help="Use a count vectorizer, otherwise a tfidf vectorizer.")
    op.add_option("--n_features",
                  action="store", type=int, default=1,
                  help="n_gram when using the count vectorizer.")
    op.add_option("--filtered",
                  action="store_true",
                  help="Remove newsgroup information that is easily overfit: "
                       "headers, signatures, and quoting.")
    op.add_option("--bin_class",
                  action="store", type="int", dest="bin_class",default=-999,
                  help="Build a binary classifier on this specific label.")
    op.add_option("--dataname",
                  action="store", type=str, default="jichu",
                  help="define the data name, which define the name of the categories.")
 
    op.add_option("--classifier",
                  action="store", type=str, default="lsvc",
                  help="define the classifier, lsvc as linearsvc, rf as random forest.")
    op.add_option("--pre_filter",
                  action="store", type=str, default="pre_filter.dict", dest="filterfile",
                  help="define the pre feature filter file name, by default pre_filter.dict.")
    op.add_option("--pos_filter",
                  action="store", type=str, default="pos_filter.dict", dest="pos_filter",
                  help="define the pos feature filter file name, by default pos_filter.dict.")
    op.add_option("--testmodel",
                  action="store", type=str, default="", dest="testmodel",
                  help="load a pre-trained model file, use it for later prediction and test.")
    op.add_option("--h",
                  action="store_true", dest="print_help",
                  help="Show help info.")
    
    (opts, args) = op.parse_args()
   
    #set default
    #opts.print_cm = True
    #opts.print_top10 =True
    
    opts.trainfile = 'train.cut'
    opts.testfile = 'test.cut'
    opts.expname = ''
    #opts.filterfile ='filter.dict'

    if len(args) > 1:
        opts.trainfile = args[0]
        opts.testfile = args[1]
        if len(args) > 2:
            opts.expname = args[2]
    
    if opts.print_help:
        print(__doc__)
        op.print_help()
        print()
        sys.exit(0)

    print('Start Classifier:[%s], features:[chi2:%s, vector:%s]'%( opts.classifier, opts.select_chi2, 'count' if opts.use_tf else 'tfidf'))


    return opts

def load_filter(fname):
    """
    filter dictionary is the features set
    when load the original dataset, it will 
    filter out all features not in this dictionary
    """
    filter = {}
    if not os.path.exists(fname):
        return filter
    with open(fname,'r') as f:
        for line in f:
            #w =  line.strip()
            w =  line.strip().split()[0] #get the first column
            if w in filter:
                continue
            filter[w] = 1
        print("load filter with %d features from %s\n"%(len(filter), fname))

    return filter

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

# Benchmark classifiers
def benchmark(dataset, opts, clf, name):
    """
        X_train, X_test, y_train, y_test,
        categories
        feature_names
    """
    X_train =dataset.X_train
    X_test  =dataset.X_test 
    y_train =dataset.y_train
    y_test  =dataset.y_test 
    X_id    =dataset.X_id
    feature_names =dataset.feature_names
    categories =   dataset.categories  
    data_test = dataset.data_test
    name += '-' + dataset.name

    #===========
    print(name)
    print('_' * 80)

    if opts.testmodel:
        #load the model
        print("Load pretrained model: %s"%opts.testmodel)
        clf = joblib.load(opts.testmodel + '.pkl')
        print(clf)
        train_time = 0

    else:
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        #save the model
        joblib.dump(clf, name + '.pkl')

    # now, it's tim run pos-filtering
    # filter out all the feature columns in X_test that not appear in the pos filter dict

    # coef_ or feature_importances_ are readonly parameters for clf, so it's not easy to 
    # modify the models, try pos-filter on X_test instead. 01112017
    #if opts.pos_filter:
    #    pos_filter = load_filter(opts.pos_filter)
    #    if pos_filter:
    #        logger.info('run pos filtering on the model')
    #        


    # start test
    t0 = time()
    pred = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    #save the result
    with open(name + '.res','w') as outf:
        for idx in range(len(X_id)):
            # todo: temporary test
            # outf.write('%s %s %s\n'%(y_test[idx]-1, pred[idx]-1, X_id[idx]))
            # write the data directly, for future feature engineering need.
            outf.write('%s %s %s\n'%(y_test[idx], pred[idx], data_test.data[idx]))

    #save the full res
    #label, predict, [proba for all classes]
    with open(name + '.prob','w') as outf:
        for idx in range(len(X_id)):
            outf.write('%s %s %s\n'%(y_test[idx], pred[idx], " ".join(['%.04f'%p for p in pred_prob[idx]])))

    precision = metrics.precision_score(y_test, pred, average=None)
    recall = metrics.recall_score(y_test, pred, average=None)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    print("precision:   %s" % precision)
    print("recall:   %s" % recall)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                print(clf.coef_.shape)
                if clf.coef_.shape[0] <= i:
                    continue
                top10 = np.argsort(clf.coef_[i])[-10:]
#print(trim("%s: %s"
#                      % (category, " ".join(feature_names[top10]))).encode('utf-8'))

                print(trim("%s: %s"
                    % (category, " ".join(feature_names[top10]))))
        print()

        # save coef by features names
        print(' '.join(categories))
        with open(name + '.coef', 'w') as cof:
            for i, category in enumerate(categories):
                if clf.coef_.shape[0] <= i:
                    continue
                _all = np.argsort(clf.coef_[i])
                for _id in range(clf.coef_[i].shape[0]):
                    vid = _all[-_id]
                    #if clf.coef_[i][vid] > 0:
                    #cof.write("%s\t%s\t%s\n"%(category, feature_names[vid].encode('utf-8'), clf.coef_[i][vid]))
                    #cof.write("%s\t%s\t%s\n"%(category, feature_names[vid] if feature_names is not None else vid, clf.coef_[i][vid]))

                    #if opts.use_tf or opts.use_word2vec:
                    if type(feature_names[vid]) == unicode:
                        cof.write("%s\t%s\t%s\n"%(category, feature_names[vid].encode('utf-8') if feature_names is not None else vid, clf.coef_[i][vid]))
                    else:
                        cof.write("%s\t%s\t%s\n"%(category, feature_names[vid] if feature_names is not None else vid, clf.coef_[i][vid]))

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        cmstr = metrics.confusion_matrix(y_test, pred)
        print(cmstr)
        with open(name + '.cm', 'w') as cmf:
            cmf.write('%s\n'%name)
            cmf.write('%s\n'%(clf))
            cmf.write('%s\n'%(cmstr))
            cmf.write("precision:   %s\n" % precision)
            cmf.write("recall:   %s\n\n" % recall)



    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

def run_classifer(dataset, opts):
    #stack the data
    clfset = []

    if opts.testmodel:
        #test model , just skip the classifier settings
        clfset.append(('TestModel',LinearSVC(penalty='l1',dual=False, tol=1e-3 )))

    elif opts.classifier == "lsvc":
        #clfset.append(('LinearSVC l1 penality' ,LinearSVC(loss='l2', penalty='l1',dual=False, tol=1e-3 )))
        #clfset.append(('LinearSVC l1 penality' ,LinearSVC(loss='l2', penalty='l1',dual=False, tol=1e-3, class_weight='balanced')))
        clfset.append(('LinearSVC_l1' ,LinearSVC(penalty='l1',dual=False, tol=1e-3 )))
        clfset.append(('LinearSVC_l1_balanced' ,LinearSVC(penalty='l1',dual=False, tol=1e-3, class_weight='balanced')))
    elif opts.classifier == "lsvcl2":
        clfset.append(('LinearSVC_l2' ,LinearSVC( penalty='l2', tol=1e-4 )))
        clfset.append(('LinearSVC_l2_balanced' ,LinearSVC(penalty='l2', tol=1e-4, class_weight='balanced')))
    elif opts.classifier == 'rf':
        clfset.append(("RF",RandomForestClassifier(n_estimators=500, n_jobs=4, min_samples_split=1)))
        #clfset.append(("Random forest balanced subsample",RandomForestClassifier(n_estimators=100, n_jobs=4,oob_score=True, min_samples_split=1,class_weight='balanced_subsample')))
        clfset.append(("RF_balanced",RandomForestClassifier(n_estimators=100, n_jobs=4,criterion='entropy', min_samples_split=1,class_weight='balanced_subsample')))
        #clfset.append(("Random forest balanced",RandomForestClassifier(n_estimators=100, n_jobs=4, min_samples_split=1,class_weight='balanced')))
    elif opts.classifier == 'lr':
        clfset.append(("LR",LogisticRegression() ))
        clfset.append(("LR_Balanced",LogisticRegression(class_weight = 'balanced')))
    elif opts.classifier == 'lrl1':
        #clfset.append(("LR",LogisticRegression() ))
        clfset.append(("LR_Balanced",LogisticRegression(class_weight = 'balanced', penalty='l1')))
 
        # clfset.append(("LR_Balanced",LogisticRegression(class_weight = 'balanced'), penalty='l1'))
        #clfset.append(('LinearSVC l1 penality' ,LinearSVC(C=100, penalty='l1',dual=False, tol=1e-3, class_weight='balanced')))
        #clfset.append(('LinearSVC l1 penality' ,LinearSVC(C=0.1, penalty='l1',dual=False, tol=1e-3, class_weight='balanced')))
        #clfset.append(('SGD l1 penality' , SGDClassifier(alpha=.0001, n_iter=50, penalty='l1',class_weight='balanced')))
        #clfset.append(('Ridge Classifier', RidgeClassifier(tol=1e-2, solver="lsqr")))
        #clfset.append(('SVM Classifier(rbf)', SVC(C=0.1, tol=1e-3, kernel="sigmoid", coef0=0.1)))
        #clfset.append(('SVM Classifier(poly)', SVC(tol=1e-3, kernel="poly")))
        
        #clfset.append(("Random forest",RandomForestClassifier(n_estimators=100)))
        #clfset.append(("BernoulliNB", BernoulliNB(alpha=.01)))
        #clfset.append(("linearSVC pipeline", Pipeline([
        #  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
        #  ('classification', LinearSVC())
        #])))
 
    # run it
    for name, clf in clfset:
        benchmark(dataset, opts, clf, name)

if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    opts = load_option()
    dataset = DataSet()
    dataset.preprocess(opts)
    run_classifer(dataset,opts)

