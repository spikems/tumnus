#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Learner Class for Classification of text documents using sparse features
"""

from __future__ import print_function

import logging
import numpy as np
import sys, os
from time import time
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel
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
#from sklearn.externals import joblib

logger = logging.getLogger(__name__)
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

class Learner():
    """
    Learner class
    1. load dataset 
    2. init vectorizer, featureselector
    3. run transform
    4. train and save model
    5. load model and predict
    """
    def __init__(self, train = True, name = ''):
        # trainer or predictor
        self.train = train
        # name of this learner
        self.name = ''

        # dataset
        # raw data and preprocessed data, and index as x_id
        # dataset = Bunch(data=data, target=target, ids=ids, target_names= category_names)
        self.dataset = None

        # transformed into (x,y) vectors
        self.x = None
        self.y = None
        # feature names, category names
        self.feature_names = None

        # model
        self.vectorizer = None
        self.vocabulary = None
        self.featureselector = None
        self.clf = None

    def copy_model(self,learner):
        """
            Copy model from other learner
        """
        self.vectorizer = learner.vectorizer
        self.vocabulary = learner.vocabulary
        self.featureselector = learner.featureselector
        self.clf = learner.clf

    def __str__(self):
        """
            Standard string
        """
        info = str(self.vectorizer) + '\n'
        info += str(self.featureselector)
        return info

    def init_vectorizer(self, type='tfidf', ngram_range=(1,1), max_df=0.9):
        """
        Refer to the api document of TfidfVectorizer and CountVectorizer
        We fixed on using tfidf or binary count vectorizer

        customized vectorizer:
            template    ; expand by template definitions(todo)
 
        """
        if type == 'tfidf':
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=max_df, ngram_range=ngram_range, vocabulary = self.vocabulary, token_pattern=r'[^ ]+')
#            self.vectorizer = TfidfVectorizer(analyzer=str.split, sublinear_tf=True, max_df=max_df, ngram_range=ngram_range, vocabulary = self.vocabulary)
        elif type =='count':
            self.vectorizer = CountVectorizer(binary = True, max_df=max_df, ngram_range=ngram_range, vocabulary = self.vocabulary, token_pattern=r'[^ ]+')
#            self.vectorizer = CountVectorizer(analyzer=str.split,binary = True, max_df=max_df, ngram_range=ngram_range, vocabulary = self.vocabulary)
        elif type == 'template':
            #customized vectorizer using tempalte
            # todo
            self.vectorizer = None

        return self.vectorizer

    def init_featureselector(self, type = 'chi2', selectCnt=500):
        """
        Feature selection 

        """
        if type =='chi2':
            self.featureselector = SelectKBest(chi2, k=selectCnt)
        elif type == 'l1':
            #placeholder
            self.featureselector = 'SelectFromModel'
        else:
            self.featureselector = None

        return self.featureselector

    def init_vocabulary(self, fname):
        """
        Load Vocabulary from a file, one word each line.
        This was materialized by mannual feature selection.
        Return an iterable over terms. If not given, a vocabulary is determined from the input documents. 
        """
        #check first
        if not (fname and os.path.exists(fname)):
            self.vocabulary = None
            return None

        vocab = []
        with open(fname, 'r') as inf:
            for line in inf:
                # raw presentation can be different to feature's
                # such as a 2-gram example
                # 不 需要   --> 　不需要
                # TODO: save the mapping of raw presentation from the vocabulary file, which will be used for future annotation step
                # bugfix, convert to unicode because vectorizer.analyzer output unicode by default
                vocab.append(line.strip().replace(' ','').decode('utf-8'))
            self.vocabulary = vocab
        logger.info('init vocabulary, size = %d', len(self.vocabulary))
        return self.vocabulary

    def load_dataset(self, fname, category_names = [], recordid = True,  bin_class = -999):
        """
        Load and return the dataset (classification).
            -1 23456 content...
        
        Input:
            category_names  ; name of the labels
            recordid    ; use record as id if true, otherwise use item[1] in line as id
            bin_class    ;if bin_class set, then setup a binary classifier for this special label(one-vs-rest)
    
        """
        data = []
        target = []
        ids = []
        recid = 0
        with open(fname, 'r') as inf:
            for line in inf:
                line = line.strip()
    
                pos1 = line.find(' ')
                #put the id into data, extract into X_id in the future
                pos2 = line.find(' ', pos1+1)
                #pos2 = pos1
                if recordid == True:
                    ids.append(str(recid))
                    recid += 1
                else:
                    ids.append(line[pos1+1:pos2])
    
                words = line[pos2+1:]
                data.append(words)
                #if bin_class != -999 and bin_class < len(categories):
                if bin_class != -999:
                    target.append(0 if int(line[:pos1]) == bin_class else 1)
                else:
                    target.append(int(line[:pos1]))
    
        # set the category_names
        targetCnt = len(set(target))
        if not category_names:
            # by default, use the label value as target_names
            # label start from 0
            category_names = [ str(x) for x in range(targetCnt)]

        # validate the dataset, 
        if category_names and targetCnt > len(category_names):
            #warning, label mismatch
            logger.warn('dataset %s loading labels mismatch to the category_names', fname, category_names)
        else:
            def size_mb(docs):
                return sum(len(s) for s in docs) / 1e6
            data_train_size_mb = size_mb(data)
            logger.info("loading dataset with %d documents - %0.3fMB, %d categories.",  len(data), data_train_size_mb, len(category_names))

        self.dataset = Bunch(data=data, target=target, ids=ids, target_names= category_names)

        return self.dataset

    def transform(self):
        """
        Transform the loaded dataset into standard vectors
        Input:
            self.dataset    ;
            self.vectorizer ;
        Output:
            self.x, .y
            self.feature_names
        """
        if self.vectorizer is None:
            logger.info("no vectorizer set, skip transform.")
            return

        # set x
        logger.info("Extracting features from the dataset using a sparse vectorizer")
        t0 = time()
        if self.train:
            self.x = self.vectorizer.fit_transform(self.dataset.data)
        else:
            self.x = self.vectorizer.transform(self.dataset.data)

        duration = time() - t0
        logger.info("dataset n_samples: %d, n_features: %d" % self.x.shape)
        
        # mapping from integer feature name to original token string
        feature_names = self.vectorizer.get_feature_names()
        self.feature_names = np.asarray(feature_names)
        logger.info("feature_names shape=%s"%len(self.feature_names))

        # set target
        self.y = self.dataset.target


    def feature_select(self, savename = ''):
        """
        Run feature selection
        Input:
            self.featureselector
        """
        # deal with feature_names
        feature_names = self.feature_names
 
        if self.featureselector is None:
            logger.info("no feature selector set, skip it.")
        else:
            logger.info("Extracting best features by a feature selector")
            t0 = time()
            if self.train:
                if type(self.featureselector) == str:
                    # SelectFromModel
                    _lr = LogisticRegression(C=0.2,class_weight='balanced', penalty='l1').fit(self.x, self.y)
                    self.featureselector = SelectFromModel(_lr, prefit=True)

                    self.x = self.featureselector.transform(self.x)
                else:
                    self.x = self.featureselector.fit_transform(self.x, self.y)
            else:    # run transform
                self.x = self.featureselector.transform(self.x)
            logger.info("done in %fs" % (time() - t0))
            
            if feature_names is not None:
                # keep selected feature names
                indices = self.featureselector.get_support(indices=True)
                if hasattr(self.featureselector, 'scores_'):
                    scores = [(feature_names[i], self.featureselector.scores_[i]) for i in indices]
                else:
                    scores = [(feature_names[i], i) for i in indices]

                feature_names = [feature_names[i] for i in indices]
                scores = sorted(scores, key=lambda x:x[1], reverse=True)
               
                #save the chi2 features
                if savename:
                    with open(savename + '.chi2','w') as chi2f:
                        for sc in scores:
                            #if opts.use_tf or opts.use_word2vec:
                            if type(sc[0]) == np.unicode_ or type(sc[0]) == unicode:
                                chi2f.write("%s\t%s\n"%(sc[0].encode('utf-8'), sc[1]))
                            else:
                                chi2f.write("%s\t%s\n"%(sc[0], sc[1]))

        # convert feature_names into array
        if feature_names is not None:
            self.feature_names = np.asarray(feature_names)

        logger.info("feature_names shape=%s"%len(self.feature_names))


    def train_model(self, clf, savename):
        """
        Train model by a classifier
        """
        logger.info("Training: ")
        clfinfo = str(clf).replace('\n','')
        logger.info(clfinfo)
        t0 = time()
        clf.fit(self.x, self.y)
        train_time = time() - t0
        logger.info("train time: %0.3fs" % train_time)
    
        #save the model
        if savename:
            with open(savename + '.pkl', 'wb') as fout:

                # print(self.vectorizer)
                #self.vectorizer.analyzer = 'word'
                pickle.dump((self.vocabulary, self.vectorizer, self.featureselector, clf), fout)
                #self.vectorizer.analyzer = str.split

        #save clf
        self.clf = clf

        return clf

    def load_model(self, modelname):
        """
        Load in the saved model files
        """
        with open(modelname + '.pkl', 'rb') as fin: 
            self.vocabulary, self.vectorizer, self.featureselector, self.clf = pickle.load(fin)
            #self.vectorizer.analyzer = str.split
            return self.clf
        return None

    def predict(self, clf = None, savename = 'demo-predict'):
        """
        Predict by a classifier
        Return:
            return y_test, pred, pred_prob
        """
        if clf is None:
            # use the loaded clf
            clf = self.clf

        #check validation
        if self.x is None or self.dataset is None or clf is None:
            logger.info("data not ready yet, skip prediction, quit.")
            return

        X_test  = self.x
        y_test  = self.y
        X_id    = self.dataset.ids
        feature_names = self.feature_names
        categories = self.dataset.target_names
        data_test = self.dataset
        name = savename
    
        # start test
        t0 = time()
        pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)
        test_time = time() - t0
        logger.info("Predict test time:  %0.3fs" % test_time)
    
        #save the result
        with open(name + '.res','w') as outf:
            for idx in range(len(X_id)):
                # write the data directly, for future feature engineering need.
                # add xid into output, useful for predict
                outf.write('%s %s %s %s\n'%(y_test[idx], pred[idx], 
                            X_id[idx], data_test.data[idx]))
    
        #save the full res
        #label, predict, [proba for all classes]
        with open(name + '.prob','w') as outf:
            for idx in range(len(X_id)):
                outf.write('%s %s %s %s\n' % (X_id[idx], y_test[idx], pred[idx], " ".join(['%.04f'%p for p in pred_prob[idx]])))
    
        precision = metrics.precision_score(y_test, pred, average=None)
        recall = metrics.recall_score(y_test, pred, average=None)
        score = metrics.accuracy_score(y_test, pred)
        f1_score = metrics.f1_score(y_test, pred, average=None)
        logger.info("accuracy:   %0.3f" % score)
        logger.info("precision:   %s" % precision)
        logger.info("recall:   %s" % recall)
        logger.info("f1 score:   %s" % f1_score) 
    
        if hasattr(clf, 'coef_'):
            logger.info("dimensionality: %d" % clf.coef_.shape[1])
            logger.info("density: %f" % density(clf.coef_))
    
            if feature_names is not None:
                logger.info("top 10 keywords per class:")
                for i, category in enumerate(categories):
                    logger.info(clf.coef_.shape)
                    if clf.coef_.shape[0] <= i:
                        continue
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    logger.info("%s: %s", category, " ".join(feature_names[top10]))
    
            # save coef by features names
            logger.info(' '.join(categories))
            with open(name + '.coef', 'w') as cof:
                for i, category in enumerate(categories):
                    if clf.coef_.shape[0] <= i:
                        continue
                    _all = np.argsort(clf.coef_[i])
                    for _id in range(clf.coef_[i].shape[0]):
                        vid = _all[-_id]
                        #if opts.use_tf or opts.use_word2vec:
                        if type(feature_names[vid]) in [np.unicode_, unicode]:
                            cof.write("%s\t%s\t%s\n"%(category, feature_names[vid].encode('utf-8') if feature_names is not None else vid, clf.coef_[i][vid]))
                        else:
                            cof.write("%s\t%s\t%s\n"%(category, feature_names[vid] if feature_names is not None else vid, clf.coef_[i][vid]))
    
        logger.info("confusion matrix:")
        cmstr = metrics.confusion_matrix(y_test, pred)
        [logger.info(x) for x in cmstr]
        with open(name + '.cm', 'w') as cmf:
            cmf.write('%s\n'%name)
            cmf.write('%s\n'%(clf))
            cmf.write('%s\n'%(cmstr))
            cmf.write("precision:   %s\n" % precision)
            cmf.write("recall:   %s\n\n" % recall)
            cmf.write("f1 score:   %s\n\n" % f1_score)

        #return the result, for future evaluations
        return y_test, pred, pred_prob

if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) != 3:
        logger.info('')
        sys.exit(0)

    trainfile = sys.argv[1]
    testfile = sys.argv[2]

    logger.info('demo learner on %s, %s', trainfile, testfile)

    # this is a demo
    trainer = Learner()
    trainer.load_dataset(trainfile)
   #trainer.init_vocabulary('pre_filter.dict')
    trainer.init_vocabulary('feature.dict')
    #trainer.init_vectorizer(type = 'tfidf')
    trainer.init_vectorizer(type = 'count')
    #trainer.init_featureselector(type = 'chi2', selectCnt = 200)
    trainer.transform()
    trainer.feature_select(savename = 'demo-train')

    #clf = LogisticRegression(class_weight = 'balanced', penalty='l1')
    clf = LogisticRegression(class_weight = 'balanced', penalty='l2')
    trainer.train_model(clf, 'demo-model')

    predictor = Learner(train = False)
    predictor.load_model('demo-model')
    
    predictor.load_dataset(testfile)
    predictor.transform()
    #predictor.feature_select()
    predictor.predict(savename = 'demo-predict')

