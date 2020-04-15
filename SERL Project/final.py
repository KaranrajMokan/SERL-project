#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:57:17 2020

@author: karanrajmokan
"""


import warnings
warnings.filterwarnings("ignore")

import pandas
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy.io import arff

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.metrics import label_ranking_loss, coverage_error, zero_one_loss
from sklearn.metrics import f1_score



class SoftmaxRegression(object):

    def __init__(self, eta=0.01, epochs=50, l2=0.0, minibatches=1, n_classes=None, random_seed=None):

        self.eta = eta
        self.epochs = epochs
        self.l2 = l2
        self.minibatches = minibatches
        self.n_classes = n_classes
        self.random_seed = random_seed

    def _fit(self, X, y, init_params=True):
        if init_params:
            if self.n_classes is None:
                self.n_classes = np.max(y) + 1
            self._n_features = X.shape[1]

            self.b_, self.w_ = self._init_params(
                weights_shape=(self._n_features, self.n_classes),
                bias_shape=(self.n_classes,),
                random_seed=self.random_seed)
            self.cost_ = []

        y_enc = self._one_hot(y=y, n_labels=self.n_classes, dtype=np.float)

        for i in range(self.epochs):
            for idx in self._yield_minibatches_idx(
                    n_batches=self.minibatches,
                    data_ary=y,
                    shuffle=True):

                net = self._net_input(X[idx], self.w_, self.b_)
                softm = self._softmax(net)
                diff = softm - y_enc[idx]
                grad = np.dot(X[idx].T, diff)
                
                self.w_ -= (self.eta * grad + self.eta * self.l2 * self.w_)
                self.b_ -= (self.eta * np.sum(diff, axis=0))

            net = self._net_input(X, self.w_, self.b_)
            softm = self._softmax(net)
            cross_ent = self._cross_entropy(output=softm, y_target=y_enc)
            cost = self._cost(cross_ent)
            self.cost_.append(cost)
        return self

    def fit(self, X, y, init_params=True):

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self._fit(X=X, y=y, init_params=init_params)
        self._is_fitted = True
        return self
    
    def _predict(self, X):
        probas = self.predict_proba(X)
        return self._to_classlabels(probas)
 
    def predict(self, X):

        if not self._is_fitted:
            raise AttributeError('Model is not fitted, yet.')
        return self._predict(X)

    def predict_proba(self, X):
 
        net = self._net_input(X, self.w_, self.b_)
        softm = self._softmax(net)
        return softm

    def _net_input(self, X, W, b):
        return (X.dot(W) + b)

    def _softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def _cross_entropy(self, output, y_target):
        return - np.sum(np.log(output) * (y_target), axis=1)

    def _cost(self, cross_entropy):
        L2_term = self.l2 * np.sum(self.w_ ** 2)
        cross_entropy = cross_entropy + L2_term
        return 0.5 * np.mean(cross_entropy)

    def _to_classlabels(self, z):
        return z.argmax(axis=1)
    
    def _init_params(self, weights_shape, bias_shape=(1,), dtype='float64',
                     scale=0.01, random_seed=None):
        """Initialize weight coefficients."""
        if random_seed:
            np.random.seed(random_seed)
        w = np.random.normal(loc=0.0, scale=scale, size=weights_shape)
        b = np.zeros(shape=bias_shape)
        return b.astype(dtype), w.astype(dtype)
    
    def _one_hot(self, y, n_labels, dtype):

        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(dtype)    
    
    def _yield_minibatches_idx(self, n_batches, data_ary, shuffle=True):
            indices = np.arange(data_ary.shape[0])

            if shuffle:
                indices = np.random.permutation(indices)
            if n_batches > 1:
                remainder = data_ary.shape[0] % n_batches

                if remainder:
                    minis = np.array_split(indices[:-remainder], n_batches)
                    minis[-1] = np.concatenate((minis[-1],
                                                indices[-remainder:]),
                                               axis=0)
                else:
                    minis = np.array_split(indices, n_batches)

            else:
                minis = (indices,)

            for idx_batch in minis:
                yield idx_batch
    
    def _shuffle_arrays(self, arrays):

        r = np.random.permutation(len(arrays[0]))
        return [ary[r] for ary in arrays]
    
    


def smaxregression(x,y,X_test,Y_test):

    sc = SoftmaxRegression()
    sc.fit(x,y,init_params=True)
    predictions_softmax = sc.predict(np.array(X_test))
    que=np.array(predictions_softmax)
    ans=np.array(Y_test)
    ans = ans.ravel()

    count=0
    for i in range(len(que)):
        if que[i].any() == ans[i].any():
            count+=1

    accuracy=count/len(que)
    return accuracy


def classifiers(X_train,Y_train,X_test):
    
    classifier1 = BinaryRelevance(GaussianNB())
    classifier2 = ClassifierChain(GaussianNB())
    classifier3 = LabelPowerset(GaussianNB())

    classifier1.fit(X_train, Y_train)
    classifier2.fit(X_train, Y_train)
    classifier3.fit(X_train, Y_train)

    predictions1 = classifier1.predict(X_test)
    predictions2 = classifier2.predict(X_test)
    predictions3 = classifier3.predict(X_test)
    
    return predictions1,predictions2,predictions3


def avg_precision_scores(arr,arr1,arr2,arr3):
    
    a1 = average_precision_score(arr,arr1)
    b1 = average_precision_score(arr,arr2)
    c1 = average_precision_score(arr,arr3)

    print("Average Precisions Scores for the three classifiers are")
    print("Using Binary Relevance: " + str(a1))
    print("Using Classifier Chain: " + str(b1))
    print("Using LabelPowerSet: " + str(c1))
    print("\n")
    
def roc_auc_scores(arr,arr1,arr2,arr3):
    
    try:
        a2 = roc_auc_score(arr,arr1)
        b2 = roc_auc_score(arr,arr2)
        c2 = roc_auc_score(arr,arr3)
    except ValueError:
        a2 = 'nan'
        b2 = 'nan'
        c2 = 'nan'

    print("MacroAUC Scores for the three classifiers are")
    print("Using Binary Relevance: " + str(a2))
    print("Using Classifier Chain: " + str(b2))
    print("Using LabelPowerSet: " + str(c2))
    print("\n")    
  
def label_rank_loss(arr,arr1,arr2,arr3):

    a3 = label_ranking_loss(arr, arr1)
    b3 = label_ranking_loss(arr, arr2)
    c3 = label_ranking_loss(arr, arr3)

    print("Ranking Loss Scores for the three classifiers are")
    print("Using Binary Relevance: " + str(a3))
    print("Using Classifier Chain: " + str(b3))
    print("Using LabelPowerSet: " + str(c3))
    print("\n")
    
def coverage_error_scores(arr,arr1,arr2,arr3):
    
    a4 = coverage_error(arr, arr1)
    b4 = coverage_error(arr, arr2)
    c4 = coverage_error(arr, arr3)

    print("Coverage Error Scores for the three classifiers are")
    print("Using Binary Relevance: " + str(a4))
    print("Using Classifier Chain: " + str(b4))
    print("Using LabelPowerSet: " + str(c4))
    print("\n")    
    
def zero_one_loss_score(arr,arr1,arr2,arr3):
    
    a5 = zero_one_loss(arr, arr1)
    b5 = zero_one_loss(arr, arr2)
    c5 = zero_one_loss(arr, arr3)

    print("Zero One Loss Scores for the three classifiers are")
    print("Using Binary Relevance: " + str(a5))
    print("Using Classifier Chain: " + str(b5))
    print("Using LabelPowerSet: " + str(c5))
    print("\n")    

def accuracy_scores(Y_test,predictions1,predictions2,predictions3,accuracy):

    a = accuracy_score(Y_test,predictions1)
    b = accuracy_score(Y_test,predictions2)
    c = accuracy_score(Y_test,predictions3)

    print("Accuracy Scores for the three classifiers are")
    print("Using Binary Relevance: " + str(a))
    print("Using Classifier Chain: " + str(b))
    print("Using LabelPowerSet: " + str(c))
    print("Using Softmax Regression " + str(accuracy))
    print("\n")

    
def f1_score_micro(Y_test,predictions1,predictions2,predictions3):
    
    a6 = f1_score(Y_test, predictions1,average='micro')
    b6 = f1_score(Y_test, predictions2,average='micro')
    c6 = f1_score(Y_test, predictions3,average='micro')

    print("F1 Scores(micro) for the three classifiers are")
    print("Using Binary Relevance: " + str(a6))
    print("Using Classifier Chain: " + str(b6))
    print("Using LabelPowerSet: " + str(c6))
    print("\n")  
    
def f1_score_macro(Y_test,predictions1,predictions2,predictions3):

    a7 = f1_score(Y_test, predictions1,average='macro')
    b7 = f1_score(Y_test, predictions2,average='macro')
    c7 = f1_score(Y_test, predictions3,average='macro')

    print("MacroF1 Scores for the three classifiers are")
    print("Using Binary Relevance: " + str(a7))
    print("Using Classifier Chain: " + str(b7))
    print("Using LabelPowerSet: " + str(c7))
    print("\n")    


def commonfunc(X_train,Y_train,X_test,Y_test):

    x = np.array(X_train)
    y = np.array(Y_train)
    
    accuracy = smaxregression(x,y,X_test,Y_test)
    predictions1, predictions2, predictions3 = classifiers(X_train,Y_train,X_test)
    
    arr = np.array(Y_test)
    arr1 = predictions1.toarray()
    arr2 = predictions2.toarray()
    arr3 = predictions3.toarray()
    
    print("RANKING EVALUATION METRICS:")
    avg_precision_scores(arr, arr1, arr2, arr3)
    roc_auc_scores(arr, arr1, arr2, arr3)
    label_rank_loss(arr, arr1, arr2, arr3)
    coverage_error_scores(arr, arr1, arr2, arr3)
    zero_one_loss_score(arr, arr1, arr2, arr3)

    print("CLASSIFICATION EVALUATION METRICS:")
    accuracy_scores(Y_test, predictions1, predictions2, predictions3, accuracy)
    f1_score_micro(Y_test, predictions1, predictions2, predictions3)
    f1_score_macro(Y_test, predictions1, predictions2, predictions3)    
    
def yeast():

    data, meta = arff.loadarff('yeast-train.arff')
    df = pandas.DataFrame(data)
    df = df.replace(b'0',int(0))
    df = df.replace(b'1',int(1))

    Y_train = df.iloc[:,103:117]
    df.drop(df.iloc[:,103:117],inplace=True,axis=1)
    X_train = df

    data1, meta1 = arff.loadarff('yeast-test.arff')
    df1 = pandas.DataFrame(data1)
    df1 = df1.replace(b'0',int(0))
    df1 = df1.replace(b'1',int(1))

    Y_test = df1.iloc[:,103:117]
    df1.drop(df1.iloc[:,103:117],inplace=True,axis=1)
    X_test = df1
    
    print("FOR YEAST DATASET:")
    commonfunc(X_train, Y_train, X_test, Y_test)

    

def corel16k01():

    data, meta = arff.loadarff('corel16k01-train.arff')
    df = pandas.DataFrame(data)
    df = df.replace(b'0',int(0))
    df = df.replace(b'1',int(1))

    Y_train = df.iloc[:,500:653]
    df.drop(df.iloc[:,500:653],inplace=True,axis=1)
    X_train = df

    data1, meta1 = arff.loadarff('corel16k01-test.arff')
    df1 = pandas.DataFrame(data1)
    df1 = df1.replace(b'0',int(0))
    df1 = df1.replace(b'1',int(1))

    Y_test = df1.iloc[:,500:653]
    df1.drop(df1.iloc[:,500:653],inplace=True,axis=1)
    X_test = df1

    print("FOR Corel16k(Sample-01) DATASET:")
    commonfunc(X_train, Y_train, X_test, Y_test)



def corel16k02():
    
    data, meta = arff.loadarff('corel16k02-train.arff')
    df = pandas.DataFrame(data)
    df = df.replace(b'0',int(0))
    df = df.replace(b'1',int(1))

    Y_train = df.iloc[:,500:653]
    df.drop(df.iloc[:,500:653],inplace=True,axis=1)
    X_train = df

    data1, meta1 = arff.loadarff('corel16k02-test.arff')
    df1 = pandas.DataFrame(data1)
    df1 = df1.replace(b'0',int(0))
    df1 = df1.replace(b'1',int(1))

    Y_test = df1.iloc[:,500:653]
    df1.drop(df1.iloc[:,500:653],inplace=True,axis=1)
    X_test = df1

    print("FOR Corel16k(Sample-02) DATASET:")
    commonfunc(X_train, Y_train, X_test, Y_test)



def corel5k():

    data, meta = arff.loadarff('corel5k-train.arff')
    df = pandas.DataFrame(data)
    df = df.replace(b'0',int(0))
    df = df.replace(b'1',int(1))

    Y_train = df.iloc[:,499:873]
    df.drop(df.iloc[:,499:873],inplace=True,axis=1)
    X_train = df

    data1, meta1 = arff.loadarff('corel5k-test.arff')
    df1 = pandas.DataFrame(data1)
    df1 = df1.replace(b'0',int(0))
    df1 = df1.replace(b'1',int(1))

    Y_test = df1.iloc[:,499:873]
    df1.drop(df1.iloc[:,499:873],inplace=True,axis=1)
    X_test = df1

    print("FOR Corel5k DATASET:")
    commonfunc(X_train, Y_train, X_test, Y_test)


    
if __name__=="__main__":
    yeast()
    corel16k01()
    corel16k02()
    corel5k()

    
    