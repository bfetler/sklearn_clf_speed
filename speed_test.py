# -*- coding: utf-8 -*-

import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation
import os

# don't really need app here, just for tests, use in importing scripts
def get_app_title():
    "get app title"
    return 'SVM'

def get_app_file():
    "get app file prefix"
    return 'svm_'

def get_datadir():
    return "data/UCI_HAR_Dataset/"

def get_plotdir():
    "get plot directory"
    return 'speed_plots/'

def make_plotdir():
    "make plot directory on file system"
    plotdir = get_plotdir()  # add plotdir arg
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

# dfcol from features.txt, but not needed, use x1-x561
def read_train_data(datadir, printOut=False):
    "read in raw train data"
    traindir = datadir + "train/"
    dftrain = pd.read_table(traindir + "X_train.txt", \
        sep='\s+', header=None)    # names=dfcol['label2']
    
# add column names x0-x560
    d1 = dftrain.shape[1]
    dftrain.columns = map(lambda k: "x" + str(k), range(d1))
    
    # drop last two columns
#    dftrain.drop(["x"+str(d1-2), "x"+str(d1-1)], axis=1, inplace=True)
    
    # drop n columns so that 512 remain  # or 544 remain
    cols = map(lambda k: "x" + str(k), range(543, d1))
    dftrain.drop(cols, axis=1, inplace=True)
    
# drop last 152 rows, shape now 7200 by 560 (divisible by 2, 4, 8)
    dftrain.drop(range(7200, dftrain.shape[0]), inplace=True)
    
    dftrain['subject'] = pd.read_table(traindir + "subject_train.txt", \
        sep='\s+', header=None)
    dftrain_y = pd.read_table(traindir + "y_train.txt", \
        sep='\s+', header=None, names=['Y'])
# drop last 152 rows, add TF column for bivariate classification
    dftrain_y.drop(range(7200, dftrain_y.shape[0]), inplace=True)
    dftrain_y['TF'] = dftrain_y['Y'].apply(lambda x: 0 if x<4 else 1)
    
    if printOut==True:
        print("dftrain shape", dftrain.shape, "head\n", dftrain[:3])
        print("dftrain_y shape", dftrain_y.shape, "head\n", 
              dftrain_y[:3], "\n", dftrain_y[294:305])
        print("dftrain tail\n", dftrain[-3:])
        print("dftrain_y tail\n", dftrain_y[-3:])
        print("dftrain_y describe \n%s" % dftrain_y.describe())
    return dftrain, dftrain_y

def get_partial_data(dfbig_X, dfbig_y, rowf=4, colf=4, printOut=False):
    '''Get data frames of size reduced by row fraction 1/rowf, column fraction 1/colf.
       May apply to train or test data.'''
# Don't need to randomize, main purpose is speed tests on reproducible datasets.
    
    oldrow = dfbig_X.shape[0]
    oldcol = dfbig_X.shape[1]
    newrow = (oldrow+1) // rowf
    newcol = (oldcol+1) // colf
#    print("newrow %d oldrow %d, newcol %d oldcol %d" % (newrow, oldrow, newcol, oldcol))
    
    cols = map(lambda k: "x" + str(k-1), range(newcol, oldcol))
#    print("cols", cols)
    
    dfx = dfbig_X.drop(cols, axis=1)
    dfx.drop(range(newrow, oldrow), inplace=True)  # inplace=True fails if =
    dfy = dfbig_y.drop(range(newrow, oldrow))
    
    if printOut==True:
        print("dfx shape", dfx.shape, "head\n", dfx[:3])
        print("dfy shape", dfy.shape, "head\n", dfy[:3])
        print("dfx tail\n", dfx[-3:])
        print("dfy tail\n", dfy[-3:])
        print("dfy describe \n%s" % dfy.describe())
    
    return dfx, dfy

def do_fit(clf, dfx, dfy, print_out=False):
    "fit train data"
    clf.fit(dfx, dfy)
    fit_score = clf.score(dfx, dfy)
    if print_out:
        print("params", clf.get_params())
        print("fit done, score %.5f" % fit_score)
    return fit_score

def do_predict(clf, test_X, test_y, print_out=False):
    "predict test data"
    pred_y = clf.predict( test_X )
    pred_correct = sum(pred_y == test_y)
    pred_score = pred_correct/test_y.shape[0]
    if print_out:
        print("pred score %.5f (%d of %d)" % \
          (pred_score, pred_correct, test_y.shape[0]))
    return pred_y, pred_score

def time_fit_pred(clf, dfx, dfy, var='TF', num=10, rp=3):
    '''time fit and predict with classifier clf on training set dfx, dfy
       using num loops and rp repeats'''
    
    # if I use dfy[var] fit sometimes twice as long
    # dfy['TF'] has two states (0, 1)
    def fit_clf():
        do_fit(clf, dfx, dfy['TF'])
    
    # should run predict on test not train data
    def predict_clf():
        do_predict(clf, dfx, dfy['TF'])
    
    # dfy['Y'] has six states (1-6)
    def fit_clf_multi():
        do_fit(clf, dfx, dfy['Y'])
    
    def predict_clf_multi():
        do_predict(clf, dfx, dfy['Y'])
    
    if var=='Y':
        tfit  = min(timeit.repeat(fit_clf_multi, number=num)) * 1e3 / num
        tpred = min(timeit.repeat(predict_clf_multi, number=num)) * 1e3 / num
    else:
        tfit  = min(timeit.repeat(fit_clf, number=num)) * 1e3 / num
        tpred = min(timeit.repeat(predict_clf, number=num)) * 1e3 / num

    return tfit, tpred


def main():
    datadir = get_datadir()
    dftrain, dftrain_y = read_train_data(datadir, printOut=True)
    
    dfx, dfy = get_partial_data(dftrain, dftrain_y, rowf=4, colf=16, printOut=True)
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)

    do_fit(clf, dfx, dfy['TF'], print_out=True)
    do_predict(clf, dfx, dfy['TF'], print_out=True)  # should run on test data
    
    tfit, tpred = time_fit_pred(clf, dfx, dfy, num=30)
    print("df shape %s svm time: fit %.3f ms, predict %.3f ms" % (dfx.shape, tfit, tpred))

if __name__ == '__main__':
    main()


