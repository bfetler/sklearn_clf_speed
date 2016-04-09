# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression as lr
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn import svm

from speed_test import init_data, make_plotdir, speed_test_medium

def logistic_regression_speed_test():
    atitle = 'Logistic Regression'
    afile = 'logreg'
    plotdir = make_plotdir()
    dftrain, dftrain_y = init_data()    
    clf = lr()
    speed_test_medium(clf, dftrain, dftrain_y, atitle, afile, plotdir)

def naive_bayes_speed_test():
    atitle = 'Naive Bayes'
    afile = 'nbayes'
    plotdir = make_plotdir()
    dftrain, dftrain_y = init_data()    
    clf = gnb()
    speed_test_medium(clf, dftrain, dftrain_y, atitle, afile, plotdir)

def linear_svm_speed_test():
    atitle = 'Linear SVM'
    afile = 'linsvm'
    plotdir = make_plotdir()
    dftrain, dftrain_y = init_data()    
    clf = svm.LinearSVC()
    speed_test_medium(clf, dftrain, dftrain_y, atitle, afile, plotdir)

def svm_speed_test():
    atitle = 'SVM'
    afile = 'svm'
    plotdir = make_plotdir()
    dftrain, dftrain_y = init_data()    
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    speed_test_medium(clf, dftrain, dftrain_y, atitle, afile, plotdir)

def main():
    logistic_regression_speed_test()
    naive_bayes_speed_test()
    linear_svm_speed_test()
    svm_speed_test()

if __name__ == '__main__':
    main()

