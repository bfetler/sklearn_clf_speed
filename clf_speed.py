# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression as lr
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn import svm

from speed_test import init_data, make_plotdir, speed_test_medium, \
     speed_test_large, speed_test_medium_six

def logistic_regression_speed_test(dftrain, dftrain_y, plotdir):
    atitle = 'Logistic Regression'
    afile = 'logreg'   
    clf = lr()
    speed_test_medium(clf, dftrain, dftrain_y, atitle, afile, plotdir)
    speed_test_large(clf, dftrain, dftrain_y, atitle, afile, plotdir)
    speed_test_medium_six(clf, dftrain, dftrain_y, atitle, afile, plotdir)

def naive_bayes_speed_test(dftrain, dftrain_y, plotdir):
    atitle = 'Naive Bayes'
    afile = 'nbayes'   
    clf = gnb()
    speed_test_medium(clf, dftrain, dftrain_y, atitle, afile, plotdir)
    speed_test_large(clf, dftrain, dftrain_y, atitle, afile, plotdir)
    speed_test_medium_six(clf, dftrain, dftrain_y, atitle, afile, plotdir)

def linear_svm_speed_test(dftrain, dftrain_y, plotdir):
    atitle = 'Linear SVM'
    afile = 'linsvm'    
    clf = svm.LinearSVC()
    speed_test_medium(clf, dftrain, dftrain_y, atitle, afile, plotdir)
    speed_test_large(clf, dftrain, dftrain_y, atitle, afile, plotdir)
    speed_test_medium_six(clf, dftrain, dftrain_y, atitle, afile, plotdir)

def svm_speed_test(dftrain, dftrain_y, plotdir):
    atitle = 'SVM'
    afile = 'svm'
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    speed_test_medium(clf, dftrain, dftrain_y, atitle, afile, plotdir)
    speed_test_large(clf, dftrain, dftrain_y, atitle, afile, plotdir)
    speed_test_medium_six(clf, dftrain, dftrain_y, atitle, afile, plotdir)

def main():
    plotdir = make_plotdir()
    dftrain, dftrain_y = init_data() 
    
    logistic_regression_speed_test(dftrain, dftrain_y, plotdir)
#    naive_bayes_speed_test(dftrain, dftrain_y, plotdir)
    linear_svm_speed_test(dftrain, dftrain_y, plotdir)
#    svm_speed_test(dftrain, dftrain_y, plotdir)

if __name__ == '__main__':
    main()

