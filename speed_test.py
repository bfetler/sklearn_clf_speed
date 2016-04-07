# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
    dftrain.columns = map(lambda s: "x" + str(s), range(d1))
    
    # drop last two columns
    dftrain.drop(["x"+str(d1-2), "x"+str(d1-1)], axis=1, inplace=True)
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

def get_partial_data(dftrain, dftrain_y, rowf=4, colf=4, printOut=False):
    "get data frames of size reduced by rows 1/rowf, columns 1/colf"
    oldrow = dftrain.shape[0]
    oldcol = dftrain.shape[1]
    newrow = (oldrow+1) // rowf
    newcol = (oldcol+1) // colf
#    print("newrow %d oldrow %d, newcol %d oldcol %d" % (newrow, oldrow, newcol, oldcol))
    
    blug = map(lambda x: "x"+str(x-1), range(newcol, oldcol))
#    print("blug", blug)
    
    dfx = dftrain.drop(blug, axis=1)
    dfx = dfx.drop(range(newrow, oldrow))  # use inplace=True if not =
    dfy = dftrain_y.drop(range(newrow, oldrow))
    
    if printOut==True:
        print("dfx shape", dfx.shape, "head\n", dfx[:3])
        print("dfy shape", dfy.shape, "head\n", dfy[:3])
        print("dfx tail\n", dfx[-3:])
        print("dfy tail\n", dfy[-3:])
        print("dfy describe \n%s" % dfy.describe())
    
    return dfx, dfy


def main():
    datadir = get_datadir()
    dftrain, dftrain_y = read_train_data(datadir, printOut=True)
    dfx, dfy = get_partial_data(dftrain, dftrain_y, rowf=8, colf=8, printOut=True)

if __name__ == '__main__':
    main()


