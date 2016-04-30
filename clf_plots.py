# reprocess aggregate plots into other forms

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_plotdir():
    "get plot directory, set seaborn plot params"
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.5)
    return 'speed_test_plots/'

def make_plotdir():
    "make plot directory on file system"
    plotdir = get_plotdir()
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def parse_file(fname):
    "read and parse file fname, return dataframe"

    pat = re.compile("\D+(\d+),\s(\d+).*fit\s(\d+\.\d+).*predict\s(\d+\.\d+)")
    df = pd.DataFrame()
    with open(fname, 'r') as f:
#       automatic file close, exception handling
        title = 'none'
        key = 'none'
        for line in f:
            if line.startswith('starting'):
                _, _, title = line.partition('with ')
                title = title.rstrip()
            elif line.find('rows') > 0:
                key = 'row'
            elif line.find('columns') > 0:
                key = 'column'
            elif line.startswith('df shape'):
                s1 = pat.findall(line)
                rc, cc, ft, pt = s1[0]
# now enter this nonsense into a dataframe so I can plot it
                dd = {'title':title, 'key':key, 'row':int(rc), 'column':int(cc), 'fit':float(ft), 'predict':float(pt) }
                df = df.append(dd, ignore_index=True)

#   add colors - really there's a better way to do this
    titles = list(set(df['title']))
    colors = ['red', 'blue', 'orange', 'green']
    df['color'] = df['title'].apply(lambda s: colors[titles.index(s)])
    return df

def plot_results(plotdir, df, key, var):
    "plot results, key='row'|'column', var='fit'|'predict'"
    titles = list(set(df['title']))
    dfg = df[df['key'] == key]
    plt.clf()
#   df.plot.scatter(key, var)
    for t in titles:
        dft = dfg[dfg['title']==t]
        plt.scatter(dft[key], dft[var], c=dft['color'], edgecolors="face", s=60, alpha=0.7)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((0, x2, -0.02*y2, y2))
    plt.xlabel("%s count" % key)
    plt.ylabel("time (ms)")
    plt.legend(titles, loc='upper left')
    plt.title("Classifier %s %s speed" % (key, var))
    plt.tight_layout()
    plt.savefig("%sclf_time_%s_%s.png" % (plotdir, key, var))

def main():
    df = parse_file('clf_speed_outputB.txt')
#   print(df)
    print(df[df['key']=='row'])
    print(df[df['key']=='col'])
    plotdir = make_plotdir()
    plot_results(plotdir, df, 'row', 'fit')
    plot_results(plotdir, df, 'row', 'predict')
    plot_results(plotdir, df, 'column', 'fit')
    plot_results(plotdir, df, 'column', 'predict')
    print("Fit speed is fast, about the same for Naive Bayes, LinearSVC.  Logistic Regression is slowest.")
    print("Predict speed is fast, about the same for Logistic Regression, LinearSVC (matters more in production).")

if __name__ == '__main__':
    main()

