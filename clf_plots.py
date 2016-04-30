# reprocess plots into other forms

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
#   automatic file close, exception handling
    with open(fname, 'r') as f:
        title = 'none'
        key = 'none'
        for line in f:
            if line.startswith('starting'):
                _, _, title = line.partition('with ')
                title = title.rstrip()
#               print(title)
            elif line.find('rows') > 0:
                key = 'row'
            elif line.find('columns') > 0:
                key = 'col'
            elif line.startswith('df shape'):
#               print(line, end='')
                s1 = pat.findall(line)
                rc, cc, ft, pt = s1[0]
#               print("%s %s: rows %d, cols %d, fitt %.3f, predt %.3f" % \
#                   (title, key, int(rc), int(cc), float(ft), float(pt)))
# now enter this nonsense into a dataframe so I can plot it
                dd = {'title':title, 'key':key, 'row':int(rc), 'col':int(cc), 'fit':float(ft), 'pred':float(pt) }
                df = df.append(dd, ignore_index=True)

#   add colors
#   titles = list(set(df['title']))
#   df['color'] = df['title'].apply()
    return df

def plot_results(df, key, var, label, plotdir):
    "key=row|col, var=fit|pred"
#   titles = list(set(df['title']))
    titles = set(df['title'])
    print("titles", titles)
    plt.clf()
#   plt.scatter(df[key], df[var], c=titles.index(df['title']), s=60)
    df.plot.scatter(key, var)
#   df.plot.scatter(key, var, c='title')
#   for t in titles:
#       dft = df[df['title']==t]
#       ax = dft.plot.scatter(key, var)
#   plt.title("key %s, var %s" % (key, var))
    plt.savefig(plotdir+"fits_"+label+"_"+key+"_"+var+".png")

def main():
    df = parse_file('clf_speed_outputB.txt')
#   print(df)
    print(df[df['key']=='row'])
    print(df[df['key']=='col'])
#   plotdir = make_plotdir()
#   plot_results(df, 'row', 'fit', 'aa', plotdir)

if __name__ == '__main__':
    main()

