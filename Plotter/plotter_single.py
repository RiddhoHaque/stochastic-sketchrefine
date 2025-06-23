import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import numpy as np

rcParams.update({'figure.autolayout': True})

def read_runtime_file(filename):
    file = open(filename, 'r')
    y = []
    y_err = []
    for line in file.readlines():
        line = line.strip()
        words = line.split(',')
        y.append(float(words[0]))
        y_err.append(float(words[1]))
    return y, y_err

def read_x_axis(filename):
    x = []
    for line in open(filename, 'r').readlines():
        line = line.strip()
        x.append(float(line))
    return x

pf_scale_dir = 'Portfolio/Scale'
pf_scale_methods = ['SummarySearch', 'RCLSolve', 'StochasticSketchRefine']

queries = ['Q1.txt']
colors = ['tab:blue', 'tab:green', 'tab:red']
markers = ['o', 'x', 's']

plt.figure(figsize=(9, 5))
x = read_x_axis(pf_scale_dir + '/X-Axis.txt')

matplotlib.rcParams.update({'font.size': 15})

plt.ylim(top=4200/60)
plt.xlim(min(x), max(x))
#plt.title('Runtime Comparisons', fontsize=25)

plt.ylabel('Runtime (mins)', fontsize=25)
plt.xlabel('No. of Tuples', fontsize=25)

methodcnt = 0
for method in pf_scale_methods:
    y, y_err = read_runtime_file(pf_scale_dir + '/' + method + '/' + queries[0])
    for _ in range(len(y)):
        y[_] /= 60
        y_err[_] /= 60
    
    color = colors[methodcnt]
    marker = markers[methodcnt]
    plt.xscale('log')
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.errorbar(x=x[:len(y)], y=y, yerr=y_err, marker=marker,
                 label=pf_scale_methods[methodcnt], linewidth=5, markersize=15)
    methodcnt += 1

plt.legend(fontsize=20)
plt.savefig('Demo_Plot.jpg')
