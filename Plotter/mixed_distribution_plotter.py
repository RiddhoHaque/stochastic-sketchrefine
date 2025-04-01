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
        words=line.split(',')
        y.append(float(words[0]))
        y_err.append(float(words[1]))
    return y, y_err

def read_quality_file(filename):
    quality, quality_std = \
        read_runtime_file(filename)
    return quality, quality_std

def read_x_axis(filename):
    x = []
    for line in \
        open(filename, 'r').readlines():
        line = line.strip()
        x.append(float(line))
    return x

def read_labels(filename):
    labels = []
    for line in\
        open(filename, 'r').readlines():
        line = line.strip()
        labels.append(line)
    return labels

tpch_mixed_dir = 'TPC-H/Mixed_Distributions'
tpch_mixed_methods = ['SummarySearch', 'RCLSolve', 'StochasticSketchRefine']

queries = ['Q1.txt']
colors = ['tab:blue', 'tab:green', 'tab:red']
markers = ['o', 'x', 's']

mixed_fig, axs = plt.subplots(1, 1, figsize=(8, 4))

# TPC-H Scale

qcnt = 0
quality, quality_std = \
    read_quality_file(tpch_mixed_dir+'/Quality.txt')
x = read_x_axis(tpch_mixed_dir + '/X-Axis.txt')
        
for query in queries:
    matplotlib.rcParams.update({'font.size': 15})
    axs.set_ylim(top=1450/60) 
    axs.set_xlim(min(x), max(x))
    axs.set_title('TPC-H With Mixed Distributions', fontsize=20)
    axs.text(0.1, 0.75, '\u03BC = ' + str(round(1-quality[qcnt], 2)) + '\n\u03C3 = '+ str(round(quality_std[qcnt], 2)),
                      transform=axs.transAxes, ha='center', fontsize=20)
    axs.set_xlabel('No. of Tuples', fontsize=20)
    axs.set_ylabel('Runtime (mins)', fontsize=20)
    
    axs.set_xscale('log')

    methodcnt = 0
    for method in tpch_mixed_methods:
        y, y_err = read_runtime_file(tpch_mixed_dir + '/' + method\
                                     + '/' + query)
        for _ in range(len(y)):
            y[_] /= 60
            y_err[_] /= 60
        color = colors[methodcnt]
        marker = markers[methodcnt]
        axs.tick_params(axis='both', which='major', labelsize=20)
        axs.errorbar(x=x[:len(y)], y=y, yerr=y_err, marker=marker,
                     label=tpch_mixed_methods[methodcnt], linewidth=2, markersize=7)
        methodcnt += 1
    qcnt += 1


# Combine legends
handles, labels = [], []
h, l = axs.get_legend_handles_labels()
handles.extend(h)
labels.extend(l)

mixed_fig.legend(handles, labels, loc='outside lower center',
                 bbox_to_anchor=(0.5,-0.1), ncol=3)

mixed_fig.tight_layout()
mixed_fig.savefig('Runtime_Mixed_Dir.jpg', bbox_inches='tight')


plt.clf()

