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

pf_scale_dir = 'Portfolio/Scale'
pf_scale_methods = ['SummarySearch', 'RCLSolve', 'StochasticSketchRefine']
pf_scale_hardness = [0.50, 0.61, 0.89, 1.31, 1.54]

tpch_scale_dir = 'TPC-H/Scale'
tpch_scale_methods = ['SummarySearch', 'RCLSolve', 'StochasticSketchRefine']
tpch_scale_hardness = [0.24, 0.35, 0.69, 0.82, 1.05]


pf_volatility_dir = 'Portfolio/Volatility'
pf_volatility_methods = ['SummarySearch', 'RCLSolve']
pf_volatility_hardness = [0.72, 0.93, 1.11, 1.78, 2.12]

tpch_volatility_dir = 'TPC-H/Volatility'
tpch_volatility_methods = ['SummarySearch', 'RCLSolve']
tpch_volatility_hardness = [0.43, 0.63, 0.94, 1.09, 1.43]

queries = ['Q1.txt', 'Q2.txt', 'Q3.txt', 'Q4.txt', 'Q5.txt']
colors = ['tab:blue', 'tab:green', 'tab:red']
markers = ['o', 'x', 's']

scale_fig, axs = plt.subplots(2, 5, figsize=(20, 5))
# Portfolio Scale

qcnt = 0
quality, quality_std = \
    read_quality_file(pf_scale_dir+'/Quality.txt')
x = read_x_axis(pf_scale_dir + '/X-Axis.txt')
    
for query in queries:
    matplotlib.rcParams.update({'font.size': 15})
    axs[0, qcnt].set_ylim(top=6300/60)
    axs[0, qcnt].set_xlim(min(x), max(x))
    axs[0, qcnt].set_title('Portfolio Q' + str(qcnt+1))# +\n' +\
                           #'\u03BC = ' + str(round(quality[qcnt], 2)) + ' \u03C3 = '+ str(round(quality_std[qcnt], 2)))
    axs[0, qcnt].text(0.84, 0.85, 'H \u2248 ' + str(round(pf_scale_hardness[qcnt], 2)),
                      transform=axs[0, qcnt].transAxes, ha='center', fontsize=15)
    axs[0, qcnt].text(0.15, 0.68, '\u03BC = ' + str(round(1-quality[qcnt], 2)) + '\n\u03C3 = '+ str(round(quality_std[qcnt], 2)),
                      transform=axs[0, qcnt].transAxes, ha='center', fontsize=15)
    if qcnt == 0:
        axs[0, qcnt].set_ylabel('Runtime (mins)', fontsize=15)
    methodcnt = 0
    for method in pf_scale_methods:
        y, y_err = read_runtime_file(pf_scale_dir + '/' + method\
                                     + '/' + query)
        for _ in range(len(y)):
            y[_] /= 60
            y_err[_] /= 60
        color = colors[methodcnt]
        marker = markers[methodcnt]
        axs[0, qcnt].set_xscale('log')
        #axs[0, qcnt].set_yscale('log')
        axs[0, qcnt].tick_params(axis='both', which='major', labelsize=15)
        axs[0, qcnt].errorbar(x=x[:len(y)], y=y, yerr=y_err, marker=marker,
                     label=pf_scale_methods[methodcnt], linewidth=2, markersize=7)
        methodcnt += 1
    qcnt += 1

# TPC-H Scale

qcnt = 0
quality, quality_std = \
    read_quality_file(tpch_scale_dir+'/Quality.txt')
x = read_x_axis(tpch_scale_dir + '/X-Axis.txt')
        
for query in queries:
    matplotlib.rcParams.update({'font.size': 15})
    axs[1, qcnt].set_ylim(top=3200/60) 
    axs[1, qcnt].set_xlim(min(x), max(x))
    axs[1, qcnt].set_title('TPC-H Q' + str(qcnt+1))
    axs[1, qcnt].text(0.15, 0.68, '\u03BC = ' + str(round(1-quality[qcnt], 2)) + '\n\u03C3 = '+ str(round(quality_std[qcnt], 2)),
                      transform=axs[1, qcnt].transAxes, ha='center', fontsize=15)
    axs[1, qcnt].text(0.84, 0.85, 'H \u2248 ' + str(round(tpch_scale_hardness[qcnt], 2)),
                      transform=axs[1, qcnt].transAxes, ha='center', fontsize=15)
    if qcnt == 0:
        axs[1, qcnt].set_xlabel('No. of Tuples', fontsize=15)
        axs[1, qcnt].set_ylabel('Runtime (mins)', fontsize=15)
    else:
        axs[1, qcnt].set_xlabel('No. of Tuples', fontsize=15)
    methodcnt = 0
    for method in tpch_scale_methods:
        y, y_err = read_runtime_file(tpch_scale_dir + '/' + method\
                                     + '/' + query)
        for _ in range(len(y)):
            y[_] /= 60
            y_err[_] /= 60
        color = colors[methodcnt]
        marker = markers[methodcnt]
        axs[1, qcnt].set_xscale('log')
        # axs[1, qcnt].set_yscale('log')
        axs[1, qcnt].tick_params(axis='both', which='major', labelsize=15)
        axs[1, qcnt].errorbar(x=x[:len(y)], y=y, yerr=y_err, marker=marker,
                     label=tpch_scale_methods[methodcnt], linewidth=2, markersize=7)
        methodcnt += 1
    qcnt += 1


# Combine legends
handles, labels = [], []
for row in axs:
  for ax in row:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
    break
  break

scale_fig.legend(handles, labels, loc='outside lower center',
                 bbox_to_anchor=(0.5,-0.1), ncol=3)

scale_fig.tight_layout()
scale_fig.savefig('Runtime_Combined.jpg', bbox_inches='tight')


plt.clf()
volatility_fig, axs = plt.subplots(2, 5, figsize=(20, 5))
# Portfolio Volatility

qcnt = 0
quality, quality_std = \
    read_quality_file(pf_volatility_dir+'/Quality.txt')
x = read_x_axis(pf_volatility_dir + '/X-Axis.txt')

for query in queries:
    matplotlib.rcParams.update({'font.size': 15})
    axs[0, qcnt].set_ylim(top=3000/60) 
    axs[0, qcnt].set_title('Portfolio Q' + str(qcnt+1), pad=25)#+ '\nH \u2248 [' + str(round(pf_scale_hardness[qcnt], 2)) + ', ' + str(round(pf_volatility_hardness[qcnt], 2)) + ']')# +\
                           # '\n\u03BC = ' + str(round(quality[qcnt], 2)) + ', \u03C3 = '+ str(round(quality_std[qcnt], 2)))
    axs[0, qcnt].text(0.5, 1.08, str(round(pf_scale_hardness[qcnt], 2)) + '------------------ H ------------------' + str(round(pf_volatility_hardness[qcnt], 2)),
                      color=(0.3, 0.3, 0.3), transform=axs[0, qcnt].transAxes, ha='center', fontdict={'family': 'sans-serif', 'size': 12, 'style': 'italic'})
    axs[0, qcnt].text(0.17, 0.53, '\u03BC = ' + str(round(1-quality[qcnt], 2)) + '\n\u03C3 = '+ str(round(quality_std[qcnt], 2)),
                      transform=axs[0, qcnt].transAxes, ha='center', fontsize=15)
    if qcnt == 0:
        axs[0, qcnt].set(xlabel='Volatility Coefficient', ylabel='Runtime (mins)')
    else:
        axs[0, qcnt].set(xlabel='Volatility Coefficient')
    methodcnt = 0
    for method in pf_volatility_methods:
        y, y_err = read_runtime_file(pf_volatility_dir + '/' + method\
                                     + '/' + query)
        for _ in range(len(y)):
            y[_] /= 60
            y_err[_] /= 60
        color = colors[methodcnt]
        marker = markers[methodcnt]
        axs[0, qcnt].errorbar(x=x[:len(y)], y=y, yerr=y_err, marker=marker,
                     label=tpch_scale_methods[methodcnt], linewidth=2, markersize=7)
        methodcnt += 1
    qcnt += 1


# TPC-H Volatility

qcnt = 0
quality, quality_std = \
    read_quality_file(tpch_volatility_dir+'/Quality.txt')
x = read_x_axis(tpch_volatility_dir + '/X-Axis.txt')

for query in queries:
    matplotlib.rcParams.update({'font.size': 15})
    axs[1, qcnt].set_ylim(top=900/60) 
    
    axs[1, qcnt].set_title('TPC-H Q' + str(qcnt+1), pad=25)#+ '\nH \u2248 [' + str(round(pf_scale_hardness[qcnt], 2)) + ', ' + str(round(pf_volatility_hardness[qcnt], 2)) + ']')# +\
                           # '\n\u03BC = ' + str(round(quality[qcnt], 2)) + ', \u03C3 = '+ str(round(quality_std[qcnt], 2)))
    axs[1, qcnt].text(0.5, 1.08,  str(round(tpch_scale_hardness[qcnt], 2)) + '------------------ H ------------------' + str(round(tpch_volatility_hardness[qcnt], 2)),
                      color=(0.3, 0.3, 0.3), transform=axs[1, qcnt].transAxes, ha='center', fontdict={'family': 'sans-serif', 'size': 12, 'style': 'italic'})
    axs[1, qcnt].text(0.17, 0.53, '\u03BC = ' + str(round(1-quality[qcnt], 2)) + '\n\u03C3 = '+ str(round(quality_std[qcnt], 2)),
                      transform=axs[1, qcnt].transAxes, ha='center', fontsize=15)
    
    #axs[1, qcnt].set_title('TPC-H Q' + str(qcnt+1) + '\nH \u2248 [' + str(round(tpch_scale_hardness[qcnt], 2)) + ', ' + str(round(tpch_volatility_hardness[qcnt], 2)) + ']')# +\
    #                       # '\n\u03BC = ' + str(round(quality[qcnt], 2)) + ', \u03C3 = '+ str(round(quality_std[qcnt], 2)))
    #axs[1, qcnt].text(0.15, 0.50, '\u03BC = ' + str(round(quality[qcnt], 2)) + '\n\u03C3 = '+ str(round(quality_std[qcnt], 2)),
    #                  transform=axs[1, qcnt].transAxes, ha='center', fontsize=15)
    if qcnt == 0:
        axs[1, qcnt].set(xlabel='Variance Coefficient', ylabel='Runtime (mins)')
    else:
        axs[1, qcnt].set(xlabel='Variance Coefficient')
    methodcnt = 0
    for method in tpch_volatility_methods:
        y, y_err = read_runtime_file(tpch_volatility_dir + '/' + method\
                                     + '/' + query)
        for _ in range(len(y)):
            y[_] /= 60
            y_err[_] /= 60
        color = colors[methodcnt]
        marker = markers[methodcnt]
        axs[1, qcnt].errorbar(x=x[:len(y)], y=y, yerr=y_err, marker=marker,
                     label=tpch_scale_methods[methodcnt], linewidth=2, markersize=7)
        methodcnt += 1
    qcnt += 1


# Combine legends
handles, labels = [], []
for row in axs:
  for ax in row:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
    break
  break

volatility_fig.legend(handles, labels, loc='outside lower center',
                 bbox_to_anchor=(0.5,-0.1), ncol=3)

volatility_fig.tight_layout()
volatility_fig.savefig('Variance_Combined.jpg', bbox_inches='tight')



