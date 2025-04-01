import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
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

pf_scale_dir = 'Portfolio/Scale'
pf_scale_methods = ['SummarySearch', 'RCLSolve', 'StochasticSketchRefine']
pf_scale_hardness = [0.61, 0.82, 0.94, 1.28, 1.54]

tpch_scale_dir = 'TPC-H/Scale'
tpch_scale_methods = ['SummarySearch', 'RCLSolve', 'StochasticSketchRefine']
tpch_scale_hardness = [0.80, 1.24, 1.43, 1.56, 1.68]

pf_volatility_dir = 'Portfolio/Volatility'
pf_volatility_methods = ['SummarySearch', 'RCLSolve']

tpch_volatility_dir = 'TPC-H/Volatility'
tpch_volatility_methods = ['SummarySearch', 'RCLSolve']

queries = ['Q1.txt', 'Q2.txt', 'Q3.txt', 'Q4.txt', 'Q5.txt']
colors = ['tab:blue', 'tab:green', 'tab:red']
markers = ['o', 'v', 's']

# Portfolio Scale

qcnt = 0
quality, quality_std = \
    read_quality_file(pf_scale_dir+'/Quality.txt')
x = read_x_axis(pf_scale_dir + '/X-Axis.txt')

for query in queries:
    matplotlib.rcParams.update({'font.size': 15})
    plt.clf()
    plt.ylim(top=4500) 
    plt.title('\u03BC = ' + str(round(quality[qcnt], 2)) +
              ' \u03A3 = '+ str(round(quality_std[qcnt], 2)))
    plt.xlabel("No. of Tuples")
    plt.ylabel("Runtime (secs)")
    qcnt += 1
    methodcnt = 0
    for method in pf_scale_methods:
        y, y_err = read_runtime_file(pf_scale_dir + '/' + method\
                                     + '/' + query)
        color = colors[methodcnt]
        marker = markers[methodcnt]
        plt.errorbar(x=x[:len(y)], y=y, yerr=y_err, marker=marker,
                     label=pf_scale_methods[methodcnt], linewidth=3.0)
        methodcnt += 1
    plt.legend(loc="upper right")
    plt.savefig('PF_Scale_Q'+str(qcnt)+'.jpg')

# Portfolio Volatility

qcnt = 0
quality, quality_std = \
    read_quality_file(pf_volatility_dir+'/Quality.txt')
x = read_x_axis(pf_volatility_dir + '/X-Axis.txt')

for query in queries:
    plt.clf()
    plt.ylim(top=2500) 
    matplotlib.rcParams.update({'font.size': 15})
    plt.title('\u03BC = ' + str(round(quality[qcnt], 2)) +
              ' \u03A3 = '+ str(round(quality_std[qcnt], 2)))
    plt.xlabel("Volatility Coefficient")
    plt.ylabel("Runtime (secs)")
    qcnt += 1
    methodcnt = 0
    for method in pf_volatility_methods:
        y, y_err = read_runtime_file(pf_volatility_dir + '/' + method\
                                     + '/' + query)
        color = colors[methodcnt]
        marker = markers[methodcnt]
        plt.errorbar(x=x[:len(y)], y=y, yerr=y_err, marker=marker,
                     label=pf_volatility_methods[methodcnt], linewidth=3.0)
        methodcnt += 1
    plt.legend(loc="upper right")
    plt.savefig('PF_volatility_Q'+str(qcnt)+'.jpg')


# TPC-H Scale

qcnt = 0
quality, quality_std = \
    read_quality_file(tpch_scale_dir+'/Quality.txt')
x = read_x_axis(tpch_scale_dir + '/X-Axis.txt')

for query in queries:
    plt.clf()
    plt.ylim(top=2500)
    matplotlib.rcParams.update({'font.size': 15})
    plt.title('\u03BC = ' + str(round(quality[qcnt], 2)) +
              ' \u03A3 = '+ str(round(quality_std[qcnt], 2)))
    plt.xlabel("No. of Tuples")
    plt.ylabel("Runtime (secs)")
    qcnt += 1
    methodcnt = 0
    for method in tpch_scale_methods:
        y, y_err = read_runtime_file(tpch_scale_dir + '/' + method\
                                     + '/' + query)
        color = colors[methodcnt]
        marker = markers[methodcnt]
        plt.errorbar(x=x[:len(y)], y=y, yerr=y_err, marker=marker,
                     label=tpch_scale_methods[methodcnt], linewidth=3.0)
        methodcnt += 1
    plt.legend(loc="upper right")
    plt.savefig('TPCH_Scale_Q'+str(qcnt)+'.jpg')


# TPC-H Volatility

qcnt = 0
quality, quality_std = \
    read_quality_file(tpch_volatility_dir+'/Quality.txt')
x = read_x_axis(tpch_volatility_dir + '/X-Axis.txt')

for query in queries:
    plt.clf()
    plt.ylim(top=1000) 
    matplotlib.rcParams.update({'font.size': 15})
    plt.title('\u03BC = ' + str(round(quality[qcnt], 2)) +
              ' \u03A3 = '+ str(round(quality_std[qcnt], 2)))
    plt.xlabel("Variance Coefficient")
    plt.ylabel("Runtime (secs)")
    qcnt += 1
    methodcnt = 0
    for method in tpch_volatility_methods:
        y, y_err = read_runtime_file(tpch_volatility_dir + '/' + method\
                                     + '/' + query)
        color = colors[methodcnt]
        marker = markers[methodcnt]
        plt.errorbar(x=x[:len(y)], y=y, yerr=y_err, marker=marker,
                     label=tpch_volatility_methods[methodcnt], linewidth=3.0)
        methodcnt += 1
    plt.legend(loc="upper right")
    plt.savefig('TPCH_volatility_Q'+str(qcnt)+'.jpg')