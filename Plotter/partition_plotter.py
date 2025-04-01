import matplotlib.pyplot as plt
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

def read_labels(filename):
    labels = []
    for line in\
        open(filename, 'r').readlines():
        line = line.strip()
        labels.append(line)
    return labels

tpch_dpart_runtimes = read_x_axis(
    'TPC-H/Partitioning/Runtime/DistPartition.txt')
pf_dpart_runtimes = read_x_axis(
    'Portfolio/Partitioning/Runtime/DistPartition.txt')
tpch_kdtree_runtimes = read_x_axis(
    'TPC-H/Partitioning/Runtime/KD-Tree.txt')
pf_kdtree_runtimes = read_x_axis(
    'Portfolio/Partitioning/Runtime/KD-Tree.txt')

labels = read_labels(
    'Portfolio/Partitioning/Runtime/Labels.txt')

bars = ['With DistPartition', 'With KD-Trees']

colors = ['blue', 'green', 'red', 'yellow', 'pink', 'orange']

plt.cla()
plt.figure(figsize=(20,6))

index = 0

bs = []

for label in labels:
    left_dpart = 0
    subindex_dpart = 0
    while subindex_dpart < index:
        left_dpart +=\
            tpch_dpart_runtimes[
                subindex_dpart]
        subindex_dpart += 1
    
    left_kd = 0
    subindex_kd = 0
    while subindex_kd < index:
        left_kd +=\
            tpch_kdtree_runtimes[subindex_kd]
        subindex_kd += 1
    
    b = plt.barh(bars, [tpch_dpart_runtimes[index], tpch_kdtree_runtimes[index]],
                  left=[left_dpart, left_kd], color=colors[index])
    
    bs.append(b)
    index += 1

plt.title('Offline Preprocessing Runtimes (TPC-H)')
plt.legend(bs, labels, title="Runtime (secs)", loc='upper left',  bbox_to_anchor=(1.05,1.0))
plt.savefig('TPCH_Partitioning.jpg', bbox_inches='tight')

plt.cla()
plt.figure(figsize=(20,6))

index = 0

bs = []

for label in labels:
    left_dpart = 0
    subindex_dpart = 0
    while subindex_dpart < index:
        left_dpart +=\
            pf_dpart_runtimes[
                subindex_dpart]
        subindex_dpart += 1
    
    left_kd = 0
    subindex_kd = 0
    while subindex_kd < index:
        left_kd +=\
            pf_kdtree_runtimes[subindex_kd]
        subindex_kd += 1
    
    b = plt.barh(bars, [pf_dpart_runtimes[index], pf_kdtree_runtimes[index]],
                  left=[left_dpart, left_kd], color=colors[index])
    
    bs.append(b)
    index += 1

plt.title('Offline Preprocessing Runtimes (Portfolio)')
plt.legend(bs, labels, title="Runtime (secs)", loc='upper left',  bbox_to_anchor=(1.05,1.0))
plt.savefig('PF_Partitioning.jpg', bbox_inches='tight')