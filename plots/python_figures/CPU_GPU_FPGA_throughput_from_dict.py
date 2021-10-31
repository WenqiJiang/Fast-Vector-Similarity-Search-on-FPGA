# Example usaage: 
#   python CPU_GPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir '../cpu_performance_result_m6i.4xlarge/cpu_throughput_SIFT100M.pkl' --gpu_performance_dict_dir '../gpu_performance_result_p3.8xlarge_V100/gpu_throughput_SIFT100M.pkl' --fpga_performance_dict_dir '../fpga_performance_result/FPGA_perf_dict_SIFT100M.pkl' --dbname SIFT100M --topK 100 --recall_goal 0.95

# Full command list
#   python CPU_GPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir '../cpu_performance_result_m6i.4xlarge/cpu_throughput_SIFT100M.pkl' --gpu_performance_dict_dir '../gpu_performance_result_p3.8xlarge_V100/gpu_throughput_SIFT100M.pkl' --fpga_performance_dict_dir '../fpga_performance_result/FPGA_perf_dict_SIFT100M.pkl' --dbname SIFT100M --topK 1 --recall_goal 0.25 --legend_loc_x 0.75
#   python CPU_GPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir '../cpu_performance_result_m6i.4xlarge/cpu_throughput_SIFT100M.pkl' --gpu_performance_dict_dir '../gpu_performance_result_p3.8xlarge_V100/gpu_throughput_SIFT100M.pkl' --fpga_performance_dict_dir '../fpga_performance_result/FPGA_perf_dict_SIFT100M.pkl' --dbname SIFT100M --topK 10 --recall_goal 0.6 --legend_loc_x 0.75
#   python CPU_GPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir '../cpu_performance_result_m6i.4xlarge/cpu_throughput_SIFT100M.pkl' --gpu_performance_dict_dir '../gpu_performance_result_p3.8xlarge_V100/gpu_throughput_SIFT100M.pkl' --fpga_performance_dict_dir '../fpga_performance_result/FPGA_perf_dict_SIFT100M.pkl' --dbname SIFT100M --topK 100 --recall_goal 0.95 --legend_loc_x 0.0
#   python CPU_GPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir '../cpu_performance_result_m6i.4xlarge/cpu_throughput_SIFT500M.pkl' --gpu_performance_dict_dir '../gpu_performance_result_p3.8xlarge_V100/gpu_throughput_SIFT500M.pkl' --fpga_performance_dict_dir '../fpga_performance_result/FPGA_perf_dict_SIFT500M.pkl' --dbname SIFT500M --topK 1 --recall_goal 0.25 --legend_loc_x 0.75
#   python CPU_GPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir '../cpu_performance_result_m6i.4xlarge/cpu_throughput_SIFT500M.pkl' --gpu_performance_dict_dir '../gpu_performance_result_p3.8xlarge_V100/gpu_throughput_SIFT500M.pkl' --fpga_performance_dict_dir '../fpga_performance_result/FPGA_perf_dict_SIFT500M.pkl' --dbname SIFT500M --topK 10 --recall_goal 0.6 --legend_loc_x 0.75
#   python CPU_GPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir '../cpu_performance_result_m6i.4xlarge/cpu_throughput_SIFT500M.pkl' --gpu_performance_dict_dir '../gpu_performance_result_p3.8xlarge_V100/gpu_throughput_SIFT500M.pkl' --fpga_performance_dict_dir '../fpga_performance_result/FPGA_perf_dict_SIFT500M.pkl' --dbname SIFT500M --topK 100 --recall_goal 0.95 --legend_loc_x 0.0
#   python CPU_GPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir '../cpu_performance_result_m6i.4xlarge/cpu_throughput_SIFT1000M.pkl' --gpu_performance_dict_dir '../gpu_performance_result_p3.8xlarge_V100/gpu_throughput_SIFT1000M.pkl' --fpga_performance_dict_dir '../fpga_performance_result/FPGA_perf_dict_SIFT1000M.pkl' --dbname SIFT1000M --topK 1 --recall_goal 0.25 --legend_loc_x 0.75
#   python CPU_GPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir '../cpu_performance_result_m6i.4xlarge/cpu_throughput_SIFT1000M.pkl' --gpu_performance_dict_dir '../gpu_performance_result_p3.8xlarge_V100/gpu_throughput_SIFT1000M.pkl' --fpga_performance_dict_dir '../fpga_performance_result/FPGA_perf_dict_SIFT1000M.pkl' --dbname SIFT1000M --topK 10 --recall_goal 0.6 --legend_loc_x 0.75
#   python CPU_GPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir '../cpu_performance_result_m6i.4xlarge/cpu_throughput_SIFT1000M.pkl' --gpu_performance_dict_dir '../gpu_performance_result_p3.8xlarge_V100/gpu_throughput_SIFT1000M.pkl' --fpga_performance_dict_dir '../fpga_performance_result/FPGA_perf_dict_SIFT1000M.pkl' --dbname SIFT1000M --topK 100 --recall_goal 0.95 --legend_loc_x 0.0

# Run with python 3.9 if the following error occurs
# WenqideMacBook-Pro@~/Works/ANNS-FPGA/python_figures wenqi$python CPU_GPU_FPGA_throughput_from_dict.py 
# Traceback (most recent call last):
#   File "CPU_GPU_FPGA_throughput_from_dict.py", line 112, in <module>
#     fpga_cpu = pickle.load(f)
# ValueError: unsupported pickle protocol: 5

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--cpu_performance_dict_dir', type=str, default='../cpu_performance_result_m6i.4xlarge/cpu_throughput_SIFT100M.pkl', 
    help="a dictionary of d[dbname][index_key][topK][recall_goal] -> throughput (QPS)")
parser.add_argument('--gpu_performance_dict_dir', type=str, default='../gpu_performance_result_p3.8xlarge_V100/gpu_throughput_SIFT100M.pkl', 
    help="a dictionary of d[dbname][index_key][topK][recall_goal][batch_size] -> throughput (QPS)")
parser.add_argument('--fpga_performance_dict_dir', type=str, default='./fpga_performance_result/FPGA_perf_dict_SIFT100M.pkl', 
    help="a dictionary of d[dbname][index_key][topK][recall_goal] -> throughput (QPS)")
parser.add_argument('--dbname', type=str, default=0, help="dataset name, e.g., SIFT100M")
parser.add_argument('--topK', type=int, default=10, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
parser.add_argument('--recall_goal', type=float, default=0.5, help="target minimum recall, e.g., 50%=0.5")
parser.add_argument('--legend_loc_x', type=float, default=0.0, help="the x position of legend, 0~1")



args = parser.parse_args()
cpu_performance_dict_dir = args.cpu_performance_dict_dir
gpu_performance_dict_dir = args.gpu_performance_dict_dir
fpga_performance_dict_dir = args.fpga_performance_dict_dir
dbname = args.dbname
topK = args.topK
recall_goal = args.recall_goal

batch_size=1


def get_cpu_performance_tuple(d, dbname, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = throughput (QPS)
    return the performance tuple (index_key, QPS) of certain topK and recall_goal
    """
    performance_tuple = []
    for index_key in d[dbname]:
        if index_key[:len("OPQ16,IMI")] == "OPQ16,IMI":
            continue
        if 'IMI' in index_key:
            continue
        if d[dbname][index_key][topK][recall_goal] is not None:
            performance_tuple.append((index_key.replace(",PQ16",""), d[dbname][index_key][topK][recall_goal]))
        else:
            performance_tuple.append((index_key.replace(",PQ16",""), 0))

    return performance_tuple


def get_gpu_performance_tuple(d, dbname, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = throughput (QPS)
    return the performance tuple (index_key, QPS) of certain topK and recall_goal
    """
    performance_tuple = []
    for index_key in d[dbname]:
        if index_key[:len("OPQ16,IMI")] == "OPQ16,IMI":
            continue
        if topK in d[dbname][index_key] and d[dbname][index_key][topK] is not None and \
            recall_goal in d[dbname][index_key][topK] and d[dbname][index_key][topK][recall_goal] is not None and \
            d[dbname][index_key][topK][recall_goal][batch_size] is not None:
            performance_tuple.append((index_key.replace(",PQ16",""), d[dbname][index_key][topK][recall_goal][batch_size]))
        else:
            performance_tuple.append((index_key.replace(",PQ16",""), 0))

    return performance_tuple

def get_fpga_performance_tuple(d, dbname, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = throughput (QPS)
    return the performance tuple (index_key, QPS) of certain topK and recall_goal
    """
    performance_tuple = []
    for index_key in d[dbname]:
        if topK in d[dbname][index_key] and \
            recall_goal in d[dbname][index_key][topK] and \
            d[dbname][index_key][topK][recall_goal] is not None and \
            index_key[:len("OPQ16,IMI")] != "OPQ16,IMI":
            performance_tuple.append((index_key.replace(",PQ16",""), d[dbname][index_key][topK][recall_goal]))

    return performance_tuple

def normalize_performance_tuple(performance_tuple, factor):

    """
    The networked bitstream can be downgraded in terms of frequency,
        e.g., factor = 126/140 when downgrading to 126MHz from 140MHz
    """


    normalized_performance_tuple = []
    for idx_name, throughput in performance_tuple:
        normalized_performance_tuple.append([idx_name, throughput * factor])

    return normalized_performance_tuple

def sort_performance_tuple(performance_tuple):
    order = ['IVF1024', 'IVF2048', 'IVF4096', 'IVF8192', 'IVF16384', 'IVF32768', 'IVF65536', 'OPQ16,IVF1024', \
        'OPQ16,IVF2048', 'OPQ16,IVF4096', 'OPQ16,IVF8192', 'OPQ16,IVF16384', 'OPQ16,IVF32768', 'OPQ16,IVF65536']

    sorted_performance_tuple = []

    for label in order:
        find = False 
        for (idx_name, throughput) in performance_tuple:
            if idx_name == label:
                sorted_performance_tuple.append((label, throughput))
                find = True
                break
        if not find:
            sorted_performance_tuple.append((label, 0))

    return sorted_performance_tuple


d_cpu = None
with open(cpu_performance_dict_dir, 'rb') as f:
    d_cpu = pickle.load(f)
d_gpu = None
with open(gpu_performance_dict_dir, 'rb') as f:
    d_gpu = pickle.load(f)
d_fpga = None
with open(fpga_performance_dict_dir, 'rb') as f:
    d_fpga = pickle.load(f)

performance_tuple_cpu = sort_performance_tuple(get_cpu_performance_tuple(d_cpu, dbname, topK, recall_goal))
performance_tuple_gpu = sort_performance_tuple(get_gpu_performance_tuple(d_gpu, dbname, topK, recall_goal))
performance_tuple_fpga = sort_performance_tuple(get_fpga_performance_tuple(d_fpga, dbname, topK, recall_goal))

x_labels_cpu = []
y_cpu = []
for (idx_name, throughput) in performance_tuple_cpu:
    y_cpu.append(throughput)
    x_labels_cpu.append(idx_name)

x_labels_gpu = []
y_gpu = []
for (idx_name, throughput) in performance_tuple_gpu:
    y_gpu.append(throughput)
    x_labels_gpu.append(idx_name)

x_labels_fpga = []
y_fpga = []
for (idx_name, throughput) in performance_tuple_fpga:
    y_fpga.append(throughput)
    x_labels_fpga.append(idx_name)

assert x_labels_cpu == x_labels_gpu and x_labels_cpu == x_labels_fpga
x_labels = x_labels_cpu

x = np.arange(len(x_labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
# 
rects1  = ax.bar(x - width, y_cpu, width)#, label='Men')
rects2  = ax.bar(x, y_gpu, width)#, label='Men')
rects3 = ax.bar(x + width, y_fpga, width)#, label='Women')

best_qps_cpu = np.amax(y_cpu)
best_qps_gpu = np.amax(y_gpu)
best_qps_fpga = np.amax(y_fpga)

label_font = 12
tick_font = 10
tick_label_font = 10
legend_font = 10
title_font = 14

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('QPS', fontsize=label_font)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
plt.xticks(rotation=70)

if dbname == 'SIFT100M':
    legend_list = ["CPU", "GPU (1xV100)", "FPGA (1xU280)"]
    price_cpu = 0.768
    price_gpu = 3.06
    price_fpga = 1.65
elif dbname == 'SIFT500M':
    legend_list = ["CPU", "GPU (1xV100)", "FPGA (4xU280)"]
    price_cpu = 0.768
    price_gpu = 3.06
    price_fpga = 1.65 * 4
elif dbname == 'SIFT1000M':
    legend_list = ["CPU", "GPU (2xV100)", "FPGA (8xU280)"]
    price_cpu = 0.768
    price_gpu = 3.06 * 2
    price_fpga = 1.65 * 8

sec_in_hour = 3600
qpd_cpu = best_qps_cpu * sec_in_hour / price_cpu
qpd_gpu = best_qps_gpu * sec_in_hour / price_gpu
qpd_fpga = best_qps_fpga * sec_in_hour / price_fpga
print("Queries per dollar CPU: {:e}".format(qpd_cpu))
print("Queries per dollar GPU: {:e}".format(qpd_gpu))
print("Queries per dollar FPGA: {:e}".format(qpd_fpga))
price_efficiency_over_cpu = (best_qps_fpga / price_fpga) / (best_qps_cpu / price_cpu)
price_efficiency_over_gpu = (best_qps_fpga / price_fpga) / (best_qps_gpu / price_gpu)
print("Price efficiency over CPU: {}x\nPrice efficiency over GPU: {}x".format(price_efficiency_over_cpu, price_efficiency_over_gpu))

ax.legend([rects1, rects2, rects3], legend_list, facecolor='white', framealpha=1, frameon=False, loc=(args.legend_loc_x, 0.7), fontsize=legend_font, ncol=1)

ax.set_title('{} R@{}={}: {:.2f}x over CPU, {:.2f}x over GPU'.format(
    dbname, topK, int(recall_goal*100), best_qps_fpga/best_qps_cpu, best_qps_fpga/best_qps_gpu), 
    fontsize=label_font)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.0f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=tick_font, horizontalalignment='center', rotation=90)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

best_qps = np.amax([best_qps_fpga, best_qps_gpu, best_qps_cpu])
ax.set(ylim=[0, best_qps * 1.25])

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./CPU_GPU_FPGA_throughput_comparison_fig/{dbname}_R@{topK}={recall}.png'.format(dbname=dbname,topK=topK,recall=int(recall_goal*100)), transparent=False, dpi=200, bbox_inches="tight")
plt.show()
