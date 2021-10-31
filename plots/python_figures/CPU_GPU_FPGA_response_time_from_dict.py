# Example usaage: 
#   python CPU_GPU_FPGA_response_time_from_dict.py --perc 95

# Run with python 3.9 if the following error occurs
# WenqideMacBook-Pro@~/Works/ANNS-FPGA/python_figures wenqi$python CPU_GPU_FPGA_response_time_from_dict.py 
# Traceback (most recent call last):
#   File "CPU_GPU_FPGA_response_time_from_dict.py", line 112, in <module>
#     fpga_cpu = pickle.load(f)
# ValueError: unsupported pickle protocol: 5

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

import argparse 
parser = argparse.ArgumentParser()

parser.add_argument('--perc', type=int, default=95, help="x% tail latency, e.g., 95%")

args = parser.parse_args()
perc = args.perc

cpu_performance_dict_dir_100M = '../cpu_performance_result_m6i.4xlarge/cpu_response_time_SIFT100M.pkl'
gpu_performance_dict_dir_100M = '../gpu_performance_result_p3.8xlarge_V100/gpu_response_time_SIFT100M.pkl'
cpu_performance_dict_dir_500M = '../cpu_performance_result_m6i.4xlarge/cpu_response_time_SIFT500M.pkl'
gpu_performance_dict_dir_500M = '../gpu_performance_result_p3.8xlarge_V100/gpu_response_time_SIFT500M.pkl'
cpu_performance_dict_dir_1000M = '../cpu_performance_result_m6i.4xlarge/cpu_response_time_SIFT1000M.pkl'
gpu_performance_dict_dir_1000M = '../gpu_performance_result_p3.8xlarge_V100/gpu_response_time_SIFT1000M.pkl'

batch_size=1

def get_tail_latency(RT_array, perc):

    RT_array = np.array(RT_array)
    RT_array_sorted = np.sort(RT_array)
    len_array = RT_array.shape[0]

    return RT_array_sorted[int(perc / 100.0 * len_array)]


def get_cpu_RT_array(d, dbname, index_key, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = response_time 
    """
    return d[dbname][index_key][topK][recall_goal]


def get_gpu_RT_array(d, dbname, index_key, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = response_time (QPS)
    """
    return d[dbname][index_key][topK][recall_goal][batch_size]


d_cpu_100M = None
with open(cpu_performance_dict_dir_100M, 'rb') as f:
    d_cpu_100M = pickle.load(f)
d_gpu_100M = None
with open(gpu_performance_dict_dir_100M, 'rb') as f:
    d_gpu_100M = pickle.load(f)

d_cpu_500M = None
with open(cpu_performance_dict_dir_500M, 'rb') as f:
    d_cpu_500M = pickle.load(f)
d_gpu_500M = None
with open(gpu_performance_dict_dir_500M, 'rb') as f:
    d_gpu_500M = pickle.load(f)

d_cpu_1000M = None
with open(cpu_performance_dict_dir_1000M, 'rb') as f:
    d_cpu_1000M = pickle.load(f)
d_gpu_1000M = None
with open(gpu_performance_dict_dir_1000M, 'rb') as f:
    d_gpu_1000M = pickle.load(f)

x_labels = ['SIFT100M\nR@1=25%', 'SIFT100M\nR@10=60%', 'SIFT100M\nR@100=95%', \
    'SIFT500M\nR@1=25%', 'SIFT500M\nR@10=60%', 'SIFT500M\nR@100=95%', \
    'SIFT1000M\nR@1=25%', 'SIFT1000M\nR@10=60%', 'SIFT1000M\nR@100=95%']

y_cpu = [
    get_tail_latency(get_cpu_RT_array(d_cpu_100M, 'SIFT100M', 'IVF16384,PQ16', 1, 0.25), perc), 
    get_tail_latency(get_cpu_RT_array(d_cpu_100M, 'SIFT100M', 'IVF16384,PQ16', 10, 0.6), perc), 
    get_tail_latency(get_cpu_RT_array(d_cpu_100M, 'SIFT100M', 'OPQ16,IVF16384,PQ16', 100, 0.95), perc), 
    get_tail_latency(get_cpu_RT_array(d_cpu_500M, 'SIFT500M', 'IVF32768,PQ16', 1, 0.25), perc), 
    get_tail_latency(get_cpu_RT_array(d_cpu_500M, 'SIFT500M', 'OPQ16,IVF16384,PQ16', 10, 0.6), perc), 
    get_tail_latency(get_cpu_RT_array(d_cpu_500M, 'SIFT500M', 'OPQ16,IVF65536,PQ16', 100, 0.95), perc),  
    get_tail_latency(get_cpu_RT_array(d_cpu_1000M, 'SIFT1000M', 'OPQ16,IVF32768,PQ16', 1, 0.25), perc), 
    get_tail_latency(get_cpu_RT_array(d_cpu_1000M, 'SIFT1000M', 'IVF32768,PQ16', 10, 0.6), perc),  
    get_tail_latency(get_cpu_RT_array(d_cpu_1000M, 'SIFT1000M', 'OPQ16,IVF65536,PQ16', 100, 0.95), perc)]

y_gpu = [
    get_tail_latency(get_gpu_RT_array(d_gpu_100M, 'SIFT100M', 'IVF16384,PQ16', 1, 0.25), perc), 
    get_tail_latency(get_gpu_RT_array(d_gpu_100M, 'SIFT100M', 'IVF32768,PQ16', 10, 0.6), perc), 
    get_tail_latency(get_gpu_RT_array(d_gpu_100M, 'SIFT100M', 'IVF32768,PQ16', 100, 0.95), perc), 
    get_tail_latency(get_gpu_RT_array(d_gpu_500M, 'SIFT500M', 'IVF16384,PQ16', 1, 0.25), perc), 
    get_tail_latency(get_gpu_RT_array(d_gpu_500M, 'SIFT500M', 'IVF16384,PQ16', 10, 0.6), perc), 
    get_tail_latency(get_gpu_RT_array(d_gpu_500M, 'SIFT500M', 'IVF65536,PQ16', 100, 0.95), perc), 
    get_tail_latency(get_gpu_RT_array(d_gpu_1000M, 'SIFT1000M', 'OPQ16,IVF16384,PQ16', 1, 0.25), perc), 
    get_tail_latency(get_gpu_RT_array(d_gpu_1000M, 'SIFT1000M', 'IVF32768,PQ16', 10, 0.6), perc), 
    get_tail_latency(get_gpu_RT_array(d_gpu_1000M, 'SIFT1000M', 'OPQ16,IVF32768,PQ16', 100, 0.95), perc)]

# y_fpga = [
#     get_tail_latency(np.fromfile('../fpga_performance_result/VLDB_RT/SIFT100M_R@1=0.25_RT_distribution_10000_queries_nlist_2048_nprobe_2_thread_num_5', dtype=np.float32), perc),  # 100M, K=1
#     get_tail_latency(np.fromfile('../fpga_performance_result/VLDB_RT/SIFT100M_R@10=0.6_RT_distribution_10000_queries_nlist_2048_nprobe_3_thread_num_4', dtype=np.float32), perc),  # 100M, K=10
#     get_tail_latency(np.fromfile('../fpga_performance_result/VLDB_RT/SIFT100M_R@100=0.95_RT_distribution_10000_queries_nlist_16384_nprobe_33_OPQ_thread_num_4', dtype=np.float32), perc),  # 100M, K=100

#     get_tail_latency(np.fromfile('../fpga_performance_result/VLDB_RT/SIFT500M_R@1=0.25_RT_distribution_10000_queries_nlist_4096_nprobe_2_thread_num_5', dtype=np.float32), perc),  # 500M, K=1
#     get_tail_latency(np.fromfile('../fpga_performance_result/VLDB_RT/SIFT500M_R@10=0.6_RT_distribution_10000_queries_nlist_4096_nprobe_3_thread_num_4', dtype=np.float32), perc),  # 500M, K=10
#     get_tail_latency(np.fromfile('../fpga_performance_result/VLDB_RT/SIFT500M_R@100=0.95_RT_distribution_10000_queries_nlist_16384_nprobe_30_OPQ_thread_num_4', dtype=np.float32), perc), # 500M, K=100

#     get_tail_latency(np.fromfile('../fpga_performance_result/VLDB_RT/SIFT1000M_R@1=0.25_RT_distribution_10000_queries_nlist_4096_nprobe_3_thread_num_5', dtype=np.float32), perc),  # 1000M, K=1
#     get_tail_latency(np.fromfile('../fpga_performance_result/VLDB_RT/SIFT1000M_R@10=0.6_RT_distribution_10000_queries_nlist_4096_nprobe_3_thread_num_4', dtype=np.float32), perc),  # 1000M, K=10
#     get_tail_latency(np.fromfile('../fpga_performance_result/VLDB_RT/SIFT1000M_R@100=0.95_RT_distribution_10000_queries_nlist_16384_nprobe_31_OPQ_thread_num_4', dtype=np.float32), perc) # 1000M, K=100
# ]

y_fpga = [
    get_tail_latency(np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT100M_R@1=0.25', dtype=np.float32), perc),  # 100M, K=1
    get_tail_latency(np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT100M_R@10=0.6', dtype=np.float32), perc),  # 100M, K=10
    get_tail_latency(np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT100M_R@100=0.95', dtype=np.float32), perc),  # 100M, K=100

    get_tail_latency(np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT500M_R@1=0.25', dtype=np.float32), perc),  # 500M, K=1
    get_tail_latency(np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT500M_R@10=0.6', dtype=np.float32), perc),  # 500M, K=10
    get_tail_latency(np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT500M_R@100=0.95', dtype=np.float32), perc), # 500M, K=100

    get_tail_latency(np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT1000M_R@1=0.25', dtype=np.float32), perc),  # 1000M, K=1
    get_tail_latency(np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT1000M_R@10=0.6', dtype=np.float32), perc),  # 1000M, K=10
    get_tail_latency(np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT1000M_R@100=0.95', dtype=np.float32), perc) # 1000M, K=100
]

x = np.arange(len(x_labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
# 
rects1  = ax.bar(x - width, y_cpu, width)#, label='Men')
rects2  = ax.bar(x, y_gpu, width)#, label='Men')
rects3 = ax.bar(x + width, y_fpga, width)#, label='Women')

label_font = 12
tick_font = 10
tick_label_font = 10
legend_font = 10
title_font = 14

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('{perc}% Tail Latency (ms)'.format(perc=perc), fontsize=label_font)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
plt.xticks(rotation=45  )

ax.legend([rects1, rects2, rects3], ["CPU", "GPU", "FPGA"], facecolor='white', framealpha=1, frameon=False, loc=(0.02, 0.6), fontsize=legend_font, ncol=1)

# ax.set_title('{} R@{}={}: {:.2f}x over CPU, {:.2f}x over GPU'.format(
#     dbname, topK, int(recall_goal*100), best_qps_fpga/best_qps_cpu, best_qps_fpga/best_qps_gpu), 
#     fontsize=label_font)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=tick_font, horizontalalignment='center', rotation=90)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.set(ylim=[0, 1.25 * np.amax(y_cpu)])

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./CPU_GPU_FPGA_response_time_comparison_{perc}_perc_tail_latency.png'.format(perc=perc), transparent=False, dpi=200, bbox_inches="tight")
plt.show()
