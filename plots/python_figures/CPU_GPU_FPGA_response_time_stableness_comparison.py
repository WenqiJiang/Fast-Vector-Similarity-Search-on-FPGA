# Example usaage: 
#   python CPU_GPU_FPGA_response_time_stableness_comparison.py

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

# parser.add_argument('--perc_50', type=int, default=95, help="x% tail latency, e.g., 95%")


cpu_performance_dict_dir_100M = '../cpu_performance_result_m6i.4xlarge/cpu_response_time_SIFT100M.pkl'
gpu_performance_dict_dir_100M = '../gpu_performance_result_p3.8xlarge_V100/gpu_response_time_SIFT100M.pkl'
cpu_performance_dict_dir_500M = '../cpu_performance_result_m6i.4xlarge/cpu_response_time_SIFT500M.pkl'
gpu_performance_dict_dir_500M = '../gpu_performance_result_p3.8xlarge_V100/gpu_response_time_SIFT500M.pkl'
cpu_performance_dict_dir_1000M = '../cpu_performance_result_m6i.4xlarge/cpu_response_time_SIFT1000M.pkl'
gpu_performance_dict_dir_1000M = '../gpu_performance_result_p3.8xlarge_V100/gpu_response_time_SIFT1000M.pkl'

batch_size=1

def get_tail_latency(RT_array, perc_50):

    RT_array = np.array(RT_array)
    RT_array_sorted = np.sort(RT_array)
    len_array = RT_array.shape[0]

    return RT_array_sorted[int(perc_50 / 100.0 * len_array)]


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

cpu_latency_array = [
    get_cpu_RT_array(d_cpu_100M, 'SIFT100M', 'IVF16384,PQ16', 1, 0.25),
    get_cpu_RT_array(d_cpu_100M, 'SIFT100M', 'IVF16384,PQ16', 10, 0.6),
    get_cpu_RT_array(d_cpu_100M, 'SIFT100M', 'OPQ16,IVF16384,PQ16', 100, 0.95),
    get_cpu_RT_array(d_cpu_500M, 'SIFT500M', 'IVF32768,PQ16', 1, 0.25),
    get_cpu_RT_array(d_cpu_500M, 'SIFT500M', 'OPQ16,IVF16384,PQ16', 10, 0.6),
    get_cpu_RT_array(d_cpu_500M, 'SIFT500M', 'OPQ16,IVF65536,PQ16', 100, 0.95),
    get_cpu_RT_array(d_cpu_1000M, 'SIFT1000M', 'OPQ16,IVF32768,PQ16', 1, 0.25), 
    get_cpu_RT_array(d_cpu_1000M, 'SIFT1000M', 'IVF32768,PQ16', 10, 0.6),
    get_cpu_RT_array(d_cpu_1000M, 'SIFT1000M', 'OPQ16,IVF65536,PQ16', 100, 0.95)]

gpu_latency_array = [
    get_gpu_RT_array(d_gpu_100M, 'SIFT100M', 'IVF16384,PQ16', 1, 0.25), 
    get_gpu_RT_array(d_gpu_100M, 'SIFT100M', 'IVF32768,PQ16', 10, 0.6), 
    get_gpu_RT_array(d_gpu_100M, 'SIFT100M', 'IVF32768,PQ16', 100, 0.95), 
    get_gpu_RT_array(d_gpu_500M, 'SIFT500M', 'IVF16384,PQ16', 1, 0.25), 
    get_gpu_RT_array(d_gpu_500M, 'SIFT500M', 'IVF16384,PQ16', 10, 0.6), 
    get_gpu_RT_array(d_gpu_500M, 'SIFT500M', 'IVF65536,PQ16', 100, 0.95), 
    get_gpu_RT_array(d_gpu_1000M, 'SIFT1000M', 'OPQ16,IVF16384,PQ16', 1, 0.25), 
    get_gpu_RT_array(d_gpu_1000M, 'SIFT1000M', 'IVF32768,PQ16', 10, 0.6), 
    get_gpu_RT_array(d_gpu_1000M, 'SIFT1000M', 'OPQ16,IVF32768,PQ16', 100, 0.95)]

fpga_latency_array = [
    np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT100M_R@1=0.25', dtype=np.float32), # 100M, K=1
    np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT100M_R@10=0.6', dtype=np.float32), # 100M, K=10
    np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT100M_R@100=0.95', dtype=np.float32), # 100M, K=100

    np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT500M_R@1=0.25', dtype=np.float32), # 500M, K=1
    np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT500M_R@10=0.6', dtype=np.float32), # 500M, K=10
    np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT500M_R@100=0.95', dtype=np.float32),# 500M, K=100

    np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT1000M_R@1=0.25', dtype=np.float32), # 1000M, K=1
    np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT1000M_R@10=0.6', dtype=np.float32), # 1000M, K=10
    np.fromfile('../fpga_performance_result/RT_distribution_non_block_gap_5/SIFT1000M_R@100=0.95', dtype=np.float32) # 1000M, K=100
]


y_cpu_50 = np.array([get_tail_latency(a, 50) for a in cpu_latency_array])
y_cpu_95 = np.array([get_tail_latency(a, 95) for a in cpu_latency_array])

y_gpu_50 = np.array([get_tail_latency(a, 50) for a in gpu_latency_array])
y_gpu_95 = np.array([get_tail_latency(a, 95) for a in gpu_latency_array])

y_fpga_50 = np.array([get_tail_latency(a, 50) for a in fpga_latency_array])
y_fpga_95 = np.array([get_tail_latency(a, 95) for a in fpga_latency_array])

stable_ratio_cpu = y_cpu_95 / y_cpu_50
stable_ratio_gpu = y_gpu_95 / y_gpu_50
stable_ratio_fpga = y_fpga_95 / y_fpga_50

# variance 
var_cpu = np.var(cpu_latency_array)
var_gpu = np.var(gpu_latency_array)
var_fpga = np.var(fpga_latency_array)

print("CPU variance: {}".format(var_cpu))
print("GPU variance: {}".format(var_gpu))
print("FPGA variance: {}".format(var_fpga))

print("95'%' / 50'%' latency ratio CPU: {}, \nmin: {}\t max: {}\t ave: {}".format(stable_ratio_cpu, np.amin(stable_ratio_cpu), np.amax(stable_ratio_cpu), np.average(stable_ratio_cpu)))
print("95'%' / 50'%' latency ratio GPU: {}, \nmin: {}\t max: {}\t ave: {}".format(stable_ratio_gpu, np.amin(stable_ratio_gpu), np.amax(stable_ratio_gpu), np.average(stable_ratio_gpu)))
print("95'%' / 50'%' latency ratio FPGA: {}, \nmin: {}\t max: {}\t ave: {}".format(stable_ratio_fpga, np.amin(stable_ratio_fpga), np.amax(stable_ratio_fpga), np.average(stable_ratio_fpga)))
