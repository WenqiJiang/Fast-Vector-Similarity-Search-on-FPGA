# How to use the bitstreams

We evaluated the bitstream using the XRT version of 2.9.317 (2020.2_PU1).

First, go to the bitstream folder, execute the bitstream with network. For example:

```
cd FPGA-ANNS-with_network_network_general_11_K_100_12B6_6_PEs
./host ./network.xclbin 5120000 8888 10.1.212.155 1 16384 33 1 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF16384,PQ16_12_banks /mnt/scratch/wenqi/saved_npy_data/gnd
```

After 'enqueue user kernel...' shows up, run the client sender. Make sure the client executable is built:

```
cd network_client
make
```

Run the binary. For example:

```
./anns_client_non_blocking 10.1.212.155 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw /mnt/scratch/wenqi/saved_npy_data/gnd 100 SIFT100M_R@100=0.95 5
```

Note that the MTU on the CPU server may need to be set as 552 (512 + 40 byte TCP header) to send the query vectors properly. 

## Commands list

We here provide the lists of commands on the FPGA host server and the CPU client in our experiments. There are 3 dataset size scale x 3 recall goal = 9 combinations. We use the algorithm setting that can achieve the highest performance per combination.

* R@1=0.25, SIFT100M

FPGA:
```
cd FPGA-ANNS-with_network_general_11_K_1_12B_6_PE
./host ./network.xclbin 5120000 8888 10.1.212.155 1 2048 2 0 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_IVF2048,PQ16_12_banks /mnt/scratch/wenqi/saved_npy_data/gnd
```

CPU:

```
./anns_client_non_blocking 10.1.212.155 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw /mnt/scratch/wenqi/saved_npy_data/gnd 1 SIFT100M_R@1=0.25 5
```

* R@1=0.25, SIFT500M

FPGA:

```
cd FPGA-ANNS-with_network_general_11_K_1_12B_6_PE
./host ./network.xclbin 5120000 8888 10.1.212.155 1 4096 2 0 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_IVF4096,PQ16_4_FPGA_12_banks/FPGA_0 /mnt/scratch/wenqi/saved_npy_data/gnd
```

CPU:

```
./anns_client_non_blocking 10.1.212.155 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw /mnt/scratch/wenqi/saved_npy_data/gnd 1 SIFT500M_R@1=0.25 5
```

* R@1=0.25, SIFT1000M

FPGA:

```
cd FPGA-ANNS-with_network_general_11_K_1_12B_6_PE
./host ./network.xclbin 5120000 8888 10.1.212.155 1 4096 3 0 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT1000M_IVF4096,PQ16_8_FPGA_12_banks/FPGA_0 /mnt/scratch/wenqi/saved_npy_data/gnd
```

CPU:

```
./anns_client_non_blocking 10.1.212.155 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw /mnt/scratch/wenqi/saved_npy_data/gnd 1 SIFT1000M_R@1=0.25 5
```

* R10=0.6, SIFT100M



FPGA:

```
cd FPGA-ANNS-with_network_general_11_K_10_12B_4_PE
./host ./network.xclbin 5120000 8888 10.1.212.155 1 2048 3 0 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_IVF2048,PQ16_12_banks /mnt/scratch/wenqi/saved_npy_data/gnd
```

CPU:

```
./anns_client_non_blocking 10.1.212.155 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw /mnt/scratch/wenqi/saved_npy_data/gnd 10 SIFT100M_R@10=0.6 5
```


* R10=0.6, SIFT500M


FPGA:

```
cd FPGA-ANNS-with_network_general_11_K_10_12B_4_PE
./host ./network.xclbin 5120000 8888 10.1.212.155 1 4096 3 0 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_IVF4096,PQ16_4_FPGA_12_banks/FPGA_0 /mnt/scratch/wenqi/saved_npy_data/gnd
```

CPU:

```
./anns_client_non_blocking 10.1.212.155 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw /mnt/scratch/wenqi/saved_npy_data/gnd 10 SIFT500M_R@10=0.6 5
```

* R10=0.6, SIFT1000M

FPGA:

```
cd FPGA-ANNS-with_network_general_11_K_10_12B_4_PE
./host ./network.xclbin 5120000 8888 10.1.212.155 1 4096 3 0 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT1000M_IVF4096,PQ16_8_FPGA_12_banks/FPGA_0 /mnt/scratch/wenqi/saved_npy_data/gnd
```

CPU:

```
./anns_client_non_blocking 10.1.212.155 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw /mnt/scratch/wenqi/saved_npy_data/gnd 10 SIFT1000M_R@10=0.6 5
```

* R@100=0.95, SIFT100M

FPGA:

```
cd FPGA-ANNS-with_network_network_general_11_K_100_12B6_6_PE
./host ./network.xclbin 5120000 8888 10.1.212.155 1 16384 33 1 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF16384,PQ16_12_banks /mnt/scratch/wenqi/saved_npy_data/gnd
```

CPU:

```
./anns_client_non_blocking 10.1.212.155 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw /mnt/scratch/wenqi/saved_npy_data/gnd 100 SIFT100M_R@100=0.95 5
```

* R@100=0.95, SIFT500M

FPGA:

```
cd FPGA-ANNS-with_network_network_general_11_K_100_12B6_6_PE 
./host ./network.xclbin 5120000 8888 10.1.212.155 1 16384 30 1 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_OPQ16,IVF16384,PQ16_4_FPGA_12_banks/FPGA_0 /mnt/scratch/wenqi/saved_npy_data/gnd
```

CPU:

```
./anns_client_non_blocking 10.1.212.155 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw /mnt/scratch/wenqi/saved_npy_data/gnd 100 SIFT500M_R@100=0.95 5
```


* R@100=0.95, SIFT1000M


FPGA:

```
cd FPGA-ANNS-with_network_network_general_11_K_100_12B6_6_PE 
./host ./network.xclbin 5120000 8888 10.1.212.155 1 16384 31 1 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT1000M_OPQ16,IVF16384,PQ16_8_FPGA_12_banks/FPGA_0 /mnt/scratch/wenqi/saved_npy_data/gnd
```

CPU:

```
./anns_client_non_blocking 10.1.212.155 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw /mnt/scratch/wenqi/saved_npy_data/gnd 100 SIFT1000M_R@100=0.95 5
```