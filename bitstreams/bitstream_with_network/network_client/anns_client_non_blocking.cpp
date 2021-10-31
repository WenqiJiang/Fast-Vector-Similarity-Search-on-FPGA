// Client side C/C++ program to demonstrate Socket programming 
#include <stdio.h> 
#include <stdlib.h> 
#include <stdint.h>
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <unistd.h> 
#include <string.h> 
#include <unistd.h>
#include <time.h>
#include <pthread.h> 
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

#define QUERY_NUM 10000
#define QUERY_SIZE 512 // 128 D float vector 
#define RESULT_SIZE 128 // 10 * (float + int) = 80 bytes + padding = 128 bytes

timespec diff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

struct Thread_info {
    char* IP; // server
    int port;
    int query_num;
    int send_recv_gap; // e.g., send is only allow to send 5 queries before recv

    // copy query_num queries from the total 10000 queries
    char* global_query_vector_buf;
    char* global_result_buf;
    timespec* query_start_time_array;
    timespec* query_end_time_array;
};

typedef struct Result{
    int vec_ID;
    float dist; 
} result_t;


int topK;
int result_size;

bool start_receiving = false;
double duration_ms; // transmission duration in milliseconds

int sock = 0;

int send_query_id = -1;
int receive_query_id = -1;

void *thread_send_queries(void* vargp) 
{ 
    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing from send thread...\n"); 
    
    const int query_num = t_info -> query_num;

    char* global_query_vector_buf = t_info -> global_query_vector_buf;

    timespec* query_start_time_array = t_info -> query_start_time_array;

    int send_recv_gap = t_info -> send_recv_gap; 

    struct sockaddr_in serv_addr; 

    char* send_buf = new char[QUERY_SIZE * query_num];

    memcpy(send_buf, global_query_vector_buf, QUERY_SIZE * query_num);

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return 0; 
    } 
   
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(t_info -> port); 
       
    if(inet_pton(AF_INET, t_info -> IP, &serv_addr.sin_addr)<=0)  
    { 
        printf("\nInvalid address/ Address not supported \n"); 
        return 0; 
    } 
   
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr))<0) 
    { 
        printf("\nConnection Failed \n"); 
        return 0; 
    } 

    printf("Start sending data.\n");
    start_receiving = true;

    int total_sent_bytes = 0;
    ////////////////   Data transfer   ////////////////
    // clock_t start = clock();

	timespec start, end;
	clock_gettime(CLOCK_BOOTTIME, &start);

    for (int query_id = 0; query_id < query_num; query_id++) {

        send_query_id = query_id;
        std::cout << "sending thread query id: " << send_query_id << std::endl;

        volatile int tmp_counter;
        do {
            // wait
            tmp_counter++;
        } while(send_query_id - receive_query_id >= send_recv_gap);

	    int current_query_sent_bytes = 0;

	    clock_gettime(CLOCK_BOOTTIME, &query_start_time_array[query_id]);

        while (current_query_sent_bytes < QUERY_SIZE) {
            int sent_bytes = send(sock, send_buf + total_sent_bytes, QUERY_SIZE - current_query_sent_bytes, 0);
            total_sent_bytes += sent_bytes;
            current_query_sent_bytes += sent_bytes;
            if (sent_bytes == -1) {
                printf("Sending data UNSUCCESSFUL!\n");
                return 0;
            } 
#ifdef DEBUG
            else {
                printf("total sent bytes = %d\n", total_sent_bytes);
            }
#endif
        }
    }

    if (total_sent_bytes != query_num * QUERY_SIZE) {
        printf("Sending error, sending more bytes than a block\n");
    }
    else {
	printf("Finish sending\n");
    }

	clock_gettime(CLOCK_BOOTTIME, &end);

    timespec total_time = diff(start, end);

    float total_time_ms = 
        ((float) total_time.tv_sec) * 1000.0 + 
        ((float) total_time.tv_nsec) / 1000.0 / 1000.0;
    printf("\nSend thread duration: %f ms\n", total_time_ms);
    float QPS = 10000.0 / (total_time_ms / 1000.0);
    printf("Send thread QPS: %f\n", QPS);

    return NULL; 
} 


void *thread_receive_results(void* vargp) 
{ 
    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing from receive thread...\n"); 
    
    const int query_num = t_info -> query_num;
    char* global_result_buf = t_info -> global_result_buf;
    timespec* query_end_time_array = t_info -> query_end_time_array;

    char* recv_buf= new char[result_size * query_num];

    volatile int tmp_counter;
    do {
        // nothing
	    tmp_counter++;
    } while(!start_receiving);
    printf("Start receiving data.\n");

    int total_recv_bytes = 0;

    ////////////////   Data transfer   ////////////////
    

	timespec start, end;
	clock_gettime(CLOCK_BOOTTIME, &start);

    for (int query_id = 0; query_id < query_num; query_id++) {

        receive_query_id = query_id;
        std::cout << "receiving thread query id: " << receive_query_id << std::endl;
        
        volatile int tmp_counter;
        do {
            // wait
            tmp_counter++;
        } while(send_query_id < receive_query_id);

        int current_query_recv_bytes = 0;

        while (current_query_recv_bytes < result_size) {
    	    int recv_bytes = recv(sock, recv_buf + total_recv_bytes, result_size - current_query_recv_bytes, 0);
            total_recv_bytes += recv_bytes;
            current_query_recv_bytes += total_recv_bytes;
            if (recv_bytes == -1) {
                printf("Receiving data UNSUCCESSFUL!\n");
                return 0;
            }
#ifdef DEBUG
            else {
                printf("totol received bytes: %d\n", total_recv_bytes);
            }
#endif
        }

	    clock_gettime(CLOCK_BOOTTIME, &query_end_time_array[query_id]);
    }

    if (total_recv_bytes != query_num * result_size) {
        printf("Receiving error, receiving more bytes than a block\n");
    }
    else {
	printf("Finish receiving\n");
    }

	clock_gettime(CLOCK_BOOTTIME, &end);

    timespec total_time = diff(start, end);

    float total_time_ms = 
        ((float) total_time.tv_sec) * 1000.0 + 
        ((float) total_time.tv_nsec) / 1000.0 / 1000.0;
    printf("\nReceive thread duration: %f ms\n", total_time_ms);
    float QPS = 10000.0 / (total_time_ms / 1000.0);
    printf("Receive thread QPS: %f\n", QPS);

    memcpy(global_result_buf, recv_buf, query_num * result_size);

    return NULL; 
} 


// boost::filesystem does not compile well, so implement this myself
std::string dir_concat(std::string dir1, std::string dir2) {
    if (dir1.back() != '/') {
        dir1 += '/';
    }
    return dir1 + dir2;
}

int main(int argc, char *argv[]) 
{ 

    if (argc < 6 || argc > 8) {
        // <data directory> is only used for loading query vector, can be any folder containing queries
        // <send_recv_gap> denotes the sender can be , e.g. x=5, queries in front of receiver, cannot be 100 queries in front which will influence performance
        printf("Usage: <executable> <IP> <port> <query vector dir> <ground truth dir> <topK> <optional RT_file_name> <optionnal send_recv_gap> , e.g., ./anns_client_non_blocking 10.1.212.155 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw /mnt/scratch/wenqi/saved_npy_data/gnd 100 'SIFT100M_R@100=0.95' 5\n");
        exit(1);
    }
    std::string s_IP = argv[1];
    int n = s_IP.length();
    char IP[n + 1];
    strcpy(IP, s_IP.c_str());

    int port = std::stoi(argv[2]);
    std::string query_vector_path = argv[3];
    std::string gnd_dir = argv[4];

    topK = std::stoi(argv[5]);
    std::string RT_file_name;
    if (argc >= 7) {
        RT_file_name = argv[6];
    }
    else {
        RT_file_name = "RT_distribution";
    }

    int send_recv_gap; 
    if (argc >= 8) {
        send_recv_gap = std::stoi(argv[7]);
    }
    else {
        send_recv_gap = 5; // default value
    }

    printf("server IP: %s, port: %d\n", IP, port);

    if (topK == 1) {
         // 1 single 512-bit packet
        result_size = 1 * 64;
        std::cout << "result size (per query) = " << result_size << " bytes" << std::endl;
    }
    if (topK == 10) {
         // 2 512-bit packets
        result_size = 2 * 64;
        std::cout << "result size (per query) = " << result_size << " bytes" << std::endl;
    }
    if (topK == 100) {
         // 13 512-bit packets
         // 100 * 8 byte per result = 800 bytes, ceil(800 / 64) = 13
        result_size = 13 * 64;
        std::cout << "result size (per query) = " << result_size << " bytes" << std::endl;
    }

    size_t query_vector_size = QUERY_NUM * QUERY_SIZE;
    char* global_query_vector_buf = new char[query_vector_size];
    char* global_result_buf = new char[QUERY_NUM * result_size];

    timespec* query_start_time_array = new timespec[QUERY_NUM];
    timespec* query_end_time_array = new timespec[QUERY_NUM];

    float* global_RT_ms_buf = new float[QUERY_NUM];

    // Load query vectors
    std::ifstream query_vector_fstream(
        query_vector_path,
        std::ios::in | std::ios::binary);
    query_vector_fstream.read(global_query_vector_buf, query_vector_size);
    if (!query_vector_fstream) {
        std::cout << "error: only " << query_vector_fstream.gcount() << " could be read";
        exit(1);
    }

    // Load ground truth
    // the raw ground truth size is the same for idx_1M.ivecs, idx_10M.ivecs, idx_100M.ivecs
    size_t raw_gt_vec_ID_len = 10000 * 1001; 
    size_t raw_gt_vec_ID_size = raw_gt_vec_ID_len * sizeof(int);
    std::vector<int> raw_gt_vec_ID(raw_gt_vec_ID_len, 0);

    std::string raw_gt_vec_ID_suffix_dir = "idx_100M.ivecs";
    std::string raw_gt_vec_ID_dir = dir_concat(gnd_dir, raw_gt_vec_ID_suffix_dir);
    std::ifstream raw_gt_vec_ID_fstream(
        raw_gt_vec_ID_dir,
        std::ios::in | std::ios::binary);
    if (!raw_gt_vec_ID_fstream) {
        std::cout << "error: only " << raw_gt_vec_ID_fstream.gcount() << " could be read";
        exit(1);
    }
    char* raw_gt_vec_ID_char = (char*) malloc(raw_gt_vec_ID_size);
    raw_gt_vec_ID_fstream.read(raw_gt_vec_ID_char, raw_gt_vec_ID_size);
    if (!raw_gt_vec_ID_fstream) {
        std::cout << "error: only " << raw_gt_vec_ID_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&raw_gt_vec_ID[0], raw_gt_vec_ID_char, raw_gt_vec_ID_size);
    free(raw_gt_vec_ID_char);
    size_t gt_vec_ID_len = 10000;
    std::vector<int> gt_vec_ID(gt_vec_ID_len, 0);
    // copy contents from raw ground truth to needed ones
    // Format of ground truth (for 10000 query vectors):
    //   1000(topK), [1000 ids]
    //   1000(topK), [1000 ids]
    //        ...     ...
    //   1000(topK), [1000 ids]
    // 10000 rows in total, 10000 * 1001 elements, 10000 * 1001 * 4 bytes
    for (int i = 0; i < 10000; i++) {
        gt_vec_ID[i] = raw_gt_vec_ID[i * 1001 + 1];
    }


    std::cout << "finish loading" << std::endl;

    pthread_t thread_send; 
    pthread_t thread_recv; 

    struct Thread_info t_info;

    t_info.IP = IP;
    t_info.port = port;
    t_info.query_num = QUERY_NUM;
    t_info.send_recv_gap = send_recv_gap;

    t_info.global_query_vector_buf = global_query_vector_buf;
    t_info.global_result_buf = global_result_buf;

    t_info.query_start_time_array = query_start_time_array;
    t_info.query_end_time_array = query_end_time_array;
        
    printf("Before Thread\n");
    pthread_create(&thread_send, NULL, thread_send_queries, (void*) &t_info); 
    sleep(0.1);
    pthread_create(&thread_recv, NULL, thread_receive_results, (void*) &t_info); 

    pthread_join(thread_send, NULL); 
    pthread_join(thread_recv, NULL); 
    printf("After Thread\n"); 

    for (int query_id = 0; query_id < QUERY_NUM; query_id++) {
        timespec diff_RT = diff(query_start_time_array[query_id], query_end_time_array[query_id]);
        global_RT_ms_buf[query_id] = 
            ((float) diff_RT.tv_sec) * 1000.0 + 
            ((float) diff_RT.tv_nsec) / 1000.0 / 1000.0;
    }

    // Save RT distribution
    std::string RT_distribution = 
        "./RT_distribution/" + RT_file_name;
    int char_len = RT_distribution.length();
    char RT_distribution_char[char_len + 1];
    strcpy(RT_distribution_char, RT_distribution.c_str());
    FILE *file = fopen(RT_distribution_char, "w");
    fwrite(global_RT_ms_buf, sizeof(float), QUERY_NUM, file);
    fclose(file);

    return 0; 
} 
