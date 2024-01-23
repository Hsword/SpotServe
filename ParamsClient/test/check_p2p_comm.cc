#include "stdio.h"
#include "src/utils/memory_utils.h"
#include "src/utils/nccl_utils.h"

#include <sys/time.h>

using namespace fastertransformer;

constexpr size_t num_float = 1024 * 1024 * 1024; // 4GB
int rank, world_size, device, visible_device_count;
NcclParam nccl_param;

inline void doSendRecv(int send_rank){
    cudaStream_t send_stream, recv_stream;
    for(int i = 0; i < visible_device_count; i++){
        float* dev_ptr;

        if(rank == send_rank){
            check_cuda_error(cudaSetDevice(i));
            cudaStreamCreate(&send_stream);
            deviceMalloc(&dev_ptr, num_float);
        }

        for(int j = 0; j < visible_device_count; j++){
            if(rank != send_rank){
                check_cuda_error(cudaSetDevice(j));
                check_cuda_error(cudaStreamCreate(&recv_stream));
                deviceMalloc(&dev_ptr, num_float);
            }

            struct timeval comm_start, comm_end;

            gettimeofday(&comm_start, NULL);
            if(rank == send_rank){
                ftNcclSend(dev_ptr, num_float, send_rank ^ 1, nccl_param, send_stream);
            }else{
                ftNcclRecv(dev_ptr, num_float, send_rank, nccl_param, recv_stream);
            }
            // sync_check_cuda_error();
            check_cuda_error(cudaDeviceSynchronize());
            gettimeofday(&comm_end, NULL);

            double elapse = (comm_end.tv_sec - comm_start.tv_sec) * 1000 + (comm_end.tv_usec - comm_start.tv_usec) * 0.001;
            printf("%d %d %.3lf %d\n", i, j, elapse, rank);

            if(rank != send_rank){
                deviceFree(dev_ptr);
                check_cuda_error(cudaStreamDestroy(recv_stream));
            }
            
        }

        if(rank == send_rank){
            deviceFree(dev_ptr);
            check_cuda_error(cudaStreamDestroy(send_stream));
        } 
    }
}

int main(int argc, char* argv[]){
    mpi::initialize(&argc, &argv);

    rank       = mpi::getCommWorldRank();
    world_size = mpi::getCommWorldSize();
    check_cuda_error(cudaGetDeviceCount(&visible_device_count));

    printf("Rank %d\n", rank);
    
    ftNcclInitialize(nccl_param, world_size, rank);

    // rank 0 -> rank 1
    doSendRecv(0);
    doSendRecv(1);
    
}