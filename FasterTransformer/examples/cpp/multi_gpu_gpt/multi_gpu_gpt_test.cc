#include "examples/cpp/multi_gpu_gpt/gpt_example_utils.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

#include <unistd.h> 
#include <sys/socket.h> 
#include <netinet/in.h> 
#include <arpa/inet.h> 

#define PORT 10052

using namespace fastertransformer;

static void* shared_ptr = NULL;

int main(){
    // Connect
    int conn_fd = 0; 
    struct sockaddr_in serv_addr; 
    if ((conn_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) { 
        printf("\n Socket creation error \n"); 
        exit(EXIT_FAILURE); 
    }
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(PORT); 
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) { 
        printf("\nInvalid address/ Address not supported \n"); 
        exit(EXIT_FAILURE); 
    }
    if (connect(conn_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) { 
        printf("\nConnection Failed \n"); 
        exit(EXIT_FAILURE); 
    }

    TensorMetaTransit_t shared_tensor_metadata;
    read(conn_fd, (void*)(&shared_tensor_metadata), sizeof(TensorMetaTransit_t)); 

    extractSharedPtr(&shared_ptr, shared_tensor_metadata);

    printf("storage size: %d\n", shared_tensor_metadata.stoage_size);
    float* host_ptr = (float*)malloc(shared_tensor_metadata.stoage_size);
    cudaAutoCpy(host_ptr, (float*)shared_ptr, shared_tensor_metadata.stoage_size / sizeof(float), NULL);

    for(int i = 0; i < 6; i++){
        printf("%.3f ", host_ptr[i]);
    }
    printf("\n");

    free(host_ptr);
    cudaIpcCloseMemHandle(shared_ptr);

    printf("Hello world\n");
}