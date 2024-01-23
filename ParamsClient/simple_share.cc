#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "simple_tensor.h"

#define PORT 10040

static const float tensor[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
static float tensor_trans[6];

int main(){
    cudaStream_t     stream;
    cudaStreamCreate(&stream);

    void* devPtr = NULL, *devPtr_trans = NULL;
    lanuchCudaMalloc(&devPtr, 24);
    lanuchCudaMalloc(&devPtr_trans, 24);
    launchCudaMemCpy(devPtr, tensor, 24);

    fastertransformer::check_cuda_error(cudaDeviceSynchronize());
    fastertransformer::invokeMatrixTranspose((float*)devPtr_trans, (float*)devPtr, 2, 3, stream);
    fastertransformer::check_cuda_error(cudaDeviceSynchronize());
    launchCudaMemCpy(tensor_trans, devPtr_trans, 24);

    for(int i = 0; i < 6; i++){
        printf("%.3f ", tensor_trans[i]);
    }

    TestTensorMetaTransit_t tensor_metadata = getTensorMetadata(devPtr_trans, 24);

    // Accept connection
    int server_fd, conn_fd; 
    int opt = 1;
    struct sockaddr_in address; 
    int addrlen = sizeof(address);
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    } 
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) { 
        perror("setsockopt"); 
        exit(EXIT_FAILURE); 
    } 
    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = INADDR_ANY; 
    address.sin_port = htons( PORT ); 
    if (bind(server_fd, (struct sockaddr *)&address,  sizeof(address)) < 0) { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
    if (listen(server_fd, 1) < 0) { 
        perror("listen"); 
        exit(EXIT_FAILURE); 
    } 
    printf("listening..\n");

    while(true){
        if ((conn_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) { 
            perror("accept"); 
            break;
        }
        printf("accepted.\n");

        // Send the packed pointer
        write(conn_fd, (void*)(&tensor_metadata), sizeof(TestTensorMetaTransit_t));

        close(conn_fd);
    }
    
    close(server_fd);

    return 0;
}