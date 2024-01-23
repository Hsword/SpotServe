#include "simple_tensor.h"
#include <cuda_runtime.h>

TestTensorMetaTransit_t getTensorMetadata(void* devPtr, size_t size, size_t offset){

    TestTensorMetaTransit_t tensor_metadata;
    tensor_metadata.stoage_size = size;
    tensor_metadata.storage_offset = 0;
    fastertransformer::check_cuda_error(cudaIpcGetMemHandle(&tensor_metadata.recv_tensor_handle, devPtr));
    return tensor_metadata;
}

void lanuchCudaMalloc(void** devPtr, size_t size){
    fastertransformer::check_cuda_error(cudaMalloc(devPtr, size));
}

void launchCudaMemCpy(void *dst, const void *src, size_t count){
    fastertransformer::check_cuda_error(cudaMemcpy(dst, src, count, cudaMemcpyDefault));
}