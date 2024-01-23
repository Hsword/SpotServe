#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#include "src/kernels/matrix_transpose_kernels.h"
#include "src/utils/cuda_utils.h"

struct TestTensorMetaTransit_t{
    cudaIpcMemHandle_t recv_tensor_handle;
    size_t stoage_size;
    size_t storage_offset;
};

TestTensorMetaTransit_t getTensorMetadata(void* devPtr, size_t size, size_t offset=0);
void lanuchCudaMalloc(void** devPtr, size_t size);
void launchCudaMemCpy(void *dst, const void *src, size_t count);
