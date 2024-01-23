#include <stdio.h>
#include <string.h>

#include "src/utils/memory_utils.h"
#include "src/kernels/matrix_transpose_kernels.h"

using namespace fastertransformer;

// static const float tensor[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
static float tensor_trans[32];

int main(){
    cudaStream_t     stream;
    cudaStreamCreate(&stream);

    float* devPtr = NULL, *devPtr_trans = NULL;
    deviceMalloc(&devPtr, 163840, true);
    deviceMalloc(&devPtr_trans, 163840, false);

    invokeMatrixTransposeInplace((float*)devPtr_trans, (float*)devPtr, 20480, 8, stream);
    // sync_check_cuda_error();
    check_cuda_error(cudaDeviceSynchronize());

    cudaD2Hcpy(tensor_trans, devPtr, 32);

    for(int i = 0; i < 32; i++){
        printf("%.3f ", tensor_trans[i]);
    }

    return 0;
}