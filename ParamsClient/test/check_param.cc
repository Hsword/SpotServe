#include<stdio.h>
#include<string>

#include "src/utils/memory_utils.h"
#include "src/kernels/matrix_transpose_kernels.h"

using namespace fastertransformer;

const std::string file_root = "/home/sj/sca/FasterTransformer/models/megatron-models/c-model/345m/1-gpu/model.layers.0";

int main(){
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t max_seq_len = 1024, vocab_size = 50304, hidden_size = 1024;
    float *devPtr_1, *devPtr_2, *devPtr;

    deviceMalloc(&devPtr_1, hidden_size * 3 * hidden_size, false);
    deviceMalloc(&devPtr_2, hidden_size * 3 * hidden_size, false);

    loadWeightFromBin<float>(devPtr_1, {hidden_size * 3, hidden_size}, file_root + ".attention.query_key_value.weight.0.bin");
    printf("1-gpu from 0:\n");
    print_to_screen(devPtr_1, 10);
    printf("1-gpu from 512:\n");
    print_to_screen(devPtr_1 + 512, 10);

    printf("------trans------\n");
    invokeMatrixTransposeInplace(devPtr_2, devPtr_1, hidden_size * 3, hidden_size, stream);
    printf("1-gpu from 0:\n");
    print_to_screen(devPtr_1, 10);
    printf("1-gpu from %d:\n", hidden_size * hidden_size * 3 / 2);
    print_to_screen(devPtr_1 + hidden_size * hidden_size * 3 / 2, 10);

    printf("------trans------\n");
    invokeMatrixTransposeInplace(devPtr_2, devPtr_1, hidden_size, hidden_size * 3, stream);
    printf("1-gpu from 0:\n");
    print_to_screen(devPtr_1, 10);
    printf("1-gpu from 512:\n");
    print_to_screen(devPtr_1 + 512, 10);
}