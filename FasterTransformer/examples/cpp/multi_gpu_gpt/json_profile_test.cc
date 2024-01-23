#include "examples/cpp/multi_gpu_gpt/gpt_estimation_utils.h"

int main(){
    CostEstimator costEstim("../examples/cpp/multi_gpu_gpt/profile/megatron_6.7B_profile.json", 4, 2, 2, 1);

    costEstim.printInfo();
    return 0;
}
