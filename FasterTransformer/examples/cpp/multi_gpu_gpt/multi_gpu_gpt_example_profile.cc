/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "3rdparty/INIReader.h"
#include "examples/cpp/multi_gpu_gpt/gpt_example_utils.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"
#include "src/fastertransformer/utils/tcp_utils.h"

#include <cuda_profiler_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>
#include <algorithm>

#ifdef USE_NVTX
bool NVTX_ON = true;
#endif

#define BASE_PORT 10051

using namespace fastertransformer;

template<typename T>
void multi_gpu_gpt_example(const INIReader reader, std::string in_csv, int mini_bsz, int tp_deg, int pp_deg, bool tp_comm);

int main(int argc, char* argv[])
{
    FT_LOG_INFO("entered main");
    mpi::initialize(&argc, &argv);
    srand(0);

    int mini_bsz = 0, tp_deg = 0, pp_deg = 0;
    int tp_comm = 1;
    if(argc >= 2) mini_bsz = atoi(argv[1]);
    if(argc >= 3) tp_deg = atoi(argv[2]);
    if(argc >= 4) pp_deg = atoi(argv[3]);
    if(argc >= 5) tp_comm = atoi(argv[4]);

    std::string ini_name;
    if (argc >= 6) {
        ini_name = std::string(argv[5]);
    }
    else {
        ini_name = "../examples/cpp/multi_gpu_gpt/gpt_config.ini";
    }

    std::string in_csv;
    if (argc == 7) {
        in_csv = std::string(argv[6]);
    }
    else {
        in_csv = "../examples/cpp/multi_gpu_gpt/start_ids.csv";
    }

    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        return -1;
    }
    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");

    if (data_type == "fp32") {
        multi_gpu_gpt_example<float>(reader, in_csv, mini_bsz, tp_deg, pp_deg, (bool)tp_comm);
    }
    else if (data_type == "fp16") {
        multi_gpu_gpt_example<half>(reader, in_csv, mini_bsz, tp_deg, pp_deg, (bool)tp_comm);
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        multi_gpu_gpt_example<__nv_bfloat16>(reader, in_csv, mini_bsz, tp_deg, pp_deg);
    }
#endif
    else {
        printf("[ERROR] data_type should be fp32, fp16 or bf16 ! \n");
        return -1;
    }
    mpi::finalize();
    return 0;
}

template<typename T>
void multi_gpu_gpt_example(const INIReader reader, std::string in_csv, int mini_bsz, int tp_deg, int pp_deg, bool tp_comm)
{
    FT_LOG_INFO("mpi inited");

    const std::string model_name         = reader.Get("ft_instance_hyperparameter", "model_name");
    const size_t      max_batch_size     = (size_t)reader.GetInteger("ft_instance_hyperparameter", "max_batch_size");
    const size_t      max_seq_len        = (size_t)reader.GetInteger("ft_instance_hyperparameter", "max_seq_len");
    const size_t      beam_width         = (size_t)reader.GetInteger("ft_instance_hyperparameter", "beam_width");
    const uint        top_k              = (uint)reader.GetInteger("ft_instance_hyperparameter", "top_k");
    const float       top_p              = reader.GetFloat("ft_instance_hyperparameter", "top_p");
    const float       temperature        = reader.GetFloat("ft_instance_hyperparameter", "temperature");
    const float       repetition_penalty = reader.GetFloat("ft_instance_hyperparameter", "repetition_penalty");
    const std::string model_dir          = std::string(reader.Get("ft_instance_hyperparameter", "model_dir"));
    const bool        sparse             = static_cast<bool>(reader.GetInteger("ft_instance_hyperparameter", "sparse"));
    const int         int8_mode          = reader.GetInteger("ft_instance_hyperparameter", "int8_mode");
    const float       len_penalty        = reader.GetFloat("ft_instance_hyperparameter", "len_penalty");
    const float       beam_search_diversity_rate =
        reader.GetFloat("ft_instance_hyperparameter", "beam_search_diversity_rate");
    const float shared_contexts_ratio = reader.GetFloat("ft_instance_hyperparameter", "shared_contexts_ratio", true);

    const int tensor_para_size   = tp_deg ? tp_deg : reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
    const int pipeline_para_size = pp_deg ? pp_deg : reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size");

    const size_t      head_num       = (size_t)reader.GetInteger(model_name, "head_num");
    const size_t      size_per_head  = (size_t)reader.GetInteger(model_name, "size_per_head");
    const size_t      vocab_size     = (size_t)reader.GetInteger(model_name, "vocab_size");
    const size_t      decoder_layers = (size_t)reader.GetInteger(model_name, "decoder_layers");
    const size_t      hidden_units   = head_num * size_per_head;
    const size_t      inter_size     = 4 * hidden_units;
    const std::string model_variant  = std::string(reader.Get(model_name, "model_variant", "gpt"));

    const size_t request_batch_size = mini_bsz ? mini_bsz : reader.GetInteger("request", "request_batch_size");
    // The length of tokens we hope this model to generate
    const int  request_output_len  = reader.GetInteger("request", "request_output_len");
    const bool is_return_log_probs = reader.GetBoolean("request", "return_log_probs", false);
    // Whether to include input contexts in computing the cumulative log probabilities.
    const bool is_return_context_cum_log_probs = reader.GetBoolean("request", "context_log_probs", false);
    if (is_return_log_probs && !is_return_context_cum_log_probs) {
        FT_LOG_WARNING("context_log_probs will be ignored since return_log_probs is disabled.");
    }
    const bool     remove_padding = reader.GetBoolean("request", "remove_padding", false);
    const uint32_t memory_len     = reader.GetInteger("request", "memory_len", 0);

    const int start_id = 50256;
    const int end_id   = 50256;

    FT_CHECK(head_num % tensor_para_size == 0);
    FT_CHECK(decoder_layers % pipeline_para_size == 0);

    // Prepare the parallelism parameters
    int rank       = mpi::getCommWorldRank();
    int world_size = mpi::getCommWorldSize();
    if (rank == 0) {
        printf("Total ranks: %d.\n", world_size);
    }
    int device, device_count;
    check_cuda_error(cudaGetDeviceCount(&device_count));
    check_cuda_error(cudaSetDevice(rank % device_count));
    check_cuda_error(cudaGetDevice(&device));

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    printf("Device %s\n", prop.name);

    printf("P%d is running with %d GPU.\n", rank, device);

    if (tensor_para_size * pipeline_para_size != world_size) {
        printf("[ERROR] tensor_para_size * pipeline_para_size should equal to world_size \n");
        exit(-1);
    }

    const int layers_per_group = decoder_layers / pipeline_para_size;
    if (layers_per_group * pipeline_para_size != (int)decoder_layers) {
        printf("[ERROR] layers_per_group (%d) * pipeline_para_size (%d) should equal to decoder_layers (%ld) \n",
               layers_per_group,
               pipeline_para_size,
               decoder_layers);
        exit(-1);
    }

    // assume gpu_num = k * n,
    // tensor parallelism group size is n
    // pipeline parallelism group size is k
    NcclParam tensor_para;
    NcclParam pipeline_para;
    ftNcclInitialize(tensor_para, pipeline_para, tensor_para_size, pipeline_para_size);
    FT_LOG_INFO(rank, "nccl inited");
    FT_PROFILE_SETRANK(rank);
    FT_PROFILE_SETFILE(fmtstr("log/tp%d_c%d_b%d.csv", tensor_para_size, tp_comm, request_batch_size));

    // Read ids of request from file.
    std::vector<int> max_input_lens;
    std::vector<std::vector<int>> v_start_lengths_list;
    std::vector<std::vector<int>> v_start_ids_list;
    read_start_ids_iter(request_batch_size, &v_start_lengths_list, &v_start_ids_list, &max_input_lens, end_id, 1, in_csv);
    printf("---------------max_input_lens.size() = %d----------------\n", max_input_lens.size());

    TcpClient tcp_client("127.0.0.1", BASE_PORT + rank);

    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle;
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle));
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap(GEMM_CONFIG, SPGEMM_CONFIG);
#else
    cublasAlgoMap*  cublas_algo_map = new cublasAlgoMap(GEMM_CONFIG);
#endif

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex* cublas_wrapper_mutex = new std::mutex();
#ifdef SPARSITY_ENABLED
    cublasMMWrapper cublas_wrapper = cublasMMWrapper(
        cublas_handle, cublaslt_handle, cusparselt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
#else
    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
#endif

    if (std::is_same<T, half>::value) {
        cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    // Prompt Learning Configurations
    int                prompt_learning_start_id = reader.GetInteger(model_name, "prompt_learning_start_id", end_id + 1);
    PromptLearningType prompt_learning_type =
        static_cast<PromptLearningType>(reader.GetInteger(model_name, "prompt_learning_type", 0));

    // NOTE：specify task names, take name id, prompt length in order to load those prompt learning tables.
    // for example:
    // std::map<std::string, std::pair<int, int>> p_prompt_tuning_table_pair_{{"sentiment", {0, 10}},
    //                                                                        {"intent_and_slot", {1, 10}},
    //                                                                        {"squad", {2, 16}}};

    std::map<std::string, std::pair<int, int>> p_prompt_tuning_table_pair_;

    // NOTE: get prompt table pairs from configuration files
    const int num_tasks = reader.GetInteger(model_name, "num_tasks", 0);
    for (int task_name_id = 0; task_name_id < num_tasks; task_name_id++) {
        std::string config_task_name = model_name + "_task_" + std::to_string(task_name_id);
        std::string task_name        = reader.Get(config_task_name, "task_name");
        const int   prompt_length    = reader.GetInteger(config_task_name, "prompt_length", 0);
        p_prompt_tuning_table_pair_.insert({task_name, {task_name_id, prompt_length}});
    }

    // NOTE: task_name_ids for each sequence in one batch
    // Each sequence can have different prompt learning task ids
    std::vector<int> p_prompt_tuning_task_name_ids(request_batch_size, 0);

    // NOTE: gpt variants parameters --> meta opt as an example here
    gptVariantParams gpt_variant_params = {};  // default is gpt
    if (model_variant == "opt-pre") {
        gpt_variant_params.layernorm_eps              = 1e-5f;
        gpt_variant_params.layernorm_type             = LayerNormType::pre_layernorm;
        gpt_variant_params.activation_type            = ActivationType::Relu;
        gpt_variant_params.has_post_decoder_layernorm = true;
    }
    else if (model_variant == "opt-post") {
        gpt_variant_params.layernorm_eps              = 1e-5f;
        gpt_variant_params.layernorm_type             = LayerNormType::post_layernorm;
        gpt_variant_params.activation_type            = ActivationType::Relu;
        gpt_variant_params.has_post_decoder_layernorm = false;
    }
    gpt_variant_params.has_adapters       = reader.GetBoolean(model_name, "has_adapters", false);
    gpt_variant_params.adapter_inter_size = reader.GetInteger(model_name, "adapter_inter_size", inter_size);
    gpt_variant_params.layernorm_eps      = reader.GetInteger(model_name, "layernorm_eps", 1e-6f);

    
    FT_LOG_INFO(rank, "malloc before");
    ParallelGptWeight<T> gpt_weights(hidden_units,
                                     inter_size,
                                     vocab_size,
                                     decoder_layers,
                                     max_seq_len,
                                     tensor_para.world_size_,
                                     tensor_para.rank_,
                                     pipeline_para.world_size_,
                                     pipeline_para.rank_,
                                     int8_mode,
                                     prompt_learning_type,
                                     p_prompt_tuning_table_pair_,
                                     gpt_variant_params, 
                                     &tcp_client);
    FT_LOG_INFO(rank, "malloc after");
    gpt_weights.loadModel(model_dir);
#ifdef SPARSITY_ENABLED
    if (sparse) {
        printf("[INFO] Compress weights for sparse inference\n");
        gpt_weights.compress_weights(cublas_wrapper);
    }
#endif

    unsigned long long random_seed;
    if (rank == 0) {
        random_seed = (unsigned long long)(0);
    }
    if (world_size > 1) {
        mpi::bcast(&random_seed, 1, mpi::MPI_TYPE_UNSIGNED_LONG_LONG, 0, mpi::COMM_WORLD);
    }
    FT_LOG_INFO(rank, "load model before");
    ParallelGpt<T> gpt = ParallelGpt<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
                                        0,  // max_seq_len, FT will adjust the buffer automatically.
                                        0,  // max_input_len, FT will adjust the buffer automatically.
                                        beam_width,
                                        head_num,
                                        size_per_head,
                                        inter_size,
                                        decoder_layers,
                                        vocab_size,
                                        start_id,
                                        end_id,
                                        prompt_learning_start_id,  // p/prompt tuning virtual token start id
                                        prompt_learning_type,
                                        gpt_variant_params,
                                        0.0f,  // beam_search_diversity_rate,
                                        0,     // top_k,
                                        0.0,   // top_p,
                                        0,     // random_seed,
                                        1.0f,  // temperature,
                                        0.0f,  // len_penalty,
                                        1.0f,  // repetition_penalty,
                                        tensor_para,
                                        pipeline_para,
                                        stream,
                                        &cublas_wrapper,
                                        &allocator,
                                        false,
                                        &prop,
                                        sparse,
                                        int8_mode,
                                        nullptr,
                                        0,
                                        remove_padding,
                                        shared_contexts_ratio,
                                        &tcp_client,
                                        tp_comm);
    FT_LOG_INFO(rank, "load model after");

    int* d_input_ids = nullptr;
    int* d_input_lengths = nullptr;
    int max_input_len = max_input_lens[0];

    int total_output_len = 0;

    int* d_output_ids = nullptr;
    int* d_sequence_lengths = nullptr;

    float *output_log_probs = nullptr, *d_cum_log_probs = nullptr;


    print_mem_usage();

    int ite = 1;
    cudaDeviceSynchronize();
    mpi::barrier();

    cudaProfilerStart();
    deviceFree(d_input_ids);
    deviceFree(d_input_lengths);
    max_input_len = max_input_lens[0];

    if (max_input_len == 0) {
        // unconditional case, no input ids, so do nothing.
        d_input_ids     = nullptr;
        d_input_lengths = nullptr;
        max_input_len   = 0;
    } else {
        // conditional case.
        printf("%d %d\n", request_batch_size, max_input_len);
        deviceMalloc(&d_input_ids, request_batch_size * max_input_len, false);
        deviceMalloc(&d_input_lengths, request_batch_size, false);
        cudaH2Dcpy(d_input_ids, v_start_ids_list[0].data(), request_batch_size * max_input_len);
        cudaH2Dcpy(d_input_lengths, v_start_lengths_list[0].data(), request_batch_size);
    }

    total_output_len = max_input_len + request_output_len;
    if (total_output_len > (int)max_seq_len) {
        printf("[ERROR] total_output_len (%d) should be <= max_seq_len (%ld). \n", total_output_len, max_seq_len);
        exit(-1);
    }

    deviceFree(d_output_ids);
    deviceFree(d_sequence_lengths);
    deviceMalloc(&d_output_ids, request_batch_size * beam_width * total_output_len, false);
    deviceMalloc(&d_sequence_lengths, request_batch_size * beam_width, false);
    std::vector<uint32_t> output_seq_len_iter(request_batch_size, total_output_len);

    std::unordered_map<std::string, Tensor> input_tensors = std::unordered_map<std::string, Tensor>{
        {"input_ids",
        Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, (size_t)max_input_len}, d_input_ids}},
        {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, d_input_lengths}},
        {"output_seq_len",
        Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len_iter.data()}}};
    if (top_k == 0 && top_p == 0.0f) {
        FT_CHECK(beam_width > 1);
        input_tensors.insert({"beam_search_diversity_rate",
                            Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
    }
    else {
        if (top_p != 0.0f) {
            input_tensors.insert({"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &top_p}});
        }
        if (top_k != 0) {
            input_tensors.insert({"runtime_top_k", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, &top_k}});
        }
    }
    if (num_tasks > 0) {
        input_tensors.insert({"prompt_learning_task_name_ids",
                            Tensor{MEMORY_CPU,
                                    TYPE_INT32,
                                    std::vector<size_t>{request_batch_size},
                                    p_prompt_tuning_task_name_ids.data()}});
    }
    input_tensors.insert({"temperature", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &temperature}});
    input_tensors.insert({"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &len_penalty}});
    input_tensors.insert(
        {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
    input_tensors.insert({"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});
    if (memory_len > 0) {
        input_tensors.insert({"memory_len", {MEMORY_CPU, TYPE_UINT32, {1}, &memory_len}});
    }

    std::unordered_map<std::string, Tensor> output_tensors = std::unordered_map<std::string, Tensor>{
        {"output_ids",
        Tensor{MEMORY_GPU,
                TYPE_INT32,
                std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                d_output_ids}},
        {"sequence_length",
        Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, beam_width}, d_sequence_lengths}}};

    deviceFree(output_log_probs);
    deviceFree(d_cum_log_probs);
    if (is_return_log_probs) {
        deviceMalloc(&output_log_probs, request_batch_size * beam_width * request_output_len);
        output_tensors.insert({"output_log_probs",
                            Tensor{MEMORY_GPU,
                                    TYPE_FP32,
                                    std::vector<size_t>{request_batch_size, beam_width, (size_t)request_output_len},
                                    output_log_probs}});
        deviceMalloc(&d_cum_log_probs, request_batch_size * beam_width);
        output_tensors.insert(
            {"cum_log_probs",
            Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{request_batch_size, beam_width}, d_cum_log_probs}});
        input_tensors.insert({"is_return_context_cum_log_probs",
                            Tensor{MEMORY_CPU, TYPE_BOOL, std::vector<size_t>{1}, &is_return_context_cum_log_probs}});
    }
    
    // sync_and_profile("call_gpt_forward", 0);
    gpt.forward(&output_tensors, &input_tensors, &gpt_weights);

    cudaDeviceSynchronize();
    mpi::barrier();

    FT_LOG_INFO(rank, "test before");
    // test time
    struct timeval start, end;
    mpi::barrier();
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);

    nvtx::setScope("total_time");
    PUSH_RANGE("total time");

    std::vector<double> inference_latencies;
    double sum_latencies = 0;
    // ite = max_input_lens.size();
    ite = 1;

    FT_PROFILE_CLEAN();

    for(int i = 0; i < ite; i++){
        struct timeval iter_start, iter_end;
        gettimeofday(&iter_start, NULL);

        std::unordered_map<std::string, Tensor> input_tensors_iter = std::unordered_map<std::string, Tensor>{
                {"input_ids",
                Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, (size_t)max_input_len}, d_input_ids}},
                {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, d_input_lengths}},
                {"output_seq_len",
                Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len_iter.data()}}};
        if (top_k == 0 && top_p == 0.0f) {
            FT_CHECK(beam_width > 1);
            input_tensors_iter.insert({"beam_search_diversity_rate",
                                Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
        }
        else {
            if (top_p != 0.0f) {
                input_tensors_iter.insert({"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &top_p}});
            }
            if (top_k != 0) {
                input_tensors_iter.insert({"runtime_top_k", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, &top_k}});
            }
        }
        if (num_tasks > 0) {
            input_tensors_iter.insert({"prompt_learning_task_name_ids",
                                Tensor{MEMORY_CPU,
                                        TYPE_INT32,
                                        std::vector<size_t>{request_batch_size},
                                        p_prompt_tuning_task_name_ids.data()}});
        }
        input_tensors_iter.insert({"temperature", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &temperature}});
        input_tensors_iter.insert({"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &len_penalty}});
        input_tensors_iter.insert(
            {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
        input_tensors_iter.insert({"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});
        if (memory_len > 0) {
            input_tensors_iter.insert({"memory_len", {MEMORY_CPU, TYPE_UINT32, {1}, &memory_len}});
        }

        std::unordered_map<std::string, Tensor> output_tensors_iter = std::unordered_map<std::string, Tensor>{
            {"output_ids",
            Tensor{MEMORY_GPU,
                    TYPE_INT32,
                    std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                    d_output_ids}},
            {"sequence_length",
            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, beam_width}, d_sequence_lengths}}};

        deviceFree(output_log_probs);
        deviceFree(d_cum_log_probs);
        if (is_return_log_probs) {
            deviceMalloc(&output_log_probs, request_batch_size * beam_width * request_output_len);
            output_tensors_iter.insert({"output_log_probs",
                                Tensor{MEMORY_GPU,
                                        TYPE_FP32,
                                        std::vector<size_t>{request_batch_size, beam_width, (size_t)request_output_len},
                                        output_log_probs}});
            deviceMalloc(&d_cum_log_probs, request_batch_size * beam_width);
            output_tensors_iter.insert(
                {"cum_log_probs",
                Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{request_batch_size, beam_width}, d_cum_log_probs}});
            input_tensors_iter.insert({"is_return_context_cum_log_probs",
                                Tensor{MEMORY_CPU, TYPE_BOOL, std::vector<size_t>{1}, &is_return_context_cum_log_probs}});
        }
        
        // sync_and_profile("call_gpt_forward", 0);
        gpt.forward(&output_tensors_iter, &input_tensors_iter, &gpt_weights);

        cudaDeviceSynchronize();
        mpi::barrier();

        gettimeofday(&iter_end, NULL);
        double latency = (iter_end.tv_sec - iter_start.tv_sec) * 1000 + (iter_end.tv_usec - iter_start.tv_usec) * 0.001;
        if(rank == 0) {
            inference_latencies.push_back(latency);
            printf("iteration %d: max_input_lens %d latency %.2fms\n", i, max_input_lens[i], latency);
            sum_latencies += latency;
        }
    }
    

    POP_RANGE;
    nvtx::resetScope();
    gettimeofday(&end, NULL);
    FT_LOG_INFO(rank, "test end");

    cudaProfilerStop();
    
    FT_LOG_INFO(rank, "cudaProfilerStop after");

    if (rank == 0) {
        size_t outCount = total_output_len * request_batch_size * beam_width;
        int*   hBuf     = new int[outCount];
        cudaD2Hcpy(hBuf, d_output_ids, outCount);

        {
            std::cout << "Writing " << outCount << " elements\n";
            int zeroCount = 0;
            for (size_t i = 0; i < outCount; i++) {
                if (hBuf[i] == int(0)) {
                    zeroCount++;
                }

                printf("%5d ", hBuf[i]);

                if ((i + 1) % (total_output_len) == 0) {
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl << "zeroCount = " << zeroCount << std::endl;
        }
        delete[] hBuf;
    }

    if(rank == 0){
        printf("[INFO] request_batch_size %ld beam_width %ld head_num %ld size_per_head %ld total_output_len %d"
           " decoder_layers %ld vocab_size %ld FT-CPP-decoding-beamsearch-time %.2f ms\n",
           request_batch_size,
           beam_width,
           head_num,
           size_per_head,
           total_output_len,
           decoder_layers,
           vocab_size,
           ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / (ite*request_batch_size));
        
        std::sort(inference_latencies.begin(), inference_latencies.end());
        int p99_index = (int)(request_batch_size * ite * 0.90) / request_batch_size - 1;
        if(p99_index < 0) p99_index = 0;

        printf("****************[INFO] max %.2f ms, avg %.2f ms, p90 %.2f ms****************\n", inference_latencies[ite-1], sum_latencies / ite, inference_latencies[p99_index]);

        FT_PROFILE_PRINT();
    }

    ftNcclParamDestroy(tensor_para);
    ftNcclParamDestroy(pipeline_para);

#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparselt_handle);
#endif
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return;
}
