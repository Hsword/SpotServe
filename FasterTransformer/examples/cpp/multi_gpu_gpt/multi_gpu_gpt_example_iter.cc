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

#include "3rdparty/json.hpp"
#include "3rdparty/INIReader.h"
#include "examples/cpp/multi_gpu_gpt/gpt_example_utils.h"
#include "examples/cpp/multi_gpu_gpt/gpt_estimation_utils.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"
#include "src/fastertransformer/utils/tcp_utils.h"
#include "src/fastertransformer/utils/request_pool.h"

#include <cuda_profiler_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <ctime>
#include <csignal>
#include <thread>
#include <vector>

#ifdef USE_NVTX
bool NVTX_ON = true;
#endif

#define BASE_PORT 10051

using namespace fastertransformer;
using json = nlohmann::json;

template<typename T>
void multi_gpu_gpt_example(const INIReader reader, std::string in_csv, int mini_bsz, int replica_id, int tp_deg, int pp_deg,
     const char* api_server_ip, int api_server_port, std::string profile_json);

bool LOG_DEBUG = false;

void debug_print(std::string msg) {
    if (!LOG_DEBUG) return;
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::cout << "[" << std::put_time(std::localtime(&in_time_t), "%F %T") << "." << std::setw(3)
              << std::setfill('0') << ms.count() << "] " << msg << std::endl;
}

// a global request pool
RequestPool request_pool;

void signalHandler(int signum) {
    int rank = mpi::getCommWorldRank();
    FT_LOG_INFO(rank, "Interrupt signal (%d) received.", signum);
    request_pool.setNoMoreRequests(-1, -1);
}

void api_inference_request_thread(const char* server_ip, int server_port, int replica_id, int rank,
                                  std::string query_file) {
    // resquest thread use `server_port`
    TcpClient tcp_client{server_ip, server_port};
    {
        std::vector<int> msg{replica_id, rank};
        tcp_client.tcpSend((void*)msg.data(), msg.size() * sizeof(int));
        request_pool.setReqConnFinish();
        FT_LOG_INFO(rank, "Send replica id %d, rank %d to request server.", replica_id, rank);
    }

    char* is_triton = std::getenv("BASELINE_TRITON");
    char* need_notify = std::getenv("NAIVE_NOTIFY");
    if (is_triton || need_notify) {
        while (1) {
            if (request_pool.needNotifyApiServer()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(17));
        }
    }

    std::ifstream query_file_stream(query_file);
    while (1) {
        std::vector<int> query{-1, 0};
        tcp_client.tcpRecv((void*)query.data(), 2 * sizeof(int));

        debug_print("Replica rank " + std::to_string(replica_id) + " find query " +
                    std::to_string(query[0]) + ", " + std::to_string(query[1]) + " begin to put request.");

        if (query[0] == -1 && query[1] == -2) {
            // recv interrupted requests
            tcp_client.tcpRecv((void*)query.data(), 2 * sizeof(int));
            int replica_size = query[0];
            int start_step = query[1];
            request_pool.setInitBatchSize(replica_size);
            for (int i = 0; i < replica_size; i++) {
                tcp_client.tcpRecv((void*)query.data(), 2 * sizeof(int));
                FT_LOG_INFO(rank, "Recv **int** request %d.", query[0]);
                request_pool.putRequest(query_file_stream, query[0], query[1], start_step);
            }
        } else if (query[0] == -1 && query[1] == -1) {
            // close the request thread
            // tcp_client.tcpRecv((void*)query.data(), 2 * sizeof(int));
            // request_pool.setNoMoreRequests(query[0], query[1]);
            break;
        } else if (query[0] == -2) {
            float xfer_cost;
            tcp_client.tcpRecv((void*)&xfer_cost, sizeof(float));
            request_pool.setEstimateXferCost(xfer_cost);
            FT_LOG_INFO(rank, "Recv estimate cost query %f.", xfer_cost);
        } else if (query[0] == -3) {
            // recv a batch of requests
            int batch_size = query[1];
            std::vector<int> ids, offsets;
            for (int i = 0; i < batch_size; i++) {
                tcp_client.tcpRecv((void*)query.data(), 2 * sizeof(int));
                FT_LOG_INFO(rank, "Recv batch request %d. (%d/%d)", query[0], i, batch_size);
                ids.push_back(query[0]);
                offsets.push_back(query[1]);
            }
            request_pool.putBatchRequest(query_file_stream, ids, offsets);
        } else if (query[0] >= 0) {
            FT_LOG_INFO(rank, "Recv request %d.", query[0]);
            request_pool.putRequest(query_file_stream, query[0], query[1]);
        }
    }
}

void api_inference_response_thread(const char* server_ip, int server_port, int replica_id, int rank) {
    // response thread use `server_port + 1`
    TcpClient tcp_client{server_ip, server_port + 1};
    {
        std::vector<int> msg{replica_id, rank};
        tcp_client.tcpSend((void*)msg.data(), msg.size() * sizeof(int));
        request_pool.setRespConnFinish();
        FT_LOG_INFO(rank, "Send replica id %d, rank %d to response server.", replica_id, rank);
    }
    char* is_triton = std::getenv("BASELINE_TRITON");
    char* need_notify = std::getenv("NAIVE_NOTIFY");
    if (is_triton || need_notify) {
        while (1) {
            if (request_pool.needNotifyApiServer()) {
                std::vector<float> resp_msg{-2, -2, -2};
                tcp_client.tcpSend((void*)resp_msg.data(), resp_msg.size() * sizeof(float));
                FT_LOG_INFO(rank, "Send notify signal to api server.");
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(13));
        }
    }

    while (1) {
        // send response
        auto resp = request_pool.getResponse();
        while (resp != nullptr) {
            debug_print("Replica rank " + std::to_string(replica_id) + " find response " +
                        std::to_string(resp->id_) + " begin to send response.");
            std::vector<float> resp_msg{float(resp->id_), resp->getScheduleLatency(), resp->getInferenceLatency()};
            tcp_client.tcpSend((void*)resp_msg.data(), resp_msg.size() * sizeof(float));
            resp = request_pool.getResponse();
        }

        if (request_pool.isFinished()) {
            // finished signal: -1, replica_size, infer_lat
            std::vector<float> resp_msg{-1, -1, -1};
            if (request_pool.interrupt_resp_queue_.size() > 0) {
                resp_msg[1] = float(request_pool.interrupt_resp_queue_.size());
                resp_msg[2] = request_pool.interrupt_resp_queue_.front()->getInferenceLatency();
            }
            // std::cout << "Find response finished, begin to send signal to api_server" << std::endl;
            tcp_client.tcpSend((void*)resp_msg.data(), resp_msg.size() * sizeof(float));
            while (request_pool.interrupt_resp_queue_.size() > 0) {
                resp = request_pool.interrupt_resp_queue_.front();
                request_pool.interrupt_resp_queue_.pop();
                std::vector<float> resp_msg{float(resp->id_), float(resp->end_step_), resp->getScheduleLatency()};
                tcp_client.tcpSend((void*)resp_msg.data(), resp_msg.size() * sizeof(float));
            }
            FT_LOG_INFO(rank, "Replica rank %d find response finished, exit...", replica_id);
            break;
        }
        // sleep a while to wait for new response
        std::this_thread::sleep_for(std::chrono::milliseconds(27));
    }
}


static std::shared_ptr<std::thread> api_service_req_thread;
static std::shared_ptr<std::thread> api_service_resp_thread;
std::chrono::system_clock::time_point main_start_time;

int main(int argc, char* argv[])
{
    main_start_time = std::chrono::system_clock::now();
    // signal(SIGUSR1, signalHandler);
    mpi::initialize(&argc, &argv);
    srand(0);

    // handle command line arguments
    int mini_bsz              = argc >= 2 ? atoi(argv[1]) : 1;
    int replica_id            = argc >= 3 ? atoi(argv[2]) : 0;
    int tp_deg                = argc >= 4 ? atoi(argv[3]) : 1;
    int pp_deg                = argc >= 5 ? atoi(argv[4]) : 1;
    const char* api_server_ip = argc >= 6 ? argv[5] : "127.0.0.1";
    int api_server_port       = argc >= 7 ? atoi(argv[6]) : 14762;
    std::string ini_name      = argc >= 8 ? argv[7] : "../examples/cpp/multi_gpu_gpt/gpt_config.ini";
    std::string in_csv        = argc >= 9 ? argv[8] : "../examples/cpp/multi_gpu_gpt/start_ids_176.csv";
    std::string profile_json  = argc >= 10? argv[9] : "../examples/cpp/multi_gpu_gpt/profile/megatron_6.7B_profile.json";

    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        return -1;
    }
    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");

    FT_LOG_INFO("Replica " + std::to_string(replica_id) + " mpi inited");

    int rank       = mpi::getCommWorldRank();
    int world_size = mpi::getCommWorldSize();

    // only the device that receives request or sends response need to start api_service_thread
    // moved to global vars
    // std::shared_ptr<std::thread> api_service_req_thread;
    // std::shared_ptr<std::thread> api_service_resp_thread;

    if (data_type == "fp32") {
        multi_gpu_gpt_example<float>(reader, in_csv, mini_bsz, replica_id, tp_deg, pp_deg, api_server_ip, api_server_port, profile_json);
    }
    else if (data_type == "fp16") {
        multi_gpu_gpt_example<half>(reader, in_csv, mini_bsz, replica_id, tp_deg, pp_deg, api_server_ip, api_server_port, profile_json);
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        multi_gpu_gpt_example<__nv_bfloat16>(reader, in_csv, mini_bsz, replica_id, tp_deg, pp_deg);
    }
#endif
    else {
        printf("[ERROR] data_type should be fp32, fp16 or bf16 ! \n");
        return -1;
    }

    // wait for api_service_thread to finish
    if (rank == 0) {
        api_service_resp_thread->join();
        api_service_req_thread->join();
    }
    mpi::barrier();
    FT_LOG_INFO("Replica rank " + std::to_string(replica_id) + " exited main.");

    mpi::finalize();
    return 0;
}

int check_request_pool_status(int rank, int world_size, int M1, size_t request_batch_size, int& work_req_bs, const CostEstimator& est) {
    std::vector<int> status_vec({0, 0});
    if (rank == 0) {
        status_vec[0] = (int)request_pool.isRequestFinished();
        status_vec[1] = (int)request_pool.numPendingRequests();

        // work_req_bs \in [0, 1, 2, 4, 8, ...]
        work_req_bs = 1;
        while (work_req_bs * 2 <= request_batch_size && work_req_bs * 2 <= status_vec[1]) {
            work_req_bs *= 2;
        }
        work_req_bs = std::min(work_req_bs, status_vec[1]);

        int steps = -1;
        if (request_pool.estimate_xfer_cost_ != 0) {
            const auto now = std::chrono::system_clock::now();
            // in sec
            double t = (double)std::chrono::duration_cast<std::chrono::microseconds>(now - request_pool.signal_time_).count() / 1e6;
            //  (signal_time_ + 30s - xfer_cost) - now
            double slot = 30 - request_pool.estimate_xfer_cost_ - t;
            double min_t1 = est.get_t1(1, true);
            steps = slot <= min_t1 ? 0 : est.calc_remain_step(slot, work_req_bs, request_pool.estimate_xfer_cost_>=0);
        }

        if(steps == -1){
            // can run full request, do nothing
        }else if(steps > 0){
            request_pool.setNoMoreRequests(work_req_bs, steps);
        }else if(steps == 0){
            // cannot run any step, set exit
            status_vec[0] = 1;
            request_pool.setNoMoreRequests(0, -1);
        }

        mpi::bcast(status_vec.data(), 2, mpi::MPI_TYPE_INT, 0, mpi::COMM_WORLD);
    } else {
        mpi::bcast(status_vec.data(), 2, mpi::MPI_TYPE_INT, 0, mpi::COMM_WORLD);
    }

    if (status_vec[0] == 1) {
        return -1;
    } else {
        return status_vec[1];
    }
}

int prepare_input_tokens(std::vector<int>& start_ids, std::vector<int>& start_lengths,
                          std::vector<std::shared_ptr<Request>>& requests, const int padding_id) {
    int max_length = 0;
    for (auto& req : requests) {
        max_length = std::max(max_length, req->getSeqLen());
    }

    // TODO: design a padding mechanism to fully use pipeline parallelism!
    for (auto& req: requests) {
        int i = 0;
        for (; i < req->input_ids_.size(); i++) {
            start_ids.push_back(req->input_ids_[i]);
        }
        for (; i < max_length; i++) {
            start_ids.push_back(padding_id);
        }
        start_lengths.push_back(req->getSeqLen());
    }
    return max_length;
}

void debug_show_output(int batch_size, int beam_width, int total_output_len, int* d_output_ids,
                       float* d_cum_log_probs) {
    std::string fName = "log/infer.out";
    auto outFile = std::ofstream(fName, std::ios::out);
    if (!outFile.is_open()) {
        printf("[WARNING] Cannot write results into output file %s \n", fName.c_str());
    } else {
        size_t outCount = total_output_len * batch_size * beam_width;
        int* hBuf = new int[outCount];
        cudaD2Hcpy(hBuf, d_output_ids, outCount);

        {
            std::cout << "Writing " << outCount << " elements\n";
            int zeroCount = 0;
            for (size_t i = 0; i < outCount; i++) {
                if (hBuf[i] == int(0)) {
                    zeroCount++;
                }
                outFile << hBuf[i] << " ";
                if ((i + 1) % (total_output_len) == 0) {
                    outFile << std::endl;
                }

                if (i < 10) {
                    printf("%5d ", hBuf[i]);
                }
                if ((i + 1) % (total_output_len) == 0 && i < 10) {
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl << "zeroCount = " << zeroCount << std::endl;
        }
        delete[] hBuf;
    }
    outFile.close();

    if (d_cum_log_probs != nullptr) {
        std::string logprob_fname = "log/logprob.out";
        std::ofstream logprob_file = std::ofstream("logprob.out", std::ios::out);
        if (!logprob_file.is_open()) {
            printf("[WARNING] Cannot write results into output file %s \n", logprob_fname.c_str());
        } else {
            size_t cum_log_probs_size = batch_size * beam_width;
            printf("[INFO] Writing %ld elements (log probs)\n", cum_log_probs_size);
            float* h_buf = new float[cum_log_probs_size];
            cudaD2Hcpy(h_buf, d_cum_log_probs, cum_log_probs_size);
            for (size_t i = 0; i < cum_log_probs_size; i++) {
                logprob_file << h_buf[i] << std::endl;
                if (i < 10) {
                    printf(" %10.6f\n", h_buf[i]);
                }
            }
            delete[] h_buf;
        }
        logprob_file.close();
    }
}

std::vector<int> get_coord_info(int local_rank) {
    const char* coord_map_str = std::getenv("COORD_MAP");
    json coord_map = json::parse(coord_map_str);

    // retrieve local node ip address
    std::string local_ip = getHostIP();
    std::string key = local_ip + "-" + std::to_string(local_rank);
    return coord_map[key];
}

template<typename T>
void multi_gpu_gpt_example(const INIReader reader, std::string in_csv, int mini_bsz, int replica_id, int tp_deg, int pp_deg,
    const char* api_server_ip, int api_server_port, std::string profile_json)
{
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
    const size_t      real_layers    = (size_t)reader.GetInteger(model_name, "real_layers", -1);
    FT_LOG_INFO("main.cc: real layer %ld", real_layers);
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

    // Prompt Learning Configurations (set disabled)
    int                prompt_learning_start_id = reader.GetInteger(model_name, "prompt_learning_start_id", end_id + 1);
    PromptLearningType prompt_learning_type =
        static_cast<PromptLearningType>(reader.GetInteger(model_name, "prompt_learning_type", 0));
    std::map<std::string, std::pair<int, int>> p_prompt_tuning_table_pair_({});

    int BSZ, M1, M2;

    char* env_value = std::getenv("FT_MICRO_M");
    if (env_value) {
        std::string env_input(env_value);
        std::stringstream ss(env_input);
        char delimiter;
        ss >> BSZ >> delimiter >> M1 >> delimiter >> M2;
        FT_LOG_INFO("FT_MICRO_M is set to, BSZ: %d, M1: %d, M2: %d", BSZ, M1, M2);
    } else {
        M1 = 0;
        M2 = 0;
    }

    FT_CHECK(head_num % tensor_para_size == 0);
    FT_CHECK(decoder_layers % pipeline_para_size == 0);

    // Prepare the parallelism parameters
    int rank       = mpi::getCommWorldRank();
    int world_size = mpi::getCommWorldSize();
    if (rank == 0) {
        printf("Total ranks: %d.\n", world_size);
    }

    CostEstimator costEstimator(profile_json, tensor_para_size, pipeline_para_size, M1, M2, request_output_len);
    if (rank == 0) {
        api_service_req_thread = std::make_shared<std::thread>(api_inference_request_thread, api_server_ip,
                                                               api_server_port, replica_id, rank, in_csv);
        api_service_resp_thread = std::make_shared<std::thread>(api_inference_response_thread, api_server_ip,
                                                                api_server_port, replica_id, rank);
    }

    int device, device_count;
    check_cuda_error(cudaGetDeviceCount(&device_count));
    check_cuda_error(cudaSetDevice(rank % device_count));
    check_cuda_error(cudaGetDevice(&device));

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    printf("P%d is running with %d GPU (%s).\n", rank, device, prop.name);

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

    // get coordinate info
    auto coord_info = get_coord_info(rank % device_count);
    int global_pc_rank = coord_info[0];
    int tp_rank = coord_info[1];
    int pp_rank = coord_info[2];

    // assume gpu_num = k * n,
    // tensor parallelism group size is n
    // pipeline parallelism group size is k
    NcclParam tensor_para;
    NcclParam pipeline_para;
    ftNcclInitializeWithCoordinate(tensor_para, pipeline_para, tensor_para_size, pipeline_para_size, tp_rank, pp_rank);
    FT_LOG_INFO(rank, "nccl inited");

    // prepare cuda env
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

    // Prepare model weights
    int pc_port = BASE_PORT + global_pc_rank;
    TcpClient tcp_client("127.0.0.1", pc_port);

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
                                        true,
                                        real_layers);
    FT_LOG_INFO(rank, "load model after");
    std::chrono::system_clock::time_point main_end_time = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(main_end_time - main_start_time);
    FT_LOG_INFO(rank, "Main start total time: %d ms", diff.count());

    // block util finish model load
    char* is_triton = std::getenv("BASELINE_TRITON");
    char* need_block = std::getenv("NAIVE_BLOCK");
    char* need_notify = std::getenv("NAIVE_NOTIFY");
    if (is_triton || need_block) {
        while (1) {
            int is_finish = request_pool.isConnFinish();
            mpi::bcast(&is_finish, 1, mpi::MPI_TYPE_INT, 0, mpi::COMM_WORLD);
            if (is_finish) {
                std::this_thread::sleep_for(std::chrono::milliseconds(47));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }
        gpt.blockUntilAllParamReady();
        mpi::barrier();
        std::this_thread::sleep_for(std::chrono::milliseconds(diff.count()));
        if(need_block) std::this_thread::sleep_for(std::chrono::milliseconds(diff.count()));
        request_pool.setNotifyApiServer();
        FT_LOG_INFO(rank, "Finish block until all param ready, %d", request_pool.needNotifyApiServer());
    } else if (need_notify) {
        while (1) {
            int is_finish = request_pool.isConnFinish();
            mpi::bcast(&is_finish, 1, mpi::MPI_TYPE_INT, 0, mpi::COMM_WORLD);
            if (is_finish) {
                std::this_thread::sleep_for(std::chrono::milliseconds(47));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }
        request_pool.setNotifyApiServer();
    }

    bool debug_out = false;


    // Inference Service
    while (1) {
        int work_req_bs;
        int status = check_request_pool_status(rank, world_size, M1, request_batch_size, work_req_bs, costEstimator);
        if (status < 0) break;
        if (status == 0) { // No request in the pool.
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        // Get request from request pool and prepare input data.
        // int work_req_bs = std::min((int)request_batch_size, (status / M1_BS) * M1_BS);
        std::vector<std::shared_ptr<Request>> working_requests;
        std::vector<int> start_ids, start_lengths;
        // token infos: cur max_seq_len, cur batch_size, cur start_step, cur end_step
        std::vector<int> token_info({0, (int)work_req_bs, 0, -1});

        if (rank == 0) {
            int bs = request_pool.getBatchRequests(work_req_bs, token_info[2], token_info[3], working_requests);
            std::string query_ids = "";
            for (auto request : working_requests) {
                request->setStart();
                query_ids += std::to_string(request->id_) + ", ";
            }

            token_info[0] = prepare_input_tokens(start_ids, start_lengths, working_requests, end_id);
            token_info[1] = working_requests.size();
            FT_LOG_INFO(rank, ">> Replica rank %d Processing request ids %s with max_len %d start_step: %d end_step: %d",
                        replica_id, query_ids.c_str(), token_info[0], token_info[2], token_info[3]);

            mpi::bcast(token_info.data(), 4, mpi::MPI_TYPE_INT, 0, mpi::COMM_WORLD);

            mpi::bcast(start_ids.data(), start_ids.size(), mpi::MPI_TYPE_INT, 0, mpi::COMM_WORLD);
            mpi::bcast(start_lengths.data(), start_lengths.size(), mpi::MPI_TYPE_INT, 0, mpi::COMM_WORLD);
        } else {
            mpi::bcast(token_info.data(), 4, mpi::MPI_TYPE_INT, 0, mpi::COMM_WORLD);

            for (int i = 0; i < token_info[0] * token_info[1]; i++) {
                start_ids.push_back(0);
            }
            for (int i = 0; i < token_info[1]; i++) {
                start_lengths.push_back(0);
            }
            mpi::bcast(start_ids.data(), start_ids.size(), mpi::MPI_TYPE_INT, 0, mpi::COMM_WORLD);
            mpi::bcast(start_lengths.data(), start_lengths.size(), mpi::MPI_TYPE_INT, 0, mpi::COMM_WORLD);
        }

        int max_input_len = token_info[0];
        size_t batch_size = (size_t)token_info[1];
        int start_step = token_info[2];
        int iter_step = token_info[3];
        int total_output_len = max_input_len + request_output_len;
        if (total_output_len > (int)max_seq_len) {
            printf("[ERROR] total_output_len (%d) should be <= max_seq_len (%ld). \n", total_output_len,
                   max_seq_len);
            exit(-1);
        }

        int* d_input_ids;
        int* d_input_lengths;
        deviceMalloc(&d_input_ids, batch_size * max_input_len, false);
        deviceMalloc(&d_input_lengths, batch_size, false);
        cudaH2Dcpy(d_input_ids, start_ids.data(), batch_size * max_input_len);
        cudaH2Dcpy(d_input_lengths, start_lengths.data(), batch_size);

        int* d_output_ids;
        int* d_sequence_lengths;
        deviceMalloc(&d_output_ids, batch_size * beam_width * total_output_len, false);
        deviceMalloc(&d_sequence_lengths, batch_size * beam_width, false);
        std::vector<uint32_t> output_seq_len(batch_size, total_output_len);
        // std::cout << "  ---- Output len: " << total_output_len << " batch: " << batch_size << " beam width: " << beam_width << std::endl;

        std::unordered_map<std::string, Tensor> input_tensors = std::unordered_map<std::string, Tensor>{
            {"input_ids", Tensor{MEMORY_GPU, TYPE_INT32,
                                 std::vector<size_t>{batch_size, (size_t)max_input_len}, d_input_ids}},
            {"input_lengths",
             Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, d_input_lengths}},
            {"output_seq_len",
             Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{batch_size}, output_seq_len.data()}}};
        if (top_k == 0 && top_p == 0.0f) {
            FT_CHECK(beam_width > 1);
            input_tensors.insert(
                {"beam_search_diversity_rate",
                 Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
        } else {
            if (top_p != 0.0f) {
                input_tensors.insert(
                    {"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &top_p}});
            }
            if (top_k != 0) {
                input_tensors.insert(
                    {"runtime_top_k", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, &top_k}});
            }
        }
        input_tensors.insert(
            {"temperature", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &temperature}});
        input_tensors.insert(
            {"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &len_penalty}});
        input_tensors.insert({"repetition_penalty",
                              Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
        input_tensors.insert(
            {"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});
        if (memory_len > 0) {
            input_tensors.insert({"memory_len", {MEMORY_CPU, TYPE_UINT32, {1}, &memory_len}});
        }
        if (start_step > 0) {
            input_tensors.insert({"start_step", {MEMORY_CPU, TYPE_INT32, {1}, &start_step}});
        }
        if (iter_step > 0) {
            input_tensors.insert({"iter_step", {MEMORY_CPU, TYPE_INT32, {1}, &iter_step}});
        }

        // Prepare output tensors.
        std::unordered_map<std::string, Tensor> output_tensors = std::unordered_map<std::string, Tensor>{
            {"output_ids",
             Tensor{MEMORY_GPU, TYPE_INT32,
                    std::vector<size_t>{batch_size, beam_width, (size_t)total_output_len}, d_output_ids}},
            {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, beam_width},
                                       d_sequence_lengths}}};

        float* output_log_probs = nullptr;
        float* d_cum_log_probs  = nullptr;
        if (is_return_log_probs) {
            deviceMalloc(&output_log_probs, batch_size * beam_width * request_output_len);
            output_tensors.insert(
                {"output_log_probs",
                 Tensor{MEMORY_GPU, TYPE_FP32,
                        std::vector<size_t>{batch_size, beam_width, (size_t)request_output_len},
                        output_log_probs}});
            deviceMalloc(&d_cum_log_probs, batch_size * beam_width);
            output_tensors.insert(
                {"cum_log_probs", Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{batch_size, beam_width},
                                         d_cum_log_probs}});
            input_tensors.insert(
                {"is_return_context_cum_log_probs",
                 Tensor{MEMORY_CPU, TYPE_BOOL, std::vector<size_t>{1}, &is_return_context_cum_log_probs}});
        }

        gpt.forward(&output_tensors, &input_tensors, &gpt_weights);
        // FT_LOG_INFO(mpi::getCommWorldRank(), "FT returned from gpt.forward");

        // debug
        if (debug_out) {
            debug_show_output(batch_size, beam_width, total_output_len, d_output_ids, d_cum_log_probs);
            debug_out = false;
        }

        // Postprocess: clean memorys and buffers.
        for (auto& req : working_requests) {
            req->setEnd();
            if (iter_step > 0) {
                req->end_step_ = gpt.get_step();
                debug_print("Request " + std::to_string(req->id_) + " interrupt at step: " + std::to_string(req->end_step_));
            }
            request_pool.putResponse(req);
            debug_print("Finish request " + std::to_string(req->id_) + " with latency " +
                        std::to_string(req->getInferenceLatency()) + " + " + std::to_string(req->getScheduleLatency()) + " ms.");
            FT_LOG_INFO(rank, "Finish request %d with latency %f ms", req->id_, req->getInferenceLatency());
        }

        deviceFree(d_input_ids);
        deviceFree(d_input_lengths);
        deviceFree(d_output_ids);
        deviceFree(d_sequence_lengths);
        if (is_return_log_probs) {
            deviceFree(output_log_probs);
            deviceFree(d_cum_log_probs);
        }
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
