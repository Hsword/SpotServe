#include <chrono>
#include <stdio.h>
#include <fcntl.h>
#include <thread>
#include <stdexcept>
#include <signal.h>

#include "3rdparty/INIReader.h"
#include "src/utils/mpi_utils.h"
#include "src/utils/nccl_utils.h"
#include "src/utils/tcp_utils.h"

#include "src/client/TensorStorage.h"
#include "src/client/layerConfig.h"

using namespace fastertransformer;

#define GLOBAL_PORT 10040
#define BASE_PORT 10051

static int print_tensor_verbose = 0;

TensorStorage* tensor_storage;

void print_all_tensor(int rank){
    if (print_tensor_verbose){
        int fd = open(fmtstr("log/rank%d.log", rank).c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        int std_out = dup(1);
        dup2(fd, 1);
        printf("====[rank %d]====\n", rank);
        tensor_storage->print_test_params(print_tensor_verbose);
        dup2(std_out, 1);
        close(fd);
    }
}

void interact_thread(int rank, int device_id){
    check_cuda_error(cudaSetDevice(device_id));
    FT_LOG_INFO(rank, "interaction thread started, on GPU %d.", getDevice());

    TcpServer tcp_server(BASE_PORT + rank);

    while(1){
        int conn_fd = tcp_server.tcpAccept();
        TcpAgent tcp_agent = TcpAgent(conn_fd, tcp_server.address);
        try{
            while(1){
                FT_LOG_DEBUG(rank, "IT, waiting request.");
                ClientRequest_t client_req;
                tcp_agent.tcpRecv(&client_req, sizeof(ClientRequest_t));
                // TODO: ######
                // param_type:   29 bits zero + 2 bits(00 get_param; 01 get_buff; 11 alloc_buff) + 1 bit(is_tp)

                FT_LOG_DEBUG(rank, "Recv request {id: %d, t: %d, idx: %d, sz: %d}",
                    client_req.layer_id, client_req.param_type, client_req.param_idx, client_req.req_size);
                if(client_req.param_type > 9){
                    FT_LOG_DEBUG(rank, "IT, invalid request.");
                    break;
                }else if(client_req.param_type == 9){
                    tensor_storage->send_metadata(tcp_agent, client_req.layer_id, client_req.param_idx, client_req.param_type);
                }else if(client_req.param_type < 2){
                    tensor_storage->send_metadata(tcp_agent, client_req.layer_id, client_req.param_idx, client_req.param_type); // sync function
                }else if ((client_req.param_type & 6) == 6){
                    // allocate buffer
                    tensor_storage->alloc_new_buffer_and_send(tcp_agent, client_req.layer_id, client_req.param_idx, client_req.param_type & 1, client_req.req_size);
                }else if((client_req.param_type & 6) == 2){
                    // get buffer
                    tensor_storage->send_buffdata(tcp_agent, client_req.layer_id, client_req.param_idx, client_req.param_type & 1);
                }
            }
        }catch(TcpEOFException){
            FT_LOG_DEBUG(rank, "IT, connection reset.");
        }
        close(conn_fd);
    }
}


int main(int argc, char* argv[]){
    signal(SIGPIPE, SIG_IGN);

    FT_LOG_DEBUG("entering main.");
    mpi::initialize(&argc, &argv);

    // handle args
    std::string ini_cfg   = argc >= 2 ? argv[1] : "/home/duanjiangfei/spot_inference/ft-auto-switch/examples/cpp/multi_gpu_gpt/gpt_config.ini";
    std::string ckpt_path = argc >= 3 ? argv[2] : "/home/duanjiangfei/spot_inference/ft-auto-switch/models/megatron-models/c-model/345m/1-gpu/";
    const char *master_ip = argc >= 4 ? argv[3] : "127.0.0.1";
    int master_port       = argc >= 5 ? atoi(argv[4]) : GLOBAL_PORT;
    int init_dp_para_size = argc >= 6 ? atoi(argv[5]) : 1;
    int init_tp_para_size = argc >= 7 ? atoi(argv[6]) : 1;
    int init_pp_para_size = argc >= 8 ? atoi(argv[7]) : 1;
    // (dp, tp, pp) = rank//(tp*pp), rank%tp, (rank//tp)%pp

    int rank       = mpi::getCommWorldRank();
    int world_size = mpi::getCommWorldSize();
    if (rank == 0) {
        FT_LOG_DEBUG(rank, "Total ranks: %d.\n", world_size);
    }
    int device, visible_device_count;
    check_cuda_error(cudaGetDeviceCount(&visible_device_count));
    check_cuda_error(cudaSetDevice(rank % visible_device_count));
    check_cuda_error(cudaGetDevice(&device));

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    printf("Device %s\n", prop.name);

    // init nccl
    NcclParam nccl_param;
    ftNcclInitialize(nccl_param, world_size, rank);

    cudaStream_t     stream;
    cudaStreamCreate(&stream);

    printf("P%d(rank %d)is running with %d GPU.\n", rank, nccl_param.rank_, device);

    // cubulas
    if (nccl_param.rank_ == 0) {
        FT_LOG_DEBUG(nccl_param.rank_, "nccl intialized, starting tcp");
    }

    // read model config
    INIReader reader = INIReader(ini_cfg);
    const std::string model_name     = reader.Get("ft_instance_hyperparameter", "model_name");
    const size_t      max_seq_len    = (size_t)reader.GetInteger("ft_instance_hyperparameter", "max_seq_len");
    const size_t      head_num       = (size_t)reader.GetInteger(model_name, "head_num");
    const size_t      size_per_head  = (size_t)reader.GetInteger(model_name, "size_per_head");
    const size_t      vocab_size     = (size_t)reader.GetInteger(model_name, "vocab_size");
    const size_t      decoder_layers = (size_t)reader.GetInteger(model_name, "decoder_layers");
    const size_t      hidden_units   = head_num * size_per_head;

    std::cout << "Model layer: " << decoder_layers << std::endl;
    std::cout << "Model hidden units: " << hidden_units << std::endl;
    std::cout << "Model max seq len: " << max_seq_len << std::endl;
    std::cout << "Model vocab size: " << vocab_size << std::endl;
    std::cout << "Load checkpoint from: " << ckpt_path << std::endl;
    ParamConfig_t* config = get_gpt_configs(decoder_layers, hidden_units, max_seq_len, vocab_size);

    tensor_storage = new TensorStorage{nccl_param.rank_, stream, *config, nccl_param};
    std::thread interaction_thread(interact_thread, nccl_param.rank_, rank % visible_device_count);
    interaction_thread.detach();

    FT_LOG_DEBUG(rank, "connected, assigning initial params");

    auto start = std::chrono::high_resolution_clock::now();
    std::string model_weight_path(ckpt_path);
    tensor_storage->load_initial_weights(model_weight_path, init_pp_para_size, init_dp_para_size, init_tp_para_size);

    check_cuda_error(cudaDeviceSynchronize());
    // sync_check_cuda_error();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    FT_LOG_DEBUG(rank, "initial params loaded, takes time: %fs", elapsed_seconds.count());

    TcpClient tcp_client{master_ip, master_port};
    tcp_client.tcpSend((void*)(&rank), sizeof(int));

    // tensor_storage.assign_test_params_value();
    FT_LOG_DEBUG(rank, "finished, listening");
    print_all_tensor(rank);


    while(1){
        cJSON* switch_json = tcp_client.tcpRecvJson();

        FT_LOG_DEBUG(rank, "received json.");

        // check if the job is completed
        cJSON* job_done = cJSON_GetObjectItem(switch_json, "job_done");
        if (job_done != NULL){
            FT_LOG_INFO(rank, "Job finished.");
            break;
        }

        cJSON* clear_json = cJSON_GetObjectItem(switch_json, "clean_rank");
        cJSON* load_disk = cJSON_GetObjectItem(switch_json, "load_disk");
        if(clear_json == NULL && load_disk == NULL){
            tensor_storage->do_op(switch_json);
        }else{
            if(clear_json != NULL){
                FT_LOG_INFO(rank, "Clear all params");
                tensor_storage->clear_weights();
            }
            if(load_disk != NULL){
                FT_LOG_INFO(rank, "Load from disk.");
                tensor_storage->do_load(switch_json, model_weight_path);
            }
        }
        
        FT_LOG_DEBUG(nccl_param.rank_, "done.");

        sync_check_cuda_error();

        cJSON_Delete(switch_json);

        print_all_tensor(nccl_param.rank_);
    }

    delete tensor_storage;
    mpi::finalize();

    return 0;
}
