#pragma once
#include <vector>

#include "3rdparty/cJSON.h"

#include "src/client/p2pOp.hpp"
#include "src/utils/memory_utils.h"
#include "src/utils/nccl_utils.h"
#include "src/utils/tcp_utils.h"
#include "src/client/layerConfig.h"
#include "src/client/TensorStorageLayer.h"

class TensorStorage{
protected:
    int rank;
    float* transpose_buf = nullptr;

    cudaStream_t cuda_stream;
    NcclParam& nccl_param;
    ParamConfig_t model_param_config;

    int layer_num;
    int hidden_size;
    TensorStorageLayer** storage_layer;

// ---------------------
    FloatParam_t cache_buffers;
    std::mutex cache_mutex;
    
    void append_kvcache_to_layers(int batchxbeam, int men_len);
    void alloc_kvcache_buffers(int batchxbeam, int men_len, int new_layern_node, int new_tp_grain);

public:
    TensorStorage(int _rank, cudaStream_t _cuda_stream, const ParamConfig_t& _param_config, NcclParam& _nccl_param);
    ~TensorStorage();

    void do_op(const cJSON* entry);
    // void do_op_nolock(const cJSON* entry);
    void do_load(const cJSON* entry, const std::string& model_dir);
    void print_test_params(int verbose);
    void load_initial_weights(const std::string& model_dir, int pp_para_size=1, int dp_para_size=1, int tp_para_size=1);
    void send_metadata(const TcpAgent& tcp_agent, int layer_id, int param_idx, int param_type);
    void send_buffdata(const TcpAgent& tcp_agent, int layer_id, int param_idx, int param_type);
    void alloc_new_buffer_and_send(const TcpAgent& tcp_agent, int layer_id, int param_idx, int param_type, int siz);
    void clear_weights();

    TensorStorageLayer** get_storage_layer() const { return storage_layer; }
};