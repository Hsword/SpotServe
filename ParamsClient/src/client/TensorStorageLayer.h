#pragma once
#include <vector>
#include <memory>
#include <string>
#include <mutex>

#include "3rdparty/cJSON.h"

#include "src/client/p2pOp.hpp"
#include "src/utils/memory_utils.h"
#include "src/client/layerConfig.h"
#include "src/client/TensorWrapper.hpp"

typedef std::vector<std::shared_ptr<TensorWrapper<float>>> FloatParam_t;

class TensorStorageLayer{
private:
    int _new_tp_stage;
    bool _recv_allocated = false, _recv_allocated_buff = false;
    bool _switching_buffers = false;
    bool _transferring = false;
protected:
    int rank;
    cudaStream_t cuda_stream;
    LayerConfig_t layer_param_config;
    bool for_compute = true; // comp <<-- (transpose) -->> comm
    float* transpose_buf = nullptr;

    int layer_id;
    int tp_grain = 1;
    int tp_stage = 0;
    bool tp_changed;

    // std::vector<P2pOp<float>> p2p_ops;
    FloatParam_t *tp_params, *temp_tp_params, *dp_params, *temp_dp_params;
    FloatParam_t *tp_buffs, *temp_tp_buffs, *dp_buffs, *temp_dp_buffs;

public:
    std::mutex in_comm;

    TensorStorageLayer(int _layer_id, int _rank, cudaStream_t _cuda_stream, const LayerConfig_t& _layer_param_config, float* t_buf);
    ~TensorStorageLayer();

    TensorStorageLayer(const TensorStorageLayer&) = delete;
    TensorStorageLayer& operator=(const TensorStorageLayer&) = delete;

    void switch_init(int new_tp_grain);
    // for params
    void do_send_tp_param(const cJSON* entry, int new_tp_stage, int new_tp_grain, std::vector<P2pOp<float>>& p2p_ops);
    void do_send_tp_param_plain(const cJSON* entry, std::vector<P2pOp<float>>& p2p_ops);
    void do_send_tp_param_plain_free();
    void do_send_dp_param(const cJSON* entry, std::vector<P2pOp<float>>& p2p_ops);
    void do_recv_allocated(const std::vector<int>& stay_ids, int new_tp_grain);
    void do_recv_tp_param(const cJSON* entry, int new_tp_grain, int r_tp_stage, int r_tp_grain, std::vector<P2pOp<float>>& p2p_ops);
    void do_recv_dp_param(const cJSON* entry, std::vector<P2pOp<float>>& p2p_ops);
    void do_stay_tp_param(int stay_id, int new_tp_grain);
    void do_stay_tp_param() {temp_tp_params = tp_params;}
    void do_stay_dp_param() {temp_dp_params = dp_params;}

    // for buffs
    void do_send_tp_buff(const cJSON* entry, int new_tp_stage, int new_tp_grain, std::vector<P2pOp<float>>& p2p_ops);
    void do_send_dp_buff(const cJSON* entry, std::vector<P2pOp<float>>& p2p_ops);
    void do_recv_allocated_buff(const std::vector<int>& stay_ids, int new_tp_grain, const FloatParam_t& cache_buffers, const std::vector<int> *shape, const int layer_offset);
    void do_recv_tp_buff(const cJSON* entry, int new_tp_grain, int r_tp_stage, int r_tp_grain, std::vector<P2pOp<float>>& p2p_ops);
    void do_recv_dp_buff(const cJSON* entry, std::vector<P2pOp<float>>& p2p_ops,
        int batchxbeam, int max_session_len, int hidden_size, int mem_len, int beam);
    void do_stay_tp_buff(int stay_id, int new_tp_grain, const FloatParam_t& cache_buffers, const int layer_offset);
    void do_stay_tp_buff() {temp_tp_buffs = tp_buffs;}
    void do_stay_dp_buff() {temp_dp_buffs = dp_buffs;}

    void switch_end(int new_tp_grain);
    void switch_end_buff();

    void load_initial_weights(const std::string& model_dir, bool lock=true, int tp_size=1, int _tp_stage=0);
    void clear_weights();
    void print_params(int verbose);

    void send_metadata(const TcpAgent& tcp_agent, int param_idx, int param_type);
    void transpose_all_for_comm(float* buf);
    void transpose_all_for_comp(float* buf);
    void transpose_buff_for_comp_and_delete(float* buf);

    void append_tp_buffer(float* devPtr, const std::vector<int>& shape, bool is_trans=true, bool now_trans=false);
    void allocate_dp_buffer(const TcpAgent& tcp_agent, int idx, const std::vector<int>& shape);

    bool is_empty_layer() const {return tp_params->size()==0 && dp_params->size() == 0;}
    // pp layer need to be consecutive and only one group on single device
    int get_tp_degree() const {return tp_grain; }
    bool get_is_transferring() const {return _transferring; }

    std::shared_ptr<TensorWrapper<float>> allocate_tensor(const std::vector<int>& shape, bool is_trans, bool now_trans, bool is_buffer=false, bool set_zero=false);
    std::shared_ptr<TensorWrapper<float>> allocate_tensor(float* devPtr, const std::vector<int>& shape, bool is_trans, bool now_trans, bool is_buffer=true, bool copy=false);
};