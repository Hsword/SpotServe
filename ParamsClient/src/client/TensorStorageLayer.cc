#include "src/client/TensorStorageLayer.h"

TensorStorageLayer::TensorStorageLayer(int _layer_id, int _rank, cudaStream_t _cuda_stream, const LayerConfig_t& _layer_param_config, float* t_buf)
    : rank(_rank)
    , layer_id(_layer_id)
    , cuda_stream(_cuda_stream)
    , layer_param_config(_layer_param_config)
    , transpose_buf(t_buf)
{
    tp_params = new std::vector<std::shared_ptr<TensorWrapper<float>>>();
    dp_params = new std::vector<std::shared_ptr<TensorWrapper<float>>>();
    tp_buffs = new std::vector<std::shared_ptr<TensorWrapper<float>>>();
    dp_buffs = new std::vector<std::shared_ptr<TensorWrapper<float>>>();
}

std::shared_ptr<TensorWrapper<float>> TensorStorageLayer::allocate_tensor(const std::vector<int>& shape, bool is_trans, bool now_trans, bool is_buffer, bool set_zero){
    float* tensor_devPtr;
    int sz = 1;
    for(int dim : shape) sz *= dim;
    // FT_LOG_DEBUG(rank, "malloc before");
    deviceMalloc(&tensor_devPtr, sz, false);
    if(set_zero) cudaMemset((void*)tensor_devPtr, 0, sz * sizeof(float));
    // FT_LOG_DEBUG(rank, "layer_id: %d", layer_id);
    // FT_LOG_DEBUG(rank, "malloc end");

    return std::make_shared<TensorWrapper<float>>(tensor_devPtr, shape, cuda_stream, is_trans, now_trans, is_buffer);
}

std::shared_ptr<TensorWrapper<float>> TensorStorageLayer::allocate_tensor(float* devPtr, const std::vector<int>& shape, bool is_trans, bool now_trans, bool is_buffer, bool copy){
    // no malloc ver.
    
    float* tensor_devPtr = devPtr;
    if(copy){
        int sz = 1; for(int dim : shape) sz *= dim;
        deviceMalloc(&tensor_devPtr, sz, false);
        cudaD2Dcpy(tensor_devPtr, devPtr, sz);
    }
    return std::make_shared<TensorWrapper<float>>(tensor_devPtr, shape, cuda_stream, is_trans, now_trans, is_buffer, copy);
}


void TensorStorageLayer::switch_init(int new_tp_grain){
    int hidden_size = layer_param_config[0].shape[0]; // TODO MARIC: NEED MODIFIED
    tp_changed = (new_tp_grain != tp_grain);

    _recv_allocated = _recv_allocated_buff = false;
    _new_tp_stage = tp_stage;
    _transferring = true;

    temp_dp_params = NULL;
    temp_tp_params = NULL; //new std::vector<std::shared_ptr<TensorWrapper<float>>>();
    temp_tp_buffs = NULL;
    temp_dp_buffs = NULL;
}

void TensorStorageLayer::do_send_tp_param(const cJSON* entry, int new_tp_stage, int new_tp_grain, std::vector<P2pOp<float>>& p2p_ops){
    const cJSON* peer_json = cJSON_GetObjectItemCaseSensitive(entry, "peer");
    const cJSON* prior_json = cJSON_GetObjectItemCaseSensitive(entry, "prior");
    assert(cJSON_IsNumber(peer_json));
    assert(cJSON_IsBool(prior_json));
    int peer = peer_json->valueint;
    bool prior = cJSON_IsTrue(prior_json);

    transpose_all_for_comm(transpose_buf);

    for(auto iter = tp_params->begin(); iter != tp_params->end(); ++iter){
        float* sliced_tensor_devPtr;
        int data_size = 0;
        sliced_tensor_devPtr = (*iter)->get_sliced_tensor(new_tp_stage, new_tp_grain, tp_grain, data_size);
        p2p_ops.push_back(P2pOp<float>{P2pOp<float>::isend, sliced_tensor_devPtr, peer, data_size, cuda_stream, prior});
    }
}

void TensorStorageLayer::do_send_tp_param_plain_free(){
    if(temp_tp_params) delete temp_tp_params;
    temp_tp_params = nullptr;
}

void TensorStorageLayer::do_send_tp_param_plain(const cJSON* entry, std::vector<P2pOp<float>>& p2p_ops){
    const cJSON* peer_json = cJSON_GetObjectItemCaseSensitive(entry, "peer");
    const cJSON* prior_json = cJSON_GetObjectItemCaseSensitive(entry, "prior");
    assert(cJSON_IsNumber(peer_json));
    assert(cJSON_IsBool(prior_json));
    int peer = peer_json->valueint;
    bool prior = cJSON_IsTrue(prior_json);
    
    assert(temp_tp_params == NULL);
    temp_tp_params = new std::vector<std::shared_ptr<TensorWrapper<float>>>();

    for(auto iter = tp_params->begin(); iter != tp_params->end(); ++iter){
        int data_size = (*iter)->get_size();
        float* tensor_devPtr = (*iter)->get_tensor();
        if((*iter)->get_is_trans()){
            auto t = allocate_tensor((*iter)->get_shape(), true, true);
            (*iter)->transpose_to(t->get_tensor());
            p2p_ops.push_back(P2pOp<float>{P2pOp<float>::isend, t->get_tensor(), peer, data_size, cuda_stream, prior});
            temp_tp_params->push_back(t);
        }else{
            p2p_ops.push_back(P2pOp<float>{P2pOp<float>::isend, tensor_devPtr, peer, data_size, cuda_stream, prior});
        }
        
    }
}

void TensorStorageLayer::do_send_tp_buff(const cJSON* entry, int new_tp_stage, int new_tp_grain, std::vector<P2pOp<float>>& p2p_ops){
    const cJSON* peer_json = cJSON_GetObjectItemCaseSensitive(entry, "peer");
    assert(cJSON_IsNumber(peer_json));
    int peer = peer_json->valueint;
    // no transpose because transposed when appending

    for(auto iter = tp_buffs->begin(); iter != tp_buffs->end(); ++iter){
        float* sliced_tensor_devPtr;
        int data_size = 0;
        sliced_tensor_devPtr = (*iter)->get_sliced_tensor(new_tp_stage, new_tp_grain, tp_grain, data_size);
        p2p_ops.push_back(P2pOp<float>{P2pOp<float>::isend, sliced_tensor_devPtr, peer, data_size, cuda_stream, false});
    }
}

void TensorStorageLayer::do_send_dp_param(const cJSON* entry, std::vector<P2pOp<float>>& p2p_ops){
    const cJSON* peer_json = cJSON_GetObjectItemCaseSensitive(entry, "peer");
    const cJSON* prior_json = cJSON_GetObjectItemCaseSensitive(entry, "prior");
    assert(cJSON_IsNumber(peer_json));
    assert(cJSON_IsBool(prior_json));
    int peer = peer_json->valueint;
    bool prior = cJSON_IsTrue(prior_json);

    for(auto iter = dp_params->begin(); iter != dp_params->end(); ++iter){
        int data_size = (*iter)->get_size();
        float* tensor_devPtr = (*iter)->get_tensor();
        p2p_ops.push_back(P2pOp<float>{P2pOp<float>::isend, tensor_devPtr, peer, data_size, cuda_stream, prior});
    }
}

void TensorStorageLayer::do_send_dp_buff(const cJSON* entry, std::vector<P2pOp<float>>& p2p_ops){
    const cJSON* peer_json = cJSON_GetObjectItemCaseSensitive(entry, "peer");
    assert(cJSON_IsNumber(peer_json));
    int peer = peer_json->valueint;

    for(auto iter = dp_buffs->begin(); iter != dp_buffs->end(); ++iter){
        int data_size = (*iter)->get_size();
        float* tensor_devPtr = (*iter)->get_tensor();
        p2p_ops.push_back(P2pOp<float>{P2pOp<float>::isend, tensor_devPtr, peer, data_size, cuda_stream, false});
    }
}

void TensorStorageLayer::do_recv_allocated(const std::vector<int>& stay_ids, int new_tp_grain){
    if(_recv_allocated) return;
    _recv_allocated = true;
    if(temp_tp_params == NULL) temp_tp_params = new std::vector<std::shared_ptr<TensorWrapper<float>>>();

    transpose_all_for_comm(transpose_buf);

    auto& temp_params = *temp_tp_params;
    auto& params = *tp_params;

    assert(temp_tp_params->size() == 0);

    for(auto iter = layer_param_config.begin(); iter != layer_param_config.end(); ++iter){
        if(!iter->is_tp_param) continue;
        std::vector<int> new_shape{iter->shape};
        new_shape[0] /= new_tp_grain;
        temp_tp_params->push_back(allocate_tensor(new_shape, iter->is_transpose, tp_changed&&iter->is_transpose));
    }

    if(new_tp_grain >= tp_grain) return;
    // new_tp_grain < tp_grain
    for(int stay_id : stay_ids){
        if(stay_id < 0) continue;
        int layer_id, tp_stage, tp_grain;
        layer_id_decode(stay_id, layer_id, tp_stage, tp_grain);

        _new_tp_stage = tp_stage / (tp_grain / new_tp_grain);
        for(int i = 0; i < tp_params->size(); i++){
            int tensor_size = params[i]->get_size();
            float* tensor_Ptr = params[i]->get_tensor();
            temp_params[i]->assign_sliced_tensor(tp_stage, tp_grain, new_tp_grain, tensor_Ptr, tensor_size);
        }

    }
}

void TensorStorageLayer::do_recv_allocated_buff(const std::vector<int>& stay_ids, int new_tp_grain, const FloatParam_t& cache_buffers, const std::vector<int> *shape, const int layer_offset){
    if(_recv_allocated_buff) return;
    _recv_allocated_buff = true;
    if(temp_tp_buffs == NULL) temp_tp_buffs = new std::vector<std::shared_ptr<TensorWrapper<float>>>();

    // transpose_all_for_comm_buff(transpose_buf); // alloc, no valid data

    auto& temp_buffs = *temp_tp_buffs;
    auto& buffs = *tp_buffs;

    assert(temp_tp_buffs->size() == 0);

    for(auto iter = cache_buffers.begin(); iter != cache_buffers.end(); ++iter){
        float* devPtr = (*iter)->get_tensor();
        temp_tp_buffs->push_back(allocate_tensor(devPtr + layer_offset, *shape, true, true, true, false));
    }

    if(new_tp_grain >= tp_grain) return;
    // new_tp_grain < tp_grain
    for(int stay_id : stay_ids){
        if(stay_id < 0) continue;
        int layer_id, tp_stage, tp_grain;
        layer_id_decode(stay_id, layer_id, tp_stage, tp_grain);

        _new_tp_stage = tp_stage / (tp_grain / new_tp_grain);
        for(int i = 0; i < tp_buffs->size(); i++){
            int tensor_size = buffs[i]->get_size();
            float* tensor_Ptr = buffs[i]->get_tensor();
            temp_buffs[i]->assign_sliced_tensor(tp_stage, tp_grain, new_tp_grain, tensor_Ptr, tensor_size);
        }

    }
}

void TensorStorageLayer::do_recv_tp_param(const cJSON* entry, int new_tp_grain, int r_tp_stage, int r_tp_grain, std::vector<P2pOp<float>>& p2p_ops){
    const cJSON* peer_json = cJSON_GetObjectItemCaseSensitive(entry, "peer");
    const cJSON* prior_json = cJSON_GetObjectItemCaseSensitive(entry, "prior");
    assert(cJSON_IsNumber(peer_json));
    assert(cJSON_IsBool(prior_json));
    int peer = peer_json->valueint;
    bool prior = cJSON_IsTrue(prior_json);

    // already transposed in do_recv_allocated

    auto& temp_params = *temp_tp_params;
    auto& params = *tp_params;
    // auto params_config = *layer_param_config;

    for(int i = 0; i < temp_tp_params->size(); i++){
        // if(!params_config[i].is_tp_param) continue;
        float* sliced_tensor_devPtr;
        int data_size = 0;
        sliced_tensor_devPtr = temp_params[i]->get_sliced_tensor(r_tp_stage, r_tp_grain, new_tp_grain, data_size);
        p2p_ops.push_back(P2pOp<float>{P2pOp<float>::irecv, sliced_tensor_devPtr, peer, data_size, cuda_stream, prior});
    }
}

void TensorStorageLayer::do_recv_tp_buff(const cJSON* entry, int new_tp_grain, int r_tp_stage, int r_tp_grain, std::vector<P2pOp<float>>& p2p_ops){
    const cJSON* peer_json = cJSON_GetObjectItemCaseSensitive(entry, "peer");
    assert(cJSON_IsNumber(peer_json));
    int peer = peer_json->valueint;

    auto& temp_buffs = *temp_tp_buffs;
    auto& buffs = *tp_buffs;

    for(int i = 0; i < temp_tp_buffs->size(); i++){
        float* sliced_tensor_devPtr;
        int data_size = 0;
        sliced_tensor_devPtr = temp_buffs[i]->get_sliced_tensor(r_tp_stage, r_tp_grain, new_tp_grain, data_size);
        p2p_ops.push_back(P2pOp<float>{P2pOp<float>::irecv, sliced_tensor_devPtr, peer, data_size, cuda_stream, false});
    }
}

void TensorStorageLayer::do_recv_dp_param(const cJSON* entry, std::vector<P2pOp<float>>& p2p_ops){
    const cJSON* peer_json = cJSON_GetObjectItemCaseSensitive(entry, "peer");
    const cJSON* prior_json = cJSON_GetObjectItemCaseSensitive(entry, "prior");
    assert(cJSON_IsNumber(peer_json));
    assert(cJSON_IsBool(prior_json));
    int peer = peer_json->valueint;
    bool prior = cJSON_IsTrue(prior_json);

    if(temp_dp_params == NULL) temp_dp_params = new std::vector<std::shared_ptr<TensorWrapper<float>>>();

    auto& temp_params = *temp_dp_params;
    auto& params = *dp_params;

    int idx = 0;
    for(auto iter = layer_param_config.begin(); iter != layer_param_config.end(); ++iter){
        if(iter->is_tp_param) continue;
        temp_params.push_back(allocate_tensor(iter->shape, iter->is_transpose, tp_changed&&iter->is_transpose));
        int data_size = temp_params[idx]->get_size();
        float* tensor_devPtr = temp_params[idx]->get_tensor();
        p2p_ops.push_back(P2pOp<float>{P2pOp<float>::irecv, tensor_devPtr, peer, data_size, cuda_stream, prior});

        ++idx;
    }
}

void TensorStorageLayer::do_recv_dp_buff(const cJSON* entry, std::vector<P2pOp<float>>& p2p_ops, 
    int batchxbeam, int max_session_len, int hidden_size, int mem_len, int beam){
    const cJSON* peer_json = cJSON_GetObjectItemCaseSensitive(entry, "peer");
    assert(cJSON_IsNumber(peer_json));
    int peer = peer_json->valueint;

    if(temp_dp_buffs == NULL) temp_dp_buffs = new std::vector<std::shared_ptr<TensorWrapper<float>>>();

    auto& temp_buffs = *temp_dp_buffs;
    auto& buffs = *dp_buffs;

    int idx = 0;
    temp_buffs.push_back(allocate_tensor({batchxbeam * hidden_size}, false, false, true)); // decoder_input_buf_
    temp_buffs.push_back(allocate_tensor({batchxbeam * 4}, false, false, true)); // combined_buffer_pointer
    temp_buffs.push_back(allocate_tensor({batchxbeam * max_session_len}, false, false, true)); // output_ids_buf_
    temp_buffs.push_back(allocate_tensor({batchxbeam * max_session_len}, false, false, true)); // parent_ids_buf_
    temp_buffs.push_back(allocate_tensor({batchxbeam * max_session_len}, false, false, true)); // output_log_probs_buf_
    if(beam > 1)
        temp_buffs.push_back(allocate_tensor({batchxbeam * mem_len * 2}, false, false, true)); // cache_indirections_

    for(; idx < temp_buffs.size(); ++idx){
        int data_size = temp_buffs[idx]->get_size();
        float* tensor_devPtr = temp_buffs[idx]->get_tensor();
        p2p_ops.push_back(P2pOp<float>{P2pOp<float>::irecv, tensor_devPtr, peer, data_size, cuda_stream, false});
    }
}

void TensorStorageLayer::do_stay_tp_param(int stay_id, int new_tp_grain){
    if(new_tp_grain < tp_grain) return;
    assert(temp_tp_params == NULL);
    if(!tp_changed) {
        do_stay_tp_param();
        return;
    }
    int a, new_tp_stage, b;
    
    temp_tp_params = new std::vector<std::shared_ptr<TensorWrapper<float>>>();
    auto& params = *tp_params;

    layer_id_decode(stay_id, a, new_tp_stage, b);
    _new_tp_stage = new_tp_stage;

    transpose_all_for_comm(transpose_buf);

    // buffer already in tp_params
    for(int i = 0; i < tp_params->size(); ++i){
        params[i]->slice_tensor_and_delete(new_tp_stage, new_tp_grain, tp_grain, nullptr);
        temp_tp_params->push_back(params[i]);
    }
}

void TensorStorageLayer::do_stay_tp_buff(int stay_id, int new_tp_grain, const FloatParam_t& cache_buffers, const int layer_offset){
    if(new_tp_grain < tp_grain) return;
    int a, new_tp_stage, b;
    
    assert(temp_tp_buffs == NULL);
    temp_tp_buffs = new std::vector<std::shared_ptr<TensorWrapper<float>>>();
    auto& buffs = *tp_buffs;

    layer_id_decode(stay_id, a, new_tp_stage, b);
    _new_tp_stage = new_tp_stage;

    int buffer_idx = 0;
    for(int i = 0; i < tp_buffs->size(); ++i){
        buffs[i]->slice_tensor_and_delete(new_tp_stage, new_tp_grain, tp_grain, cache_buffers[buffer_idx++]->get_tensor() + layer_offset);
        temp_tp_buffs->push_back(buffs[i]);
    }
}

void TensorStorageLayer::switch_end(int new_tp_grain){
    if(dp_params != temp_dp_params){
        if(dp_params) delete dp_params;
        if(temp_dp_params) dp_params = temp_dp_params;
        else dp_params =  new std::vector<std::shared_ptr<TensorWrapper<float>>>();
        temp_dp_params = nullptr;
    } 
    if(tp_params != temp_tp_params){
        if(tp_params) delete tp_params;
        if(temp_tp_params) tp_params = temp_tp_params;
        else tp_params =  new std::vector<std::shared_ptr<TensorWrapper<float>>>();
        temp_tp_params = nullptr;
    } 

    _transferring = false;
    tp_grain = new_tp_grain;
    tp_stage = _new_tp_stage;
}

void TensorStorageLayer::switch_end_buff(){
    if(dp_buffs != temp_dp_buffs){
        if(dp_buffs) delete dp_buffs;
        if(temp_dp_buffs) dp_buffs = temp_dp_buffs;
        else dp_buffs =  new std::vector<std::shared_ptr<TensorWrapper<float>>>();
        temp_dp_buffs = nullptr;
    } 
    if(tp_buffs != temp_tp_buffs){
        if(tp_buffs) delete tp_buffs;
        if(temp_tp_buffs) tp_buffs = temp_tp_buffs;
        else tp_buffs =  new std::vector<std::shared_ptr<TensorWrapper<float>>>();
        temp_tp_buffs = nullptr;
    } 
}

TensorStorageLayer::~TensorStorageLayer(){
    delete tp_params;
    delete dp_params;
}

void TensorStorageLayer::print_params(int verbose){
    printf("---------------------layer: %d-------------------\n", layer_id);
    int cnt = 0;
    for(auto iter = tp_params->begin(); iter != tp_params->end(); ++iter){
        printf("<tp_param %d>\n", cnt);
        ++cnt;
        (*iter)->print_data(verbose);
    }

    for(auto iter = dp_params->begin(); iter != dp_params->end(); ++iter){
        printf("<dp_param %d>\n", cnt);
        ++cnt;
        (*iter)->print_data(verbose);
    }

    for(auto iter = tp_buffs->begin(); iter != tp_buffs->end(); ++iter){
        printf("<tp_buff %d>\n", cnt);
        ++cnt;
        (*iter)->print_data(verbose);
    }

    for(auto iter = dp_buffs->begin(); iter != dp_buffs->end(); ++iter){
        printf("<dp_buff %d>\n", cnt);
        ++cnt;
        (*iter)->print_data(verbose);
    }
}

void TensorStorageLayer::clear_weights(){
    if(dp_params) delete dp_params;
    if(tp_params) delete tp_params;
    if(dp_buffs) delete dp_buffs;
    if(tp_buffs) delete tp_buffs;
    dp_params =  new std::vector<std::shared_ptr<TensorWrapper<float>>>();
    tp_params =  new std::vector<std::shared_ptr<TensorWrapper<float>>>();
    dp_buffs =  new std::vector<std::shared_ptr<TensorWrapper<float>>>();
    tp_buffs =  new std::vector<std::shared_ptr<TensorWrapper<float>>>();
}

void TensorStorageLayer::load_initial_weights(const std::string& model_dir, bool lock, int tp_size, int _tp_stage){
    if(lock) in_comm.lock();

    tp_grain = tp_size;
    tp_stage = _tp_stage;
    
    for(auto iter = layer_param_config.begin(); iter != layer_param_config.end(); ++iter){
        std::vector<int> real_shape(iter->shape);
        // not correct for tp, only size is match
        if (iter->is_tp_param){
            real_shape[0] /= tp_size;
            // FT_LOG_DEBUG(rank, "%d %d",  real_shape[0], iter->shape[0]);
        }
        auto tensor = allocate_tensor(real_shape, iter->is_transpose, false);
        tensor->load_init_value(model_dir + iter->name);

        if(iter->is_tp_param){
            tp_params->push_back(tensor);
        }
        else{
            dp_params->push_back(tensor);
        }
    }
    if(lock) in_comm.unlock();
}

void TensorStorageLayer::send_metadata(const TcpAgent& tcp_agent, int param_idx, int param_type){
    std::lock_guard<std::mutex> lock(in_comm); // lock for sync 

    if(param_type == 9){
        TensorMetaTransit_t tmp;
        tcp_agent.tcpSend(&tmp, sizeof(TensorMetaTransit_t));
        return;
    }

    FT_CHECK_WITH_INFO((unsigned)param_type < 4, "param_idx shoue be in 0~4");

    switch (param_type)
    {
    case 0:
        tp_params->at(param_idx)->send_metadata(tcp_agent);
        break;
    case 1:
        dp_params->at(param_idx)->send_metadata(tcp_agent);
        break;
    case 2:
        tp_buffs->at(param_idx)->send_metadata(tcp_agent);
        break;
    case 3:
        dp_buffs->at(param_idx)->send_metadata(tcp_agent);
        break;
    }
}

// static bool printed_1 = false, printed_2 = false;
void TensorStorageLayer::transpose_all_for_comm(float* buf){
    if(!tp_changed || !for_compute) return;
    for(auto iter = tp_params->begin(); iter != tp_params->end(); ++iter){
        // if(layer_id == 0 && (*iter)->get_is_buffer()){
        //     FT_LOG_DEBUG(rank, "buffer transpose_all_for_comm");
        // }
        (*iter)->transpose_inplace(buf, false);
        // FT_LOG_DEBUG(rank, " trans buffer=%d, sz=%d", (*iter)->get_is_buffer(), (*iter)->get_size());
    }
    check_cuda_error(cudaDeviceSynchronize());
    // sync_check_cuda_error();
    for_compute = false;
}

void TensorStorageLayer::transpose_all_for_comp(float* buf){
    if(!tp_changed || for_compute) return;
    for(auto iter = tp_params->begin(); iter != tp_params->end(); ++iter){
        // if(layer_id == 0 && (*iter)->get_is_buffer()){
        //     FT_LOG_DEBUG(rank, "buffer transpose_all_for_comp");
        // }
        (*iter)->transpose_inplace(buf, true);
    }
    for_compute = true;
    check_cuda_error(cudaDeviceSynchronize());
    // sync_check_cuda_error();
}

void TensorStorageLayer::transpose_buff_for_comp_and_delete(float* buf){
    for(auto iter = tp_buffs->begin(); iter != tp_buffs->end(); ++iter){
        (*iter)->transpose_inplace(buf, true);
    }
    check_cuda_error(cudaDeviceSynchronize());
    delete tp_buffs;
    tp_buffs = new std::vector<std::shared_ptr<TensorWrapper<float>>>();
}

void TensorStorageLayer::append_tp_buffer(float* devPtr, const std::vector<int>& shape, bool is_trans, bool now_trans){
    // FT_LOG_DEBUG(rank, "append_tp_buffer ptr=%p, tp_buffs=%p", devPtr, tp_buffs);
    auto new_devPtr = allocate_tensor(shape, is_trans, now_trans, true);
    tp_buffs->push_back(new_devPtr);
    if(is_trans){
        // transpose when appending
        invokeMatrixTranspose(new_devPtr->get_tensor(), devPtr, shape[1], shape[0], cuda_stream);
    }
}

void TensorStorageLayer::allocate_dp_buffer(const TcpAgent& tcp_agent, int idx, const std::vector<int>& shape){
    FT_CHECK_WITH_INFO(shape.size() == 1, "allocate_dp_buffer() shape's dim shoule be 1");

    std::lock_guard<std::mutex> lock(in_comm); // lock for sync 
    if(dp_buffs->size() > idx){
        if(dp_buffs->at(idx)->get_size() == shape[0]){
            // reuse
            cudaMemset(dp_buffs->at(idx)->get_tensor(), 0, sizeof(float) * shape[0]);
            dp_buffs->at(idx)->send_metadata(tcp_agent, 1);
            return;
        }
    }
    if(dp_buffs->size()!=idx){
        FT_LOG_INFO("dp_buffs' size %ld is unequal to idx %d, cause realloc", dp_buffs->size(), idx);
    }
    
    auto buf_p = allocate_tensor(shape, false, false, true);
    if(dp_buffs->size() > idx){
        (*dp_buffs)[idx] = buf_p;
    }else{
        dp_buffs->push_back(buf_p);
    }
    FT_LOG_DEBUG(rank, "size of dp buffs: %ld", dp_buffs->size());
    buf_p->send_metadata(tcp_agent);
}