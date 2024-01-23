#include "src/client/TensorStorage.h"
#include <unistd.h>

static inline bool isValidLayer(int rank, int pp_size, int layer_id, int layer_num){
    // printf("[%d] %d %d\n", rank, layer_id, layer_id * pp_size / (layer_num - 2));
    return rank == layer_id * pp_size / (layer_num - 2); // MAGIC NUMBER HERE
}

TensorStorage::TensorStorage(int _rank, cudaStream_t _cuda_stream, const ParamConfig_t& _param_config, NcclParam& _nccl_param)
    : rank(_rank)
    , cuda_stream(_cuda_stream)
    , model_param_config(_param_config)
    , nccl_param(_nccl_param)
{
    layer_num = model_param_config.size();

    storage_layer = new TensorStorageLayer*[layer_num];
    int max_size = 0;
    for(int i = 0; i < layer_num; i++){
        for(auto const& p : model_param_config[i]){
            if(p.is_transpose){
                int siz = 1;
                for(auto dim : p.shape){ siz *= dim; }
                if(siz > max_size) max_size = siz;
            }
        }
    }
    deviceMalloc(&transpose_buf, max_size, false);

    for(int i = 0; i < layer_num; ++i){
        storage_layer[i] = new TensorStorageLayer(i, rank, cuda_stream, model_param_config[i], transpose_buf);
    }

    hidden_size = model_param_config[0][0].shape[0]; // MAGIC
}

void TensorStorage::clear_weights(){
    for(int i = 0; i < layer_num; ++i){
        storage_layer[i]->clear_weights();
    }
}

void TensorStorage::do_load(const cJSON* entry, const std::string& model_dir){
    for(int i = 0; i < layer_num; ++i){
        FT_CHECK_WITH_INFO(storage_layer[i]->is_empty_layer(), fmtstr("layer %d is not empty!", i));
        storage_layer[i]->in_comm.lock();
    }
    int pp = cJSON_GetObjectItemCaseSensitive(entry, "pp")->valueint;
    int tp = cJSON_GetObjectItemCaseSensitive(entry, "tp")->valueint;
    int pp_stage = cJSON_GetObjectItemCaseSensitive(entry, "pp_stage")->valueint;
    int tp_stage = cJSON_GetObjectItemCaseSensitive(entry, "tp_stage")->valueint;
    
    for(int i = 0; i < layer_num; i++){
        if(i == layer_num - 2){ // MAGIC NUMBER HERE
            storage_layer[i]->load_initial_weights(model_dir + "model", false);
        }else if(i == layer_num - 1){ // post-decoder layer
            if(pp_stage == pp - 1) // no tp
                storage_layer[i]->load_initial_weights(model_dir + "model", false);
        }else if(isValidLayer(pp_stage, pp, i, layer_num)){
            storage_layer[i]->load_initial_weights(model_dir + "model.layers." + std::to_string(i), false, tp, tp_stage);
        }
    }

    for(int i = 0; i < layer_num; ++i){
        storage_layer[i]->in_comm.unlock();
    }
}

void TensorStorage::append_kvcache_to_layers(int batchxbeam, int mem_len){
    // org_shape [layers_this_node, batchxbeam, hidden_size/tp, memory_len]
    // shape [layers_this_node * batchxbeam * hidden_size / tp * memory_len]
    // shape in layer [hidden_size / tp * memory_len, batchxbeam], is_transpose=true, now_transpose=false
    FT_LOG_DEBUG(rank, "cache_buffers size = %ld", cache_buffers.size());

    for(auto iter = cache_buffers.begin(); iter != cache_buffers.end(); ++iter){
        size_t cache_offset = 0;

        for(int i = 0; i < layer_num - 2; ++i){ // MAGIC NUMBER HERE, exclude non-decoder layer
            if(storage_layer[i]->is_empty_layer()) continue;

            int tp_degree = storage_layer[i]->get_tp_degree();
            size_t cache_size = (size_t)batchxbeam * hidden_size / tp_degree * mem_len;
            storage_layer[i]->append_tp_buffer((*iter)->get_tensor() + cache_offset, {hidden_size / tp_degree * mem_len, batchxbeam}, true, true);

            cache_offset += cache_size;
        }
        FT_CHECK_WITH_INFO((*iter)->get_size() == cache_offset,
            fmtstr("Inconsistent size of original buffer size %lld and total cache size %lld ", (*iter)->get_size(), cache_offset));
        check_cuda_error(cudaDeviceSynchronize()); // for transpose
        
    }
}

void TensorStorage::alloc_kvcache_buffers(int batchxbeam, int mem_len, int layers_this_node, int new_tp_grain){
    if(cache_buffers.size() > 0) return;
    // FT_CHECK_WITH_INFO(cache_buffers.size() == 0, "cache buffer allocation is supposed to be called when no cache_buffer exists.");
    size_t buffer_size = (size_t)layers_this_node * batchxbeam * hidden_size / new_tp_grain * mem_len;
    FT_LOG_DEBUG(rank, "alloc_kvcache_buffers size=%ld", buffer_size);

    std::vector<int> shape;
    shape.push_back(buffer_size);

    float* devPtr;
    deviceMalloc(&devPtr, buffer_size, true); // k_cache
    auto k_cache = std::make_shared<TensorWrapper<float>>(devPtr, shape, cuda_stream, false, true, true, true);
    cache_buffers.push_back(k_cache);

    deviceMalloc(&devPtr, buffer_size, true); // v_cache
    auto v_cache = std::make_shared<TensorWrapper<float>>(devPtr, shape, cuda_stream, false, true, true, true);
    cache_buffers.push_back(v_cache);
}

void TensorStorage::do_op(const cJSON* entry){
    int pre_decoder_layer = layer_num - 2; // magic number here
    int new_tp_grain = cJSON_GetObjectItemCaseSensitive(entry, "tp")->valueint;
    cJSON* nolock_json = cJSON_GetObjectItemCaseSensitive(entry, "nolock");
    bool no_lock = cJSON_IsTrue(nolock_json);
    if(no_lock) {
        int old_tp = 0;
        for(int i = 0; i < layer_num - 2; i++)
            if(!storage_layer[i]->is_empty_layer()){
                old_tp = storage_layer[i]->get_tp_degree();
                break;
            }
        FT_CHECK_WITH_INFO(new_tp_grain==old_tp, fmtstr("new_tp_grain=%d != old_tp=%d", new_tp_grain, old_tp));
    }
    bool ignore_buffer = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(entry, "ignore_buff"));

    if(!ignore_buffer) cache_mutex.lock();
    for(int i = 0; i < layer_num; ++i){
        storage_layer[i]->in_comm.lock();
    }
    double sleep_time = -1; // <0 disable, otherwise: no overlap, release lock at end
    cJSON* sleep_json = cJSON_GetObjectItemCaseSensitive(entry, "sleep");
    if(cJSON_IsNumber(sleep_json)) sleep_time = sleep_json->valuedouble;

    cJSON* order_json =  cJSON_GetObjectItemCaseSensitive(entry, "order");
    bool memory_efficient_switch = cJSON_IsArray(order_json); // if on, disable dyingrank first
    std::vector<int> transfer_order;
    if(memory_efficient_switch){
        cJSON* ele;
        cJSON_ArrayForEach(ele, order_json){
            transfer_order.push_back(ele->valueint);
        }
        FT_CHECK(transfer_order.size() == layer_num);
    }

    cJSON* buffer_json = cJSON_GetObjectItemCaseSensitive(entry, "buffer");
    BufferConfig old_buffer, new_buffer;
    cJSON* buffer_old_json = cJSON_GetObjectItemCaseSensitive(buffer_json, "old");
    cJSON* buffer_new_json = cJSON_GetObjectItemCaseSensitive(buffer_json, "new");
    bool has_old_buffer_info = (!ignore_buffer) && cJSON_IsObject(buffer_old_json);
    bool has_new_buffer_info = (!ignore_buffer) && cJSON_IsObject(buffer_new_json);
    if(has_old_buffer_info) old_buffer.set_fields_by_json(buffer_old_json);
    if(has_new_buffer_info) new_buffer.set_fields_by_json(buffer_new_json);

    if(has_old_buffer_info){
        append_kvcache_to_layers(old_buffer.batchxbeam, old_buffer.mem_len);
    }
    if(!ignore_buffer){
        cache_buffers.clear(); // no need or copied to layer_params
        if(has_new_buffer_info){
            alloc_kvcache_buffers(new_buffer.batchxbeam, new_buffer.mem_len, new_buffer.new_layers_per_device, new_tp_grain);
        }
    }

    for(int i = 0; i < layer_num; ++i){
        storage_layer[i]->switch_init(new_tp_grain);
    }

    // parse stay param json array
    std::vector<std::vector<int>> stay_ids(layer_num);
    cJSON* stay_ids_json = cJSON_GetObjectItemCaseSensitive(entry, "pstay");
    cJSON* stay_id_json;
    if(cJSON_IsArray(stay_ids_json)){
        FT_LOG_DEBUG(rank, "parse_Pstay");
        cJSON_ArrayForEach(stay_id_json, stay_ids_json){
            if(!cJSON_IsNumber(stay_id_json)) continue;
            int stay_id = stay_id_json->valueint;
            int layer_id, tp_stage, tp_grain;
            if(stay_id < 0)
                layer_id = -stay_id - 1;
            else
                layer_id_decode(stay_id, layer_id, tp_stage, tp_grain);
            stay_ids[layer_id].push_back(stay_id);
        }
    }
    // parse stay buff json array
    std::vector<std::vector<int>> stay_ids_b(layer_num);
    stay_ids_json = cJSON_GetObjectItemCaseSensitive(entry, "bstay");
    bool has_bstay = false;
    if(!ignore_buffer && cJSON_IsArray(stay_ids_json)){
        FT_LOG_DEBUG(rank, "parse_Bstay");
        cJSON_ArrayForEach(stay_id_json, stay_ids_json){
            if(!cJSON_IsNumber(stay_id_json)) continue;
            int stay_id = stay_id_json->valueint;
            int layer_id, tp_stage, tp_grain;
            if(stay_id < 0)
                layer_id = -stay_id - 1;
            else
                layer_id_decode(stay_id, layer_id, tp_stage, tp_grain);
            has_bstay = true;
            stay_ids_b[layer_id].push_back(stay_id);
        }
    }
    // if(no_lock && !has_bstay) cache_buffers.clear();

    std::vector<std::vector<P2pOp<float>>> p2p_ops(layer_num);
    std::vector<std::vector<P2pOp<float>>> p2p_ops_buff(layer_num);

    // parse send param json array
    cJSON* send_ops_json = cJSON_GetObjectItemCaseSensitive(entry, "psend");
    cJSON* send_op_json;
    if(cJSON_IsArray(send_ops_json)){
        FT_LOG_DEBUG(rank, "parse_Psend");
        cJSON_ArrayForEach(send_op_json, send_ops_json){
            int lid = cJSON_GetObjectItemCaseSensitive(send_op_json, "layer")->valueint;
            if(lid < 0){
                int layer_id = -lid - 1;
                storage_layer[layer_id]->do_send_dp_param(send_op_json, p2p_ops[layer_id]);
            }else{
                int layer_id, tp_stage, tp_grain;
                layer_id_decode(lid, layer_id, tp_stage, tp_grain);
                storage_layer[layer_id]->do_send_tp_param(send_op_json, tp_stage, tp_grain, p2p_ops[layer_id]);
            }
        }
    }
    // parse send buff json array
    send_ops_json = cJSON_GetObjectItemCaseSensitive(entry, "bsend");
    if(!ignore_buffer && cJSON_IsArray(send_ops_json)){
        FT_LOG_DEBUG(rank, "parse_Bsend");
        cJSON_ArrayForEach(send_op_json, send_ops_json){
            int lid = cJSON_GetObjectItemCaseSensitive(send_op_json, "layer")->valueint;
            if(lid < 0){
                int layer_id = -lid - 1;
                storage_layer[layer_id]->do_send_dp_buff(send_op_json, p2p_ops_buff[layer_id]);
            }else{
                int layer_id, tp_stage, tp_grain;
                layer_id_decode(lid, layer_id, tp_stage, tp_grain);
                storage_layer[layer_id]->do_send_tp_buff(send_op_json, tp_stage, tp_grain, p2p_ops_buff[layer_id]);
            }
        }
    }
    
    // parse recv param json array
    std::vector<std::vector<cJSON*>> recv_ops(layer_num);

    cJSON* recv_ops_json = cJSON_GetObjectItemCaseSensitive(entry, "precv");
    cJSON* recv_op_json;
    if(cJSON_IsArray(recv_ops_json)){
        FT_LOG_DEBUG(rank, "parse_Precv_reg");
        cJSON_ArrayForEach(recv_op_json, recv_ops_json){
            FT_CHECK(no_lock == false);
            int lid = cJSON_GetObjectItemCaseSensitive(recv_op_json, "layer")->valueint;
            if(memory_efficient_switch){
                int layer_id;
                if(lid < 0){
                    layer_id = -lid - 1;
                }else{
                    int tp_stage, tp_grain;
                    layer_id_decode(lid, layer_id, tp_stage, tp_grain);
                }
                recv_ops[layer_id].push_back(recv_op_json);
            }else{
                if(lid < 0){
                    int layer_id = -lid - 1;
                    storage_layer[layer_id]->do_recv_dp_param(recv_op_json, p2p_ops[layer_id]);
                }else{
                    int layer_id, tp_stage, tp_grain;
                    layer_id_decode(lid, layer_id, tp_stage, tp_grain);
                    storage_layer[layer_id]->do_recv_allocated(stay_ids[layer_id], new_tp_grain);
                    storage_layer[layer_id]->do_recv_tp_param(recv_op_json, new_tp_grain, tp_stage, tp_grain, p2p_ops[layer_id]);
                }
            }
        }
    }

    // parse recv buff json array
    recv_ops_json = cJSON_GetObjectItemCaseSensitive(entry, "brecv");
    if(!ignore_buffer && cJSON_IsArray(recv_ops_json)){
        FT_LOG_DEBUG(rank, "parse_Brecv");
        cJSON_ArrayForEach(recv_op_json, recv_ops_json){
            // FT_CHECK(no_lock == false);
            int lid = cJSON_GetObjectItemCaseSensitive(recv_op_json, "layer")->valueint;
            if(lid < 0){
                int layer_id = -lid - 1;
                FT_CHECK(layer_id == pre_decoder_layer);
                storage_layer[layer_id]->do_recv_dp_buff(recv_op_json, p2p_ops_buff[layer_id],
                    new_buffer.batchxbeam, new_buffer.session_len, hidden_size, new_buffer.mem_len, new_buffer.beam);
            }else{
                int layer_id, tp_stage, tp_grain;
                layer_id_decode(lid, layer_id, tp_stage, tp_grain);
                // shape in layer [hidden_size / tp * memory_len, batchxbeam]
                std::vector<int> shape{hidden_size / new_tp_grain * new_buffer.mem_len, new_buffer.batchxbeam};
                int layer_offset = (layer_id % new_buffer.new_layers_per_device) * shape[0] * shape[1];
                storage_layer[layer_id]->do_recv_allocated_buff(stay_ids_b[layer_id], new_tp_grain, cache_buffers, &shape, layer_offset);
                storage_layer[layer_id]->do_recv_tp_buff(recv_op_json, new_tp_grain, tp_stage, tp_grain, p2p_ops_buff[layer_id]);
            }
        }
    }

    cache_mutex.unlock(); // unlock this lock, data sync controlled by following
    // =========================================================

    // step 1 : send buffers
    if(has_new_buffer_info || has_old_buffer_info){
        ftNcclGroupStart();
        for(int i = 0; i < layer_num; i++){
            for(auto const& op : p2p_ops_buff[i]){
                op.do_op(nccl_param);
            }
        }
        ftNcclGroupEnd();
        check_cuda_error(cudaDeviceSynchronize());

        for(int i = 0; i < layer_num; i++){
            for(int stay_id : stay_ids_b[i]){
                if(stay_id < 0){
                    storage_layer[i]->do_stay_dp_buff();
                }else/* if(!no_lock)*/{
                    int layer_offset = (i % old_buffer.new_layers_per_device) * hidden_size / new_tp_grain * old_buffer.mem_len * old_buffer.batchxbeam;
                    storage_layer[i]->do_stay_tp_buff(stay_id, new_tp_grain, cache_buffers, layer_offset);
                }
            }
            
            storage_layer[i]->switch_end_buff();
            if(has_new_buffer_info/* && !no_lock*/)
                storage_layer[i]->transpose_buff_for_comp_and_delete(transpose_buf); // remove tp buffer from layer
        }

        // post-processing is moved to last
    }else if(!ignore_buffer){
        // no buffer info, delete
        for(int i = 0; i < layer_num; i++){
            storage_layer[i]->switch_end_buff();
        }
    }
    FT_LOG_DEBUG(rank, "Buffers first Done.");

    if(no_lock){
        // if true, release lock when buffer is done, early release
        for(int i = 0; i < layer_num; ++i){
            storage_layer[i]->in_comm.unlock();
        }
    }

    if(!memory_efficient_switch){
        // step 2: dying rank first (overlap part)
        // if all p2p ops of a layer on this rank are concerned with dying rank, send params one by one for overlap,
        // until the first layer that has undying rank op, from which the overlap will be dominated by params afterwards, 
        //  just send them together here
        int first_undying, found = 0;
        for(first_undying = 0; first_undying < layer_num; ++first_undying){
            int current_layer = (first_undying == 0 ? pre_decoder_layer : first_undying - 1) + (first_undying > pre_decoder_layer);


            for(auto const& op : p2p_ops[current_layer]){
                if(!op.get_is_prior()) {found=1;break;} // encounter first undying send
            }
            if(found) break;
        }
        first_undying--;
        FT_LOG_DEBUG(rank, "first dying index: %d", first_undying);

        int i_sent_layer;
        // int first_undying_seq_num = first_undying == pre_decoder_layer ? 0 : (first_undying + (first_undying < pre_decoder_layer));
        for(i_sent_layer = 0; i_sent_layer <= first_undying /*&& i_sent_layer < first_undying_seq_num*/; ++i_sent_layer){
            int current_layer = (i_sent_layer == 0 ? pre_decoder_layer : i_sent_layer - 1) + (i_sent_layer > pre_decoder_layer);
            if(p2p_ops[current_layer].size() > 0) {
                FT_LOG_DEBUG(rank, "op for layer(dying) %d before", current_layer);

                ftNcclGroupStart();
                for(auto const& op : p2p_ops[current_layer]){
                    if(op.get_is_prior()) op.do_op(nccl_param);
                }
                ftNcclGroupEnd();

                FT_LOG_DEBUG(rank, "op for layer(dying) %d after", current_layer);
            }

            check_cuda_error(cudaDeviceSynchronize());

            for(int stay_id : stay_ids[current_layer]){
                if(stay_id < 0){
                    storage_layer[current_layer]->do_stay_dp_param();
                }else{
                    storage_layer[current_layer]->do_stay_tp_param(stay_id, new_tp_grain);
                }
            }

            storage_layer[current_layer]->switch_end(new_tp_grain);
            storage_layer[current_layer]->transpose_all_for_comp(transpose_buf); // will sync after transpose
            if(sleep_time < 0)
                storage_layer[current_layer]->in_comm.unlock();
        }

        // step 3: dying rank first (together part)
        for(; i_sent_layer < layer_num; ++i_sent_layer){
            int current_layer = (i_sent_layer == 0 ? pre_decoder_layer : i_sent_layer - 1) + (i_sent_layer > pre_decoder_layer);
            FT_LOG_DEBUG(rank, "op for layer(dying2) %d before", current_layer);
            ftNcclGroupStart();
            for(auto const& op : p2p_ops[current_layer]){
                if(op.get_is_prior()) op.do_op(nccl_param);
            }
            ftNcclGroupEnd();
            FT_LOG_DEBUG(rank, "op for layer(dying2) %d after", current_layer);
            check_cuda_error(cudaDeviceSynchronize());

        }
        FT_LOG_DEBUG(rank, "Dying rank first Done.");
    }else{
        FT_LOG_DEBUG(rank, "Enabled memory efficient switch, disabled dying rank first");
    }


    for(int i = 0; i < layer_num; i++){

        // put pre_decoder_layer into the first
        // 0->p  i->i-1 (i in 1~p)  i->i (i in p+1~n)
        int current_layer = (i == 0 ? pre_decoder_layer : i - 1) + (i > pre_decoder_layer);
        if(memory_efficient_switch) current_layer = transfer_order[i];

        if(memory_efficient_switch && recv_ops[current_layer].size() > 0){
            // alloc when recv, to save memory
            for(cJSON* recv_op_json : recv_ops[current_layer]){
                int lid = cJSON_GetObjectItemCaseSensitive(recv_op_json, "layer")->valueint;
                if(lid < 0){
                    int layer_id = -lid - 1;
                    FT_CHECK(layer_id == current_layer);
                    storage_layer[layer_id]->do_recv_dp_param(recv_op_json, p2p_ops[layer_id]);
                }else{
                    int layer_id, tp_stage, tp_grain;
                    layer_id_decode(lid, layer_id, tp_stage, tp_grain);
                    FT_CHECK(layer_id == current_layer);
                    storage_layer[layer_id]->do_recv_allocated(stay_ids[layer_id], new_tp_grain);
                    storage_layer[layer_id]->do_recv_tp_param(recv_op_json, new_tp_grain, tp_stage, tp_grain, p2p_ops[layer_id]);
                }
            }
        }

        // if(memory_efficient_switch){
        //     std::stable_sort(p2p_ops[current_layer].begin(), p2p_ops[current_layer].end(), [](const P2pOp<float>& a, const P2pOp<float>& b) -> bool {
        //         return a.get_peer() <b.get_peer();
        //     });
        // }

        int nsend = 0, nrecv = 0;
        if(p2p_ops[current_layer].size() > 0) {
            FT_LOG_DEBUG(rank, "op for layer %d before", current_layer);
            // std::vector<int> order_map(p2p_ops[current_layer].size());
            // for(int k = 0; i < p2p_ops[current_layer].size(); k++)
            //     order_map[k] = k;
            // std::stable_sort(order_map.begin(), order_map.end(), [&p2p_ops, current_layer](int a, int b) -> bool {
            //     return p2p_ops[current_layer][a].get_peer() < p2p_ops[current_layer][b].get_peer();
            // });

            ftNcclGroupStart();
            for(auto const& op : p2p_ops[current_layer]){
                if(!memory_efficient_switch && op.get_is_prior()) continue;
                // FT_LOG_DEBUG(rank, "%s", op.toString().c_str());
                op.do_op(nccl_param);
                if(op.op_type == P2pOp<float>::isend) nsend++;
                else nrecv++;
            }
            ftNcclGroupEnd();

            FT_LOG_DEBUG(rank, "op for layer %d after", current_layer);
        }
        // sync_check_cuda_error();
        check_cuda_error(cudaDeviceSynchronize());
        if(nsend + nrecv > 0){
            FT_LOG_DEBUG(rank, fmtstr("send: %d, recv %d", nsend, nrecv));
        }

        if(!storage_layer[current_layer]->get_is_transferring()) continue;

        for(int stay_id : stay_ids[current_layer]){
            if(stay_id < 0){
                storage_layer[current_layer]->do_stay_dp_param();
            }else{
                storage_layer[current_layer]->do_stay_tp_param(stay_id, new_tp_grain);
            }
        }
    
        storage_layer[current_layer]->switch_end(new_tp_grain);
        storage_layer[current_layer]->transpose_all_for_comp(transpose_buf); // will sync after transpose

        if(sleep_time < 0)
            storage_layer[current_layer]->in_comm.unlock();
    }

    if(!(sleep_time < 0)){
        if(sleep_time != 0){
            unsigned long slp = sleep_time * 1e6 + 1;
            FT_LOG_INFO(rank, "PC sleep %.6f sec before releasing lock", sleep_time);
            usleep(slp);
        }

        for(int i = 0; i < layer_num; i++)
            storage_layer[i]->in_comm.unlock();
    }
}
/*
void TensorStorage::do_op_nolock(const cJSON* entry){
    int pre_decoder_layer = layer_num - 2; // magic number here
    int new_tp_grain = cJSON_GetObjectItemCaseSensitive(entry, "tp")->valueint;
    FT_CHECK(new_tp_grain == storage_layer[0]->get_tp_degree());

    cache_mutex.lock();
    for(int i = 0; i < layer_num; ++i){
        storage_layer[i]->in_comm.lock();
    }

    cJSON* order_json =  cJSON_GetObjectItemCaseSensitive(entry, "order");
    bool memory_efficient_switch = cJSON_IsArray(order_json); // if on, disable dyingrank first
    std::vector<int> transfer_order;
    if(memory_efficient_switch){
        cJSON* ele;
        cJSON_ArrayForEach(ele, order_json){
            transfer_order.push_back(ele->valueint);
        }
        FT_CHECK(transfer_order.size() == layer_num);
    }

    cJSON* buffer_json = cJSON_GetObjectItemCaseSensitive(entry, "buffer");
    BufferConfig old_buffer, new_buffer;
    cJSON* buffer_old_json = cJSON_GetObjectItemCaseSensitive(buffer_json, "old");
    cJSON* buffer_new_json = cJSON_GetObjectItemCaseSensitive(buffer_json, "new");
    bool has_old_buffer_info = cJSON_IsObject(buffer_old_json);
    bool has_new_buffer_info = cJSON_IsObject(buffer_new_json);
    if(has_old_buffer_info) old_buffer.set_fields_by_json(buffer_old_json);
    if(has_new_buffer_info) new_buffer.set_fields_by_json(buffer_new_json);

    if(has_old_buffer_info){
        append_kvcache_to_layers(old_buffer.batchxbeam, old_buffer.mem_len);
    }

    for(int i = 0; i < layer_num; ++i){
        storage_layer[i]->switch_init();
    }

    // parse stay param json array
    std::vector<std::vector<int>> stay_ids(layer_num);
    cJSON* stay_ids_json = cJSON_GetObjectItemCaseSensitive(entry, "pstay");
    cJSON* stay_id_json;
    FT_LOG_DEBUG(rank, "parse_Pstay");
    if(cJSON_IsArray(stay_ids_json)){
        cJSON_ArrayForEach(stay_id_json, stay_ids_json){
            if(!cJSON_IsNumber(stay_id_json)) continue;
            int stay_id = stay_id_json->valueint;
            int layer_id, tp_stage, tp_grain;
            if(stay_id < 0)
                layer_id = -stay_id - 1;
            else
                layer_id_decode(stay_id, layer_id, tp_stage, tp_grain);
            stay_ids[layer_id].push_back(stay_id);
        }
    }
    // parse stay buff json array
    std::vector<std::vector<int>> stay_ids_b(layer_num);
    stay_ids_json = cJSON_GetObjectItemCaseSensitive(entry, "bstay");
    bool has_bstay = false;
    FT_LOG_DEBUG(rank, "parse_Bstay");
    if(cJSON_IsArray(stay_ids_json)){
        cJSON_ArrayForEach(stay_id_json, stay_ids_json){
            if(!cJSON_IsNumber(stay_id_json)) continue;
            int stay_id = stay_id_json->valueint;
            int layer_id, tp_stage, tp_grain;
            if(stay_id < 0)
                layer_id = -stay_id - 1;
            else
                layer_id_decode(stay_id, layer_id, tp_stage, tp_grain);
            has_bstay = true;
            stay_ids_b[layer_id].push_back(stay_id);
        }
    }
    if(!has_bstay) cache_buffers.clear();

    std::vector<std::vector<P2pOp<float>>> p2p_ops(layer_num);
    std::vector<std::vector<P2pOp<float>>> p2p_ops_buff(layer_num);

    // parse send param json array
    cJSON* send_ops_json = cJSON_GetObjectItemCaseSensitive(entry, "psend");
    cJSON* send_op_json;
    
    // parse send param json array
    std::vector<std::vector<cJSON*>> send_ops(layer_num);

    FT_LOG_DEBUG(rank, "parse_Psend");
    if(cJSON_IsArray(send_ops_json)){
        cJSON_ArrayForEach(send_op_json, send_ops_json){
            int lid = cJSON_GetObjectItemCaseSensitive(send_op_json, "layer")->valueint;
            if(lid < 0){
                int layer_id = -lid - 1;
                send_ops[layer_id].push_back(send_op_json);
            }else{
                int layer_id, tp_stage, tp_grain;
                layer_id_decode(lid, layer_id, tp_stage, tp_grain);
                send_ops[layer_id].push_back(send_op_json);
            }
        }
    }
    // parse send buff json array
    send_ops_json = cJSON_GetObjectItemCaseSensitive(entry, "bsend");
    FT_LOG_DEBUG(rank, "parse_Bsend");
    if(cJSON_IsArray(send_ops_json)){
        cJSON_ArrayForEach(send_op_json, send_ops_json){
            int lid = cJSON_GetObjectItemCaseSensitive(send_op_json, "layer")->valueint;
            if(lid < 0){
                int layer_id = -lid - 1;
                storage_layer[layer_id]->do_send_dp_buff(send_op_json, p2p_ops_buff[layer_id]);
            }else{
                int layer_id, tp_stage, tp_grain;
                layer_id_decode(lid, layer_id, tp_stage, tp_grain);
                storage_layer[layer_id]->do_send_tp_buff(send_op_json, tp_stage, tp_grain, p2p_ops_buff[layer_id]);
            }
        }
    }

    // parse recv buff json array
    cJSON* recv_ops_json = cJSON_GetObjectItemCaseSensitive(entry, "brecv");
    cJSON* recv_op_json;
    FT_LOG_DEBUG(rank, "parse_Brecv");
    if(cJSON_IsArray(recv_ops_json)){
        cJSON_ArrayForEach(recv_op_json, recv_ops_json){
            int lid = cJSON_GetObjectItemCaseSensitive(recv_op_json, "layer")->valueint;
            if(lid < 0){
                int layer_id = -lid - 1;
                FT_CHECK(layer_id == pre_decoder_layer);
                storage_layer[layer_id]->do_recv_dp_buff(recv_op_json, p2p_ops_buff[layer_id],
                    new_buffer.batchxbeam, new_buffer.session_len, hidden_size, new_buffer.mem_len, new_buffer.beam);
            }else{
                int layer_id, tp_stage, tp_grain;
                layer_id_decode(lid, layer_id, tp_stage, tp_grain);
                // shape in layer [hidden_size / tp * memory_len, batchxbeam]
                std::vector<int> shape{hidden_size / new_tp_grain * new_buffer.mem_len, new_buffer.batchxbeam};
                int layer_offset = (layer_id % new_buffer.new_layers_per_device) * shape[0] * shape[1];
                storage_layer[layer_id]->do_recv_allocated_buff(stay_ids_b[layer_id], new_tp_grain, cache_buffers, &shape, layer_offset);
                storage_layer[layer_id]->do_recv_tp_buff(recv_op_json, new_tp_grain, tp_stage, tp_grain, p2p_ops_buff[layer_id]);
            }
        }
    }
    
    cache_mutex.unlock(); // unlock this lock, data sync controlled by following
    // =========================================================

    // step 1 : send buffers
    if(has_new_buffer_info || has_old_buffer_info){
        ftNcclGroupStart();
        for(int i = 0; i < layer_num; i++){
            for(auto const& op : p2p_ops_buff[i]){
                op.do_op(nccl_param);
            }
        }
        ftNcclGroupEnd();
        check_cuda_error(cudaDeviceSynchronize());

        for(int i = 0; i < layer_num; i++){
            for(int stay_id : stay_ids_b[i]){
                if(stay_id < 0){
                    storage_layer[i]->do_stay_dp_buff();
                }else{
                    int layer_offset = (i % old_buffer.new_layers_per_device) * hidden_size / new_tp_grain * old_buffer.mem_len * old_buffer.batchxbeam;
                    storage_layer[i]->do_stay_tp_buff(stay_id, new_tp_grain, cache_buffers, layer_offset);
                }
            }
            storage_layer[i]->switch_end_buff();
        }

        // post-processing is moved to last
    }
    FT_LOG_DEBUG(rank, "Buffers first Done.");

    for(int i = 0; i < layer_num; ++i){
        storage_layer[i]->in_comm.unlock();
    }

    for(int i = 0; i < layer_num; i++){
        // put pre_decoder_layer into the first
        // 0->p  i->i-1 (i in 1~p)  i->i (i in p+1~n)
        int current_layer = (i == 0 ? pre_decoder_layer : i - 1) + (i > pre_decoder_layer);
        if(memory_efficient_switch) current_layer = transfer_order[i];

        if(memory_efficient_switch && send_ops[current_layer].size() > 0){
            // alloc when recv, to save memory
            
        }

        int nsend = 0, nrecv = 0;
        if(p2p_ops[current_layer].size() > 0) {
            FT_LOG_DEBUG(rank, "op for layer %d before", current_layer);
            // std::vector<int> order_map(p2p_ops[current_layer].size());
            // for(int k = 0; i < p2p_ops[current_layer].size(); k++)
            //     order_map[k] = k;
            // std::stable_sort(order_map.begin(), order_map.end(), [&p2p_ops, current_layer](int a, int b) -> bool {
            //     return p2p_ops[current_layer][a].get_peer() < p2p_ops[current_layer][b].get_peer();
            // });

            ftNcclGroupStart();
            for(auto const& op : p2p_ops[current_layer]){
                if(!memory_efficient_switch && op.get_is_prior()) continue;
                // FT_LOG_DEBUG(rank, "%s", op.toString().c_str());
                op.do_op(nccl_param);
                if(op.op_type == P2pOp<float>::isend) nsend++;
                else nrecv++;
            }
            ftNcclGroupEnd();

            FT_LOG_DEBUG(rank, "op for layer %d after", current_layer);
        }
        // sync_check_cuda_error();
        check_cuda_error(cudaDeviceSynchronize());
        if(nsend + nrecv > 0){
            FT_LOG_DEBUG(rank, fmtstr("send: %d, recv %d", nsend, nrecv));
        }

        if(!storage_layer[current_layer]->get_is_transferring()) continue;

        for(int stay_id : stay_ids[current_layer]){
            if(stay_id < 0){
                storage_layer[current_layer]->do_stay_dp_param();
            }else{
                storage_layer[current_layer]->do_stay_tp_param(stay_id, new_tp_grain);
            }
        }
    
        storage_layer[current_layer]->switch_end(new_tp_grain);

    }
}
*/

void TensorStorage::load_initial_weights(const std::string& model_dir, int pp_para_size, int dp_para_size, int tp_para_size){
    if(rank >= pp_para_size * dp_para_size * tp_para_size) return;
    int pp_stage = (rank/tp_para_size)%pp_para_size;
    int tp_stage = rank%tp_para_size;
    
    for(int i = 0; i < layer_num; i++){
        if(i == layer_num - 2){ // MAGIC NUMBER HERE
            storage_layer[i]->load_initial_weights(model_dir + "model");
        }else if(i == layer_num - 1){ // post-decoder layer
            if(pp_stage == pp_para_size - 1) // no tp
                storage_layer[i]->load_initial_weights(model_dir + "model");
        }else if(isValidLayer(pp_stage, pp_para_size, i, layer_num)){
            storage_layer[i]->load_initial_weights(model_dir + "model.layers." + std::to_string(i), true, tp_para_size, tp_stage);
        }
    }   
}

void TensorStorage::print_test_params(int verbose){
    printf("---------------------cache buffers-------------------\n");
    for(auto const& cache : cache_buffers){
        cache->print_data(verbose);
    }

    for(int i = 0; i < layer_num; i++){
        storage_layer[i]->print_params(verbose);
    }
}

TensorStorage::~TensorStorage(){
    for(int i = 0; i < layer_num; i++){
        delete storage_layer[i];
    }
    deviceFree(transpose_buf);
}

void TensorStorage::send_metadata(const TcpAgent& tcp_agent, int layer_id, int param_idx, int param_type){
    // layer_id = -1 -> post decoder params
    // layer_id = -2 -> other non-decoder params
    if(layer_id < 0) layer_id = layer_num + layer_id;
    storage_layer[layer_id]->send_metadata(tcp_agent, param_idx, param_type);
}

void TensorStorage::send_buffdata(const TcpAgent& tcp_agent, int layer_id, int param_idx, int param_type){
    if(param_type == 0){
        // KV Cache
        std::lock_guard<std::mutex> lock(cache_mutex);

        cache_buffers[param_idx]->send_metadata(tcp_agent);
        return;
    }
    // layer_id = -1 -> post decoder params
    // layer_id = -2 -> other non-decoder params
    if(layer_id < 0) layer_id = layer_num + layer_id;
    storage_layer[layer_id]->send_metadata(tcp_agent, param_idx, param_type + 2);
}

void TensorStorage::alloc_new_buffer_and_send(const TcpAgent& tcp_agent, int layer_id, int param_idx, int param_type, int sz){
    if(param_type == 0){
        // KV Cache
        std::lock_guard<std::mutex> lock(cache_mutex);

        float* tensor_devPtr;
        if(cache_buffers.size() > param_idx){
            if(cache_buffers[param_idx]->get_size() == sz){
                // same size: reuse
                cudaMemset((void*)(cache_buffers[param_idx]->get_tensor()), 0, sizeof(float) * sz);
                cache_buffers[param_idx]->send_metadata(tcp_agent, 1);
                return;
            }
        }
        deviceMalloc(&tensor_devPtr, sz, false);
        check_cuda_error(cudaMemset((void*)tensor_devPtr, 0, sizeof(float) * sz));
        std::vector<int> shape; shape.push_back(sz);
        auto buf_p = std::make_shared<TensorWrapper<float>>(tensor_devPtr, shape, cuda_stream, false, false, true);
        if(cache_buffers.size() > param_idx){
            // shared pointer, old tensor will be automatically free
            cache_buffers[param_idx] = buf_p;
        }else{
            FT_CHECK_WITH_INFO(cache_buffers.size() == param_idx,
                fmtstr("cache_buffers' size %ld should be equal to param_idx %d.", cache_buffers.size(), param_idx));
            cache_buffers.push_back(buf_p);
        }
        buf_p->send_metadata(tcp_agent);
        return;
    }

    if(layer_id < 0) layer_id = layer_num + layer_id;
    storage_layer[layer_id]->allocate_dp_buffer(tcp_agent, param_idx, {sz});
}