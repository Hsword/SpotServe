#pragma once
#include<vector>
#include<string>

#include "3rdparty/cJSON.h"

constexpr int _ENCODE_BASE_ = 128;

struct ParamConfig{
    std::vector<int> shape;
    bool is_transpose;
    bool is_tp_param;
    std::string name;
};

struct BufferConfig{
    int batchxbeam=0;
    int beam=0;
    int mem_len=0;
    int session_len=0;
    int new_layers_per_device=0;

    void set_fields_by_json(cJSON* buffer_dict_json);
};

typedef std::vector<ParamConfig> LayerConfig_t;
typedef std::vector<LayerConfig_t> ParamConfig_t;

ParamConfig_t* get_test_configs();
ParamConfig_t* get_gpt_configs(int layer_num=32, int hidden_size=1024, int max_seq_len=1024, int vocab_size=50304);

inline void layer_id_decode(int layer_encode_id, int& layer_id, int& tp_stage, int& tp_grain){
    layer_id = layer_encode_id % _ENCODE_BASE_;
    int tmp = layer_encode_id / _ENCODE_BASE_;
    tp_stage = tmp % _ENCODE_BASE_;
    tp_grain = tmp / _ENCODE_BASE_ + 1;
}