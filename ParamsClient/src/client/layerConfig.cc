#include "src/client/layerConfig.h"
#define LAYER_NUM 12

ParamConfig_t test_config, gpt_megatron_345M;

ParamConfig_t* get_test_configs(){
    static bool initialized = false;
    if (initialized) return &test_config;

    int hidden_size = 8;

    for(int i = 0; i < LAYER_NUM; i++){
        test_config.push_back(std::vector<ParamConfig>());

        test_config[i].push_back(ParamConfig{{3 * hidden_size, hidden_size}, false, true});
        test_config[i].push_back(ParamConfig{{3 * hidden_size}, false, true});
        test_config[i].push_back(ParamConfig{{hidden_size, hidden_size}, true, true});
        test_config[i].push_back(ParamConfig{{hidden_size}, false, false});

        test_config[i].push_back(ParamConfig{{4 * hidden_size, hidden_size}, false, true});
        test_config[i].push_back(ParamConfig{{4 * hidden_size}, false, true});
        test_config[i].push_back(ParamConfig{{4 * hidden_size, hidden_size}, true, true});
        test_config[i].push_back(ParamConfig{{hidden_size}, false, false});

        test_config[i].push_back(ParamConfig{{hidden_size}, false, false});
        test_config[i].push_back(ParamConfig{{hidden_size}, false, false});
        test_config[i].push_back(ParamConfig{{hidden_size}, false, false});
        test_config[i].push_back(ParamConfig{{hidden_size}, false, false});
        
    }

    initialized = true;
    return &test_config;

}

ParamConfig_t* get_gpt_configs(int layer_num, int hidden_size, int max_seq_len, int vocab_size){
    static bool initialized = false;
    if (initialized) return &gpt_megatron_345M;

    for(int i = 0; i < layer_num; i++){
        gpt_megatron_345M.push_back(std::vector<ParamConfig>());
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size}, false, false, ".input_layernorm.bias.bin"});
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size}, false, false, ".input_layernorm.weight.bin"});
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size, hidden_size * 3}, true, true, ".attention.query_key_value.weight.0.bin"});
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size, 3}, true, true, ".attention.query_key_value.bias.0.bin"});
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size, hidden_size}, false, true, ".attention.dense.weight.0.bin"});
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size}, false, false, ".attention.dense.bias.bin"});
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size}, false, false, ".post_attention_layernorm.bias.bin"});
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size}, false, false, ".post_attention_layernorm.weight.bin"});
        
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size * 4, hidden_size}, true, true, ".mlp.dense_h_to_4h.weight.0.bin"});
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size * 4}, false, true, ".mlp.dense_h_to_4h.bias.0.bin"});
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size * 4, hidden_size}, false, true, ".mlp.dense_4h_to_h.weight.0.bin"});
        gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size}, false, false, ".mlp.dense_4h_to_h.bias.bin"});

        // gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size, 1}, true, true, "k_cache", ParamConfig::BufferType::CacheBuffer});
        // gpt_megatron_345M[i].push_back(ParamConfig{{hidden_size, 1}, true, true, "v_cache", ParamConfig::BufferType::CacheBuffer});

    }
    // -1 layer
    gpt_megatron_345M.push_back(std::vector<ParamConfig>());
    gpt_megatron_345M[layer_num].push_back(ParamConfig{{max_seq_len, hidden_size}, false, false, ".wpe.bin"});
    gpt_megatron_345M[layer_num].push_back(ParamConfig{{vocab_size * hidden_size}, false, false, ".wte.bin"});

    gpt_megatron_345M.push_back(std::vector<ParamConfig>());
    gpt_megatron_345M[layer_num + 1].push_back(ParamConfig{{hidden_size}, false, false, ".final_layernorm.bias.bin"});
    gpt_megatron_345M[layer_num + 1].push_back(ParamConfig{{hidden_size}, false, false, ".final_layernorm.weight.bin"});
    gpt_megatron_345M[layer_num + 1].push_back(ParamConfig{{vocab_size * hidden_size}, false, false, ".wte.bin"});

    initialized = true;
    return &gpt_megatron_345M;
}

void BufferConfig::set_fields_by_json(cJSON* buffer_json){
    if(!cJSON_IsObject(buffer_json)) return;
    cJSON* batchxbeam_node = cJSON_GetObjectItemCaseSensitive(buffer_json, "batchxbeam");
    cJSON* beam_node = cJSON_GetObjectItemCaseSensitive(buffer_json, "beam");
    cJSON* memlen_node = cJSON_GetObjectItemCaseSensitive(buffer_json, "mem_len");
    cJSON* session_node = cJSON_GetObjectItemCaseSensitive(buffer_json, "session_len");
    cJSON* new_layern_node = cJSON_GetObjectItemCaseSensitive(buffer_json, "new_layers_per_device");

    batchxbeam = batchxbeam_node->valueint;
    mem_len = memlen_node->valueint;
    session_len = session_node->valueint;
    new_layers_per_device = new_layern_node->valueint;
    beam = beam_node->valueint;
}