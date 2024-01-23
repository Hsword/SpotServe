/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int        hidden_units,
                                                                const int        inter_size,
                                                                const int        tensor_para_size,
                                                                const int        tensor_para_rank,
                                                                const int        int8_mode,
                                                                gptVariantParams gpt_variant_params,
                                                                TcpAgent*        tcp_agent,
                                                                const int        layer_id):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    int8_mode_(int8_mode),
    gpt_variant_params_(gpt_variant_params),
    tcp_agent_(tcp_agent),
    layer_id_(layer_id)
{
    mallocWeights(layer_id);
    setWeightPtr();
    if (int8_mode_ != 0) {
        transposeCalibrateQuantizeWeight();
    }
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int int8_mode): int8_mode_(int8_mode)
{
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::~ParallelGptDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < weights_ptr.size(); i++) {
            if (weights_ptr[i] != nullptr) {
                if(tcp_agent_){
                    // FT_LOG_INFO("delete id: %d", i);
                    try{
                        releaseSharedPtr((void*)(weights_ptr[i] - offsets[i]));
                    }catch(const std::exception& e){
                        printf("At ParallelGptDecoderLayerWeight(%d)::weight_ptr[%d] release %p:\n", layer_id_, i, (void*)(weights_ptr[i] - offsets[i]));
                        puts(e.what());
                    }
                }
                else
                    deviceFree(weights_ptr[i]);
            }
        }

        // FT_LOG_INFO("after loop");

        pre_layernorm_weights.beta                            = nullptr;
        pre_layernorm_weights.gamma                           = nullptr;
        self_attention_weights.query_weight.kernel            = nullptr;
        self_attention_weights.query_weight.bias              = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias   = nullptr;
        self_attn_layernorm_weights.beta                      = nullptr;
        self_attn_layernorm_weights.gamma                     = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias   = nullptr;
        ffn_weights.output_weight.kernel       = nullptr;
        ffn_weights.output_weight.bias         = nullptr;

        after_attention_adapter_weights.intermediate_weight.kernel = nullptr;
        after_attention_adapter_weights.intermediate_weight.bias   = nullptr;
        after_attention_adapter_weights.output_weight.kernel       = nullptr;
        after_attention_adapter_weights.output_weight.bias         = nullptr;

        after_ffn_adapter_weights.intermediate_weight.kernel = nullptr;
        after_ffn_adapter_weights.intermediate_weight.bias   = nullptr;
        after_ffn_adapter_weights.output_weight.kernel       = nullptr;
        after_ffn_adapter_weights.output_weight.bias         = nullptr;

        if (int8_mode_ != 0) {
            for (int i = 0; i < int8_weights_ptr.size(); i++) {
                if (int8_weights_ptr[i] != nullptr) {
                    deviceFree(int8_weights_ptr[i]);
                }
            }
            for (int i = 0; i < scale_ptr.size(); i++) {
                if (scale_ptr[i] != nullptr) {
                    deviceFree(scale_ptr[i]);
                }
            }
            self_attention_weights.query_weight.int8_kernel                 = nullptr;
            self_attention_weights.query_weight.scale                       = nullptr;
            self_attention_weights.attention_output_weight.int8_kernel      = nullptr;
            self_attention_weights.attention_output_weight.scale            = nullptr;
            ffn_weights.intermediate_weight.int8_kernel                     = nullptr;
            ffn_weights.intermediate_weight.scale                           = nullptr;
            ffn_weights.output_weight.int8_kernel                           = nullptr;
            ffn_weights.output_weight.scale                                 = nullptr;
            after_attention_adapter_weights.intermediate_weight.int8_kernel = nullptr;
            after_attention_adapter_weights.intermediate_weight.scale       = nullptr;
            after_attention_adapter_weights.output_weight.int8_kernel       = nullptr;
            after_attention_adapter_weights.output_weight.scale             = nullptr;
            after_ffn_adapter_weights.intermediate_weight.int8_kernel       = nullptr;
            after_ffn_adapter_weights.intermediate_weight.scale             = nullptr;
            after_ffn_adapter_weights.output_weight.int8_kernel             = nullptr;
            after_ffn_adapter_weights.output_weight.scale                   = nullptr;
        }

        // FT_LOG_INFO("after int8 free");
        is_maintain_buffer = false;
    }
}

template<typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const ParallelGptDecoderLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    int8_mode_(other.int8_mode_),
    layer_id_(other.layer_id_),
    gpt_variant_params_(other.gpt_variant_params_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);

    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    if (gpt_variant_params_.has_adapters) {
        cudaD2Dcpy(weights_ptr[12],
                   other.weights_ptr[12],
                   hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[14],
                   other.weights_ptr[14],
                   gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], hidden_units_);
        cudaD2Dcpy(weights_ptr[16],
                   other.weights_ptr[16],
                   hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[17], other.weights_ptr[17], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[18],
                   other.weights_ptr[18],
                   gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[19], other.weights_ptr[19], hidden_units_);
    }

    if (int8_mode_ != 0) {
        cudaD2Dcpy(
            int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(scale_ptr[0], other.scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[1], other.scale_ptr[1], hidden_units_);
        cudaD2Dcpy(scale_ptr[2], other.scale_ptr[2], inter_size_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[3], other.scale_ptr[3], hidden_units_);
        if (gpt_variant_params_.has_adapters) {
            cudaD2Dcpy(int8_weights_ptr[4],
                       other.int8_weights_ptr[4],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[5],
                       other.int8_weights_ptr[5],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(int8_weights_ptr[6],
                       other.int8_weights_ptr[6],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[7],
                       other.int8_weights_ptr[7],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(scale_ptr[4], other.scale_ptr[4], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(scale_ptr[5], other.scale_ptr[5], hidden_units_);
            cudaD2Dcpy(scale_ptr[6], other.scale_ptr[6], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(scale_ptr[7], other.scale_ptr[7], hidden_units_);
        }
    }

    setWeightPtr();
}

template<typename T>
ParallelGptDecoderLayerWeight<T>&
ParallelGptDecoderLayerWeight<T>::operator=(const ParallelGptDecoderLayerWeight& other)
{
    hidden_units_       = other.hidden_units_;
    inter_size_         = other.inter_size_;
    tensor_para_size_   = other.tensor_para_size_;
    tensor_para_rank_   = other.tensor_para_rank_;
    int8_mode_          = other.int8_mode_;
    gpt_variant_params_ = other.gpt_variant_params_;
    layer_id_           = other.layer_id_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);

    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    if (gpt_variant_params_.has_adapters) {
        cudaD2Dcpy(weights_ptr[12],
                   other.weights_ptr[12],
                   hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[14],
                   other.weights_ptr[14],
                   gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], hidden_units_);
        cudaD2Dcpy(weights_ptr[16],
                   other.weights_ptr[16],
                   hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[17], other.weights_ptr[17], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[18],
                   other.weights_ptr[18],
                   gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[19], other.weights_ptr[19], hidden_units_);
    }

    if (int8_mode_ != 0) {
        cudaD2Dcpy(
            int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(scale_ptr[0], other.scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[1], other.scale_ptr[1], hidden_units_);
        cudaD2Dcpy(scale_ptr[2], other.scale_ptr[2], inter_size_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[3], other.scale_ptr[3], hidden_units_);
        if (gpt_variant_params_.has_adapters) {
            cudaD2Dcpy(int8_weights_ptr[4],
                       other.int8_weights_ptr[4],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[5],
                       other.int8_weights_ptr[5],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(int8_weights_ptr[6],
                       other.int8_weights_ptr[6],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[7],
                       other.int8_weights_ptr[7],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(scale_ptr[4], other.scale_ptr[4], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(scale_ptr[5], other.scale_ptr[5], hidden_units_);
            cudaD2Dcpy(scale_ptr[6], other.scale_ptr[6], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(scale_ptr[7], other.scale_ptr[7], hidden_units_);
        }
    }

    setWeightPtr();
    return *this;
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    if(!tcp_agent_){
        FT_CHECK(is_maintain_buffer == true);

        loadWeightFromBin<T>(weights_ptr[0], {hidden_units_}, dir_path + ".input_layernorm.bias.bin", model_file_type);
        loadWeightFromBin<T>(weights_ptr[1], {hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);
        loadWeightFromBin<T>(weights_ptr[2],
                            {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
                            dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_) + ".bin",
                            model_file_type);
        loadWeightFromBin<T>(weights_ptr[3],
                            {3, hidden_units_ / tensor_para_size_},
                            dir_path + ".attention.query_key_value.bias." + std::to_string(tensor_para_rank_) + ".bin",
                            model_file_type);
        loadWeightFromBin<T>(weights_ptr[4],
                            {hidden_units_ / tensor_para_size_, hidden_units_},
                            dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_) + ".bin",
                            model_file_type);
        loadWeightFromBin<T>(weights_ptr[5], {hidden_units_}, dir_path + ".attention.dense.bias.bin", model_file_type);
        loadWeightFromBin<T>(
            weights_ptr[6], {hidden_units_}, dir_path + ".post_attention_layernorm.bias.bin", model_file_type);
        loadWeightFromBin<T>(
            weights_ptr[7], {hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);

        loadWeightFromBin<T>(weights_ptr[8],
                            {hidden_units_, inter_size_ / tensor_para_size_},
                            dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                            model_file_type);
        loadWeightFromBin<T>(weights_ptr[9],
                            {inter_size_ / tensor_para_size_},
                            dir_path + ".mlp.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_) + ".bin",
                            model_file_type);
        loadWeightFromBin<T>(weights_ptr[10],
                            {inter_size_ / tensor_para_size_, hidden_units_},
                            dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                            model_file_type);
        loadWeightFromBin<T>(weights_ptr[11], {hidden_units_}, dir_path + ".mlp.dense_4h_to_h.bias.bin", model_file_type);

        if (gpt_variant_params_.has_adapters) {
            loadWeightFromBin<T>(weights_ptr[12],
                                {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                                dir_path + ".after_attention_adapter.dense_h_to_4h.weight."
                                    + std::to_string(tensor_para_rank_) + ".bin",
                                model_file_type);
            loadWeightFromBin<T>(weights_ptr[13],
                                {gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                                dir_path + ".after_attention_adapter.dense_h_to_4h.bias."
                                    + std::to_string(tensor_para_rank_) + ".bin",
                                model_file_type);
            loadWeightFromBin<T>(weights_ptr[14],
                                {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                                dir_path + ".after_attention_adapter.dense_4h_to_h.weight."
                                    + std::to_string(tensor_para_rank_) + ".bin",
                                model_file_type);
            loadWeightFromBin<T>(weights_ptr[15],
                                {hidden_units_},
                                dir_path + ".after_attention_adapter.dense_4h_to_h.bias.bin",
                                model_file_type);
            loadWeightFromBin<T>(weights_ptr[16],
                                {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                                dir_path + ".after_ffn_adapter.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_)
                                    + ".bin",
                                model_file_type);
            loadWeightFromBin<T>(weights_ptr[17],
                                {gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                                dir_path + ".after_ffn_adapter.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_)
                                    + ".bin",
                                model_file_type);
            loadWeightFromBin<T>(weights_ptr[18],
                                {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                                dir_path + ".after_ffn_adapter.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_)
                                    + ".bin",
                                model_file_type);
            loadWeightFromBin<T>(
                weights_ptr[19], {hidden_units_}, dir_path + ".after_ffn_adapter.dense_4h_to_h.bias.bin", model_file_type);
        }
    }
    
    if (int8_mode_ != 0) {
        transposeCalibrateQuantizeWeight();
    }
}

template<typename T>
LayerNormWeight<T>& ParallelGptDecoderLayerWeight<T>::get_pre_layernorm_weights() {
    if(!pre_layernorm_weights.beta){
        offsets[0] = requestTensor(&weights_ptr[0], hidden_units_, layer_id_, 0, 1, tcp_agent_);
        pre_layernorm_weights.beta = weights_ptr[0];
    }
    if(!pre_layernorm_weights.gamma){
        offsets[1] = requestTensor(&weights_ptr[1], hidden_units_, layer_id_, 1, 1, tcp_agent_);
        pre_layernorm_weights.gamma = weights_ptr[1];
    }
    return pre_layernorm_weights;
}

template<typename T>
AttentionWeight<T>& ParallelGptDecoderLayerWeight<T>::get_self_attention_weights() {
    if(!self_attention_weights.query_weight.kernel){
        offsets[2] = requestTensor(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_, layer_id_, 0, 0, tcp_agent_);
        self_attention_weights.query_weight.kernel = weights_ptr[2];
    }
    if(!self_attention_weights.query_weight.bias){
        offsets[3] = requestTensor(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_, layer_id_, 1, 0, tcp_agent_);
        self_attention_weights.query_weight.bias = weights_ptr[3];
    }
    if(!self_attention_weights.attention_output_weight.kernel){
        offsets[4] = requestTensor(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_, layer_id_, 2, 0, tcp_agent_);
        self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
    }
    if(!self_attention_weights.attention_output_weight.bias){
        offsets[5] = requestTensor(&weights_ptr[5], hidden_units_, layer_id_, 2, 1, tcp_agent_);
        self_attention_weights.attention_output_weight.bias = weights_ptr[5];
    }
    return self_attention_weights;
}

template<typename T>
LayerNormWeight<T>& ParallelGptDecoderLayerWeight<T>::get_self_attn_layernorm_weights() {
    if(!self_attn_layernorm_weights.beta){
        offsets[6] = requestTensor(&weights_ptr[6], hidden_units_, layer_id_, 3, 1, tcp_agent_);
        self_attn_layernorm_weights.beta = weights_ptr[6];
    }
    if(!self_attn_layernorm_weights.gamma){
        offsets[7] = requestTensor(&weights_ptr[7], hidden_units_, layer_id_, 4, 1, tcp_agent_);
        self_attn_layernorm_weights.gamma = weights_ptr[7];
    }
    return self_attn_layernorm_weights;
}

template<typename T>
FfnWeight<T>& ParallelGptDecoderLayerWeight<T>::get_ffn_weights() {
    if(!ffn_weights.intermediate_weight.kernel){
        offsets[8] = requestTensor(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_, layer_id_, 3, 0, tcp_agent_);
        ffn_weights.intermediate_weight.kernel = weights_ptr[8];
    }
    if(!ffn_weights.intermediate_weight.bias){
        offsets[9] = requestTensor(&weights_ptr[9], inter_size_ / tensor_para_size_, layer_id_, 4, 0, tcp_agent_);
        ffn_weights.intermediate_weight.bias = weights_ptr[9];
    }
    if(!ffn_weights.output_weight.kernel){
        offsets[10] = requestTensor(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_, layer_id_, 5, 0, tcp_agent_);
        ffn_weights.output_weight.kernel = weights_ptr[10];
    }
    if(!ffn_weights.output_weight.bias){
        offsets[11] = requestTensor(&weights_ptr[11], hidden_units_, layer_id_, 5, 1, tcp_agent_);
        ffn_weights.output_weight.bias = weights_ptr[11];
    }
    return ffn_weights;
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::setWeightPtr()
{
    pre_layernorm_weights.beta                            = weights_ptr[0];
    pre_layernorm_weights.gamma                           = weights_ptr[1];
    self_attention_weights.query_weight.kernel            = weights_ptr[2];
    self_attention_weights.query_weight.bias              = weights_ptr[3];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
    self_attention_weights.attention_output_weight.bias   = weights_ptr[5];
    self_attn_layernorm_weights.beta                      = weights_ptr[6];
    self_attn_layernorm_weights.gamma                     = weights_ptr[7];

    ffn_weights.intermediate_weight.kernel = weights_ptr[8];
    ffn_weights.intermediate_weight.bias   = weights_ptr[9];
    ffn_weights.output_weight.kernel       = weights_ptr[10];
    ffn_weights.output_weight.bias         = weights_ptr[11];

    after_attention_adapter_weights.intermediate_weight.kernel = weights_ptr[12];
    after_attention_adapter_weights.intermediate_weight.bias   = weights_ptr[13];
    after_attention_adapter_weights.output_weight.kernel       = weights_ptr[14];
    after_attention_adapter_weights.output_weight.bias         = weights_ptr[15];

    after_ffn_adapter_weights.intermediate_weight.kernel = weights_ptr[16];
    after_ffn_adapter_weights.intermediate_weight.bias   = weights_ptr[17];
    after_ffn_adapter_weights.output_weight.kernel       = weights_ptr[18];
    after_ffn_adapter_weights.output_weight.bias         = weights_ptr[19];

    if (int8_mode_ != 0) {
        self_attention_weights.query_weight.int8_kernel                 = int8_weights_ptr[0];
        self_attention_weights.query_weight.scale                       = scale_ptr[0];
        self_attention_weights.attention_output_weight.int8_kernel      = int8_weights_ptr[1];
        self_attention_weights.attention_output_weight.scale            = scale_ptr[1];
        ffn_weights.intermediate_weight.int8_kernel                     = int8_weights_ptr[2];
        ffn_weights.intermediate_weight.scale                           = scale_ptr[2];
        ffn_weights.output_weight.int8_kernel                           = int8_weights_ptr[3];
        ffn_weights.output_weight.scale                                 = scale_ptr[3];
        after_attention_adapter_weights.intermediate_weight.int8_kernel = int8_weights_ptr[4];
        after_attention_adapter_weights.intermediate_weight.scale       = scale_ptr[4];
        after_attention_adapter_weights.output_weight.int8_kernel       = int8_weights_ptr[5];
        after_attention_adapter_weights.output_weight.scale             = scale_ptr[5];
        after_ffn_adapter_weights.intermediate_weight.int8_kernel       = int8_weights_ptr[6];
        after_ffn_adapter_weights.intermediate_weight.scale             = scale_ptr[6];
        after_ffn_adapter_weights.output_weight.int8_kernel             = int8_weights_ptr[7];
        after_ffn_adapter_weights.output_weight.scale                   = scale_ptr[7];
    }

    is_maintain_buffer = true;
}

template<typename T>
void ParallelGptDecoderLayerWeight<T>::mallocWeights(int layer_id)
{
    if(tcp_agent_){
        /*
        FT_CHECK_WITH_INFO(layer_id >= 0, "invalid layer_id");
        offsets[0] = requestTensor(&weights_ptr[0], hidden_units_, layer_id, 0, 1, tcp_agent_);
        offsets[1] = requestTensor(&weights_ptr[1], hidden_units_, layer_id, 1, 1, tcp_agent_);
        offsets[2] = requestTensor(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_, layer_id, 0, 0, tcp_agent_);
        offsets[3] = requestTensor(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_, layer_id, 1, 0, tcp_agent_);
        offsets[4] = requestTensor(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_, layer_id, 2, 0, tcp_agent_);
        offsets[5] = requestTensor(&weights_ptr[5], hidden_units_, layer_id, 2, 1, tcp_agent_);
        offsets[6] = requestTensor(&weights_ptr[6], hidden_units_, layer_id, 3, 1, tcp_agent_);
        offsets[7] = requestTensor(&weights_ptr[7], hidden_units_, layer_id, 4, 1, tcp_agent_);

        offsets[8] = requestTensor(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_, layer_id, 3, 0, tcp_agent_);
        offsets[9] = requestTensor(&weights_ptr[9], inter_size_ / tensor_para_size_, layer_id, 4, 0, tcp_agent_);
        offsets[10] = requestTensor(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_, layer_id, 5, 0, tcp_agent_);
        offsets[11] = requestTensor(&weights_ptr[11], hidden_units_, layer_id, 5, 1, tcp_agent_);
        */

    }else{
        deviceMalloc(&weights_ptr[0], hidden_units_);
        deviceMalloc(&weights_ptr[1], hidden_units_);
        deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[5], hidden_units_);
        deviceMalloc(&weights_ptr[6], hidden_units_);
        deviceMalloc(&weights_ptr[7], hidden_units_);

        deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
        deviceMalloc(&weights_ptr[9], inter_size_ / tensor_para_size_);
        deviceMalloc(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[11], hidden_units_);
    }

    if (gpt_variant_params_.has_adapters) {
        FT_CHECK_WITH_INFO(tcp_agent_ == NULL, "prompt learning with adapters has not supported yet.");
        deviceMalloc(&weights_ptr[12], hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[13], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[14], gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[15], hidden_units_);
        deviceMalloc(&weights_ptr[16], hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[17], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[18], gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[19], hidden_units_);
    }

    if (int8_mode_ != 0) {
        FT_CHECK_WITH_INFO(tcp_agent_ == NULL, "prompt learning with int8_mode has not supported yet.");
        deviceMalloc(&int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        deviceMalloc(&int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);

        deviceMalloc(&scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&scale_ptr[1], hidden_units_);
        deviceMalloc(&scale_ptr[2], inter_size_ / tensor_para_size_);
        deviceMalloc(&scale_ptr[3], hidden_units_);

        if (gpt_variant_params_.has_adapters) {
            deviceMalloc(&int8_weights_ptr[4],
                         hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&int8_weights_ptr[5],
                         gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            deviceMalloc(&int8_weights_ptr[6],
                         hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&int8_weights_ptr[7],
                         gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            deviceMalloc(&scale_ptr[4], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&scale_ptr[5], hidden_units_);
            deviceMalloc(&scale_ptr[6], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&scale_ptr[7], hidden_units_);
        }
    }
}

#ifdef SPARSITY_ENABLED
template<typename T>
void ParallelGptDecoderLayerWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim)
{
    hidden_units_ = hidden_dim;
    inter_size_   = 4 * hidden_units_;

    const size_t num_sparse_weights            = 8;
    size_t       shapes[num_sparse_weights][2] = {
              {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
              {hidden_units_ / tensor_para_size_, hidden_units_},
              {hidden_units_, inter_size_ / tensor_para_size_},
              {inter_size_ / tensor_para_size_, hidden_units_},
              {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
              {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
              {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
              {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_}};

    const T* dense_weights[num_sparse_weights] = {self_attention_weights.query_weight.kernel,
                                                  self_attention_weights.attention_output_weight.kernel,
                                                  ffn_weights.intermediate_weight.kernel,
                                                  ffn_weights.output_weight.kernel,
                                                  after_attention_adapter_weights.intermediate_weight.kernel,
                                                  after_attention_adapter_weights.output_weight.kernel,
                                                  after_ffn_adapter_weights.intermediate_weight.kernel,
                                                  after_ffn_adapter_weights.output_weight.kernel};

    size_t real_num_sparse_weights = gpt_variant_params_.has_adapters ? num_sparse_weights : (num_sparse_weights - 4);
    for (size_t i = 0; i < real_num_sparse_weights; ++i) {
        int    m               = shapes[i][1];
        int    k               = shapes[i][0];
        size_t compressed_size = cublas_wrapper.getSparseMatrixSize(m, k);
        deviceMalloc(&sp_weights_ptr[i], static_cast<int>(compressed_size), false);
        cublas_wrapper.compressMatrix(dense_weights[i], sp_weights_ptr[i], m, k);
    }

    self_attention_weights.query_weight.sp_kernel                 = sp_weights_ptr[0];
    self_attention_weights.attention_output_weight.sp_kernel      = sp_weights_ptr[1];
    ffn_weights.intermediate_weight.sp_kernel                     = sp_weights_ptr[2];
    ffn_weights.output_weight.sp_kernel                           = sp_weights_ptr[3];
    after_attention_adapter_weights.intermediate_weight.sp_kernel = sp_weights_ptr[4];
    after_attention_adapter_weights.output_weight.sp_kernel       = sp_weights_ptr[5];
    after_ffn_adapter_weights.intermediate_weight.sp_kernel       = sp_weights_ptr[6];
    after_ffn_adapter_weights.output_weight.sp_kernel             = sp_weights_ptr[7];
    is_maintain_sp_buffer                                         = true;
}
#endif

template<typename T>
void ParallelGptDecoderLayerWeight<T>::transposeCalibrateQuantizeWeight()
{
    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[0], weights_ptr[2], hidden_units_, 3 * hidden_units_ / tensor_para_size_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[0],
                                               scale_ptr[0],
                                               weights_ptr[2],
                                               hidden_units_,
                                               3 * hidden_units_ / tensor_para_size_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[1], weights_ptr[4], hidden_units_ / tensor_para_size_, hidden_units_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[1], scale_ptr[1], weights_ptr[4], hidden_units_ / tensor_para_size_, hidden_units_, stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[2], weights_ptr[8], hidden_units_, inter_size_ / tensor_para_size_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[2], scale_ptr[2], weights_ptr[8], hidden_units_, inter_size_ / tensor_para_size_, stream_);

    invokeLdnCalibrateWeightPerChannel(
        scale_ptr[3], weights_ptr[10], inter_size_ / tensor_para_size_, hidden_units_, stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(
        int8_weights_ptr[3], scale_ptr[3], weights_ptr[10], inter_size_ / tensor_para_size_, hidden_units_, stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[4],
                                       weights_ptr[12],
                                       hidden_units_,
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[4],
                                               scale_ptr[4],
                                               weights_ptr[12],
                                               hidden_units_,
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[5],
                                       weights_ptr[14],
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       hidden_units_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[5],
                                               scale_ptr[5],
                                               weights_ptr[14],
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               hidden_units_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[6],
                                       weights_ptr[16],
                                       hidden_units_,
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[6],
                                               scale_ptr[6],
                                               weights_ptr[16],
                                               hidden_units_,
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               stream_);

    invokeLdnCalibrateWeightPerChannel(scale_ptr[7],
                                       weights_ptr[18],
                                       gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                       hidden_units_,
                                       stream_);
    invokeLdnTransposeQuantizeWeightPerChannel(int8_weights_ptr[7],
                                               scale_ptr[7],
                                               weights_ptr[18],
                                               gpt_variant_params_.adapter_inter_size / tensor_para_size_,
                                               hidden_units_,
                                               stream_);
}

template struct ParallelGptDecoderLayerWeight<float>;
template struct ParallelGptDecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct ParallelGptDecoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
