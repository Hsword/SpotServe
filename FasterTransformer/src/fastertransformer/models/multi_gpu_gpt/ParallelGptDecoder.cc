/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoder.h"

namespace fastertransformer {

template<typename T>
void ParallelGptDecoder<T>::initialize(bool tp_comm)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    self_attention_layer_ = new TensorParallelDecoderSelfAttentionLayer<T>(max_batch_size_,
                                                                           head_num_,
                                                                           size_per_head_,
                                                                           tensor_para_,
                                                                           stream_,
                                                                           cublas_wrapper_,
                                                                           allocator_,
                                                                           tp_comm,
                                                                           is_free_buffer_after_forward_,
                                                                           sparse_,
                                                                           int8_mode_,
                                                                           custom_all_reduce_comm_,
                                                                           enable_custom_all_reduce_);

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU;
    size_t max_inter_size     = has_adapters_ ? std::max(inter_size_, adapter_inter_size_) : inter_size_;
    if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::GeGLU) {
        ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
                                                       1,
                                                       head_num_,
                                                       size_per_head_,
                                                       max_inter_size,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       tp_comm,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       int8_mode_,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Relu || activation_type_ == ActivationType::ReGLU) {
        ffn_layer_ = new TensorParallelReluFfnLayer<T>(max_batch_size_,
                                                       1,
                                                       head_num_,
                                                       size_per_head_,
                                                       max_inter_size,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       tp_comm,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
}

template<typename T>
ParallelGptDecoder<T>::ParallelGptDecoder(size_t                              max_batch_size,
                                          size_t                              head_num,
                                          size_t                              size_per_head,
                                          size_t                              inter_size,
                                          size_t                              num_layer,
                                          float                               layernorm_eps,
                                          gptVariantParams                    gpt_variant_params,
                                          NcclParam                           tensor_para,
                                          NcclParam                           pipeline_para,
                                          cudaStream_t                        stream,
                                          cublasMMWrapper*                    cublas_wrapper,
                                          IAllocator*                         allocator,
                                          bool                                is_free_buffer_after_forward,
                                          bool                                sparse,
                                          int                                 int8_mode,
                                          std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                                          int                                 enable_custom_all_reduce,
                                          bool                                tp_comm,
                                          size_t                              real_layer_num):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    layernorm_eps_(layernorm_eps),
    layernorm_type_(gpt_variant_params.layernorm_type),
    activation_type_(gpt_variant_params.activation_type),
    adapter_inter_size_(gpt_variant_params.adapter_inter_size),
    has_adapters_(gpt_variant_params.has_adapters),
    hidden_units_(head_num_ * size_per_head_),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    int8_mode_(int8_mode),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    real_num_layer_ = real_layer_num == -1 ? num_layer_ : real_layer_num;
    initialize(tp_comm);
}

template<typename T>
ParallelGptDecoder<T>::ParallelGptDecoder(ParallelGptDecoder<T> const& decoder):
    BaseLayer(decoder.stream_,
              decoder.cublas_wrapper_,
              decoder.allocator_,
              decoder.is_free_buffer_after_forward_,
              decoder.cuda_device_prop_,
              decoder.sparse_),
    max_batch_size_(decoder.max_batch_size_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    layernorm_eps_(decoder.layernorm_eps_),
    layernorm_type_(decoder.layernorm_type_),
    activation_type_(decoder.activation_type_),
    adapter_inter_size_(decoder.adapter_inter_size_),
    has_adapters_(decoder.has_adapters_),
    hidden_units_(decoder.hidden_units_),
    tensor_para_(decoder.tensor_para_),
    pipeline_para_(decoder.pipeline_para_),
    int8_mode_(decoder.int8_mode_),
    custom_all_reduce_comm_(decoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(decoder.enable_custom_all_reduce_),
    real_num_layer_(decoder.real_num_layer_)
{
    initialize();
}

template<typename T>
void ParallelGptDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ParallelGptDecoder<T>::allocateBuffer(size_t batch_size)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    decoder_layer_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * hidden_units_, false));
    decoder_normed_input_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * batch_size * hidden_units_, false));
    self_attn_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(self_attn_output_, sizeof(T) * batch_size * hidden_units_, false));
    normed_self_attn_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(normed_self_attn_output_, sizeof(T) * batch_size * hidden_units_, false));
    // only allocate additionl buffers when has adapters
    after_adapter_attn_output_ = has_adapters_ ? reinterpret_cast<T*>(allocator_->reMalloc(
                                     after_adapter_attn_output_, sizeof(T) * batch_size * hidden_units_, false)) :
                                                 self_attn_output_;
    is_allocate_buffer_        = true;
}

template<typename T>
void ParallelGptDecoder<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&decoder_layer_output_));
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&normed_self_attn_output_));
        if (has_adapters_) {
            allocator_->free((void**)(&after_adapter_attn_output_));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool ParallelGptDecoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool ParallelGptDecoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool ParallelGptDecoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
bool ParallelGptDecoder<T>::isRealLayerId(uint l)
{
    if(num_layer_ == real_num_layer_) return true;
    int padding_interval = num_layer_ / (num_layer_ - real_num_layer_);
    return (l % padding_interval) != (padding_interval - 1);
}

template<typename T>
int ParallelGptDecoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
ParallelGptDecoder<T>::~ParallelGptDecoder()
{
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void ParallelGptDecoder<T>::forward(std::vector<Tensor>*                                  output_tensors,
                                    const std::vector<Tensor>*                            input_tensors,
                                    /*const*/ std::vector<ParallelGptDecoderLayerWeight<T>*>* gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [local_batch_size, hidden_dimension],
    //      finished [local_batch_size],
    //      input_lengths [local_batch_size],
    //      total_padding_tokens [local_batch_size]
    //      max_input_length [1] on cpu
    //      step [1] on cpu
    //      ite [1] on cpu
    //      cache_indirection [local_batch_size / beam_width, beam_width, memory_len]
    //              Here, local_batch_size contains the beam_width, so local_batch_size / beam_width
    //              is real local_batch_size.
    //      masked_tokens [local_batch_size, memory_len]

    // output tensors:
    //      decoder_output [local_batch_size, hidden_dimension],
    //      key_cache [num_layer, batch_size, head_num, size_per_head // x, memory_len, x]
    //      value_cache [num_layer, batch_size, head_num, memory_len, size_per_head]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 9);
    FT_CHECK(output_tensors->size() == 3);
    const size_t local_batch_size = input_tensors->at(0).shape[0];
    allocateBuffer(local_batch_size);

    const DataType data_type = getTensorType<T>();
    const int      ite       = *((int*)(input_tensors->at(6).data));

    std::vector<size_t> self_k_cache_size;
    self_k_cache_size.push_back(local_batch_size);
    for (auto t = output_tensors->at(1).shape.begin() + 2; t != output_tensors->at(1).shape.end(); ++t) {
        self_k_cache_size.push_back(*t);
    }
    std::vector<size_t> self_v_cache_size;
    self_v_cache_size.push_back(local_batch_size);
    for (auto t = output_tensors->at(2).shape.begin() + 2; t != output_tensors->at(2).shape.end(); ++t) {
        self_v_cache_size.push_back(*t);
    }

    size_t last_layer_id = isRealLayerId(num_layer_-1) ? num_layer_-1 : num_layer_-2;

    for (uint l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l) == false) {
            continue;
        }
        // sync_and_profile(fmtstr("decoder_%d start", l), 0);
        T* decoder_input  = (T*)((l == 0) ? input_tensors->at(0).data : decoder_layer_output_);
        T* decoder_output = (T*)((l == last_layer_id) ? output_tensors->at(0).data : decoder_layer_output_);

        if (isFirstLayerParallelId(l) == true && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
            // ftNcclRecv(decoder_input, local_batch_size * hidden_units_, pipeline_para_.rank_ - 1, pipeline_para_,
            // stream_);

            ftNcclRecv(decoder_input + local_batch_size * hidden_units_ / tensor_para_.world_size_ * tensor_para_.rank_,
                       local_batch_size * hidden_units_ / tensor_para_.world_size_,
                       pipeline_para_.rank_ - 1,
                       pipeline_para_,
                       stream_);
            if (tensor_para_.world_size_ > 1) {
                ftNcclAllGather(decoder_input,
                                decoder_input,
                                local_batch_size * hidden_units_ / tensor_para_.world_size_,
                                tensor_para_.rank_,
                                tensor_para_,
                                stream_);
            }
        }

        // start here
        if(isRealLayerId(l)){
            size_t cache_offset = l - getFirstLayerParallelId();
            for (auto t = output_tensors->at(1).shape.begin() + 1; t != output_tensors->at(1).shape.end(); ++t) {
                cache_offset *= *t;
            };
            size_t ite_cache_offset = ite * local_batch_size;
            for (auto t = output_tensors->at(1).shape.begin() + 2; t != output_tensors->at(1).shape.end(); ++t) {
                ite_cache_offset *= *t;
            }
            cache_offset += ite_cache_offset;

            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralLayerNorm(decoder_normed_input_,
                                    decoder_input,
                                    gpt_decoder_layer_weight->at(l)->get_pre_layernorm_weights().gamma,
                                    gpt_decoder_layer_weight->at(l)->get_pre_layernorm_weights().beta,
                                    layernorm_eps_,
                                    local_batch_size,
                                    hidden_units_,
                                    stream_);
            }
            sync_check_cuda_error();

            const int           max_prefix_prompt_length = 0;
            std::vector<Tensor> self_attention_input_tensors{
                Tensor{MEMORY_GPU,
                    data_type,
                    {local_batch_size, hidden_units_},
                    layernorm_type_ == LayerNormType::pre_layernorm ? decoder_normed_input_ : decoder_input},
                input_tensors->at(1),
                input_tensors->at(2),
                input_tensors->at(3),
                Tensor{
                    MEMORY_GPU, data_type, {(size_t)local_batch_size, (size_t)l}, nullptr},  // prefix prompt weight batch
                Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &max_prefix_prompt_length},      // max prefix prompt length
                input_tensors->at(4),
                input_tensors->at(5),
                input_tensors->at(7),
                input_tensors->at(8)};

            std::vector<Tensor> self_attention_output_tensors{
                Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, self_attn_output_},
                Tensor{MEMORY_GPU, data_type, self_k_cache_size, ((const T*)output_tensors->at(1).data) + cache_offset},
                Tensor{MEMORY_GPU, data_type, self_v_cache_size, ((const T*)output_tensors->at(2).data) + cache_offset}};

            // sync_and_profile(fmtstr("attention_%d_forward before", l), 0);
            self_attention_layer_->forward(&self_attention_output_tensors,
                                        &self_attention_input_tensors,
                                        &gpt_decoder_layer_weight->at(l)->get_self_attention_weights());

            // the adapter after attention
            if (has_adapters_) {
                invokeAddBias(self_attn_output_,
                            gpt_decoder_layer_weight->at(l)->get_self_attention_weights().attention_output_weight.bias,
                            local_batch_size,
                            hidden_units_,
                            stream_);

                std::vector<Tensor> ffn_input_tensors{
                    Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, self_attn_output_}};
                std::vector<Tensor> ffn_output_tensors{
                    Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, after_adapter_attn_output_}};

                ffn_layer_->resetInterSize(adapter_inter_size_ / tensor_para_.world_size_);
                ffn_layer_->forward(&ffn_output_tensors,
                                    &ffn_input_tensors,
                                    &gpt_decoder_layer_weight->at(l)->after_attention_adapter_weights);
            }

            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralAddBiasResidualPreLayerNorm(
                    after_adapter_attn_output_,
                    normed_self_attn_output_,
                    decoder_input,
                    has_adapters_ ? self_attn_output_ : nullptr,
                    gpt_decoder_layer_weight->at(l)->get_self_attn_layernorm_weights().gamma,
                    gpt_decoder_layer_weight->at(l)->get_self_attn_layernorm_weights().beta,
                    has_adapters_ ? gpt_decoder_layer_weight->at(l)->after_attention_adapter_weights.output_weight.bias :
                                    gpt_decoder_layer_weight->at(l)->get_self_attention_weights().attention_output_weight.bias,
                    layernorm_eps_,
                    local_batch_size,
                    hidden_units_,
                    stream_);
            }
            else if (layernorm_type_ == LayerNormType::post_layernorm) {
                invokeAddBiasResidualLayerNorm(
                    after_adapter_attn_output_,
                    decoder_input,
                    has_adapters_ ? gpt_decoder_layer_weight->at(l)->after_attention_adapter_weights.output_weight.bias :
                                    gpt_decoder_layer_weight->at(l)->get_self_attention_weights().attention_output_weight.bias,
                    gpt_decoder_layer_weight->at(l)->get_pre_layernorm_weights().gamma,
                    gpt_decoder_layer_weight->at(l)->get_pre_layernorm_weights().beta,
                    layernorm_eps_,
                    local_batch_size,
                    hidden_units_,
                    stream_);
            }

            sync_check_cuda_error();

            T* ffn_output_ptr = has_adapters_ ? self_attn_output_ : decoder_output;

            std::vector<Tensor> ffn_input_tensors{Tensor{
                MEMORY_GPU,
                data_type,
                {local_batch_size, hidden_units_},
                layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ : after_adapter_attn_output_}};
            std::vector<Tensor> ffn_output_tensors{
                Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, ffn_output_ptr}};

            ffn_layer_->resetInterSize(inter_size_ / tensor_para_.world_size_);
            // sync_and_profile(fmtstr("ffn_%d_forward before", l), 0);
            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l)->get_ffn_weights());

            // the adapter after ffn
            if (has_adapters_) {
                invokeAddBias(ffn_output_ptr,
                            gpt_decoder_layer_weight->at(l)->get_ffn_weights().output_weight.bias,
                            local_batch_size,
                            hidden_units_,
                            stream_);

                std::vector<Tensor> ffn_input_tensors{
                    Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, ffn_output_ptr}};
                std::vector<Tensor> ffn_output_tensors{
                    Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_output}};

                ffn_layer_->resetInterSize(adapter_inter_size_ / tensor_para_.world_size_);
                ffn_layer_->forward(
                    &ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l)->after_ffn_adapter_weights);
            }

            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeAddBiasResidual(decoder_output,
                                    after_adapter_attn_output_,
                                    has_adapters_ ? ffn_output_ptr : nullptr,
                                    has_adapters_ ?
                                        gpt_decoder_layer_weight->at(l)->after_ffn_adapter_weights.output_weight.bias :
                                        gpt_decoder_layer_weight->at(l)->get_ffn_weights().output_weight.bias,
                                    local_batch_size,
                                    hidden_units_,
                                    stream_);
            }
            else if (layernorm_type_ == LayerNormType::post_layernorm) {
                invokeAddBiasResidualLayerNorm(
                    decoder_output,
                    after_adapter_attn_output_,
                    has_adapters_ ? gpt_decoder_layer_weight->at(l)->after_ffn_adapter_weights.output_weight.bias :
                                    gpt_decoder_layer_weight->at(l)->get_ffn_weights().output_weight.bias,
                    gpt_decoder_layer_weight->at(l)->get_self_attn_layernorm_weights().gamma,
                    gpt_decoder_layer_weight->at(l)->get_self_attn_layernorm_weights().beta,
                    layernorm_eps_,
                    local_batch_size,
                    hidden_units_,
                    stream_);
            }
            sync_check_cuda_error();
        }else{
            FT_LOG_DEBUG("Decoder: Invalid PP layer %ld", l);
        }
        // end here

        if (isLastLayerParallelId(l) == true && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
            && pipeline_para_.world_size_ > 1) {

            ftNcclSend(decoder_output
                           + local_batch_size * hidden_units_ / tensor_para_.world_size_ * tensor_para_.rank_,
                       local_batch_size * hidden_units_ / tensor_para_.world_size_,
                       pipeline_para_.rank_ + 1,
                       pipeline_para_,
                       stream_);
        }
        // sync_and_profile(fmtstr("decoder_%d end", l), 0);
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class ParallelGptDecoder<float>;
template class ParallelGptDecoder<half>;
#ifdef ENABLE_BF16
template class ParallelGptDecoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer