/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <vector>

#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/DecoderSelfAttentionLayer.h"
#include "src/fastertransformer/models/gpt/GptDecoderLayerWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"

namespace fastertransformer {

template<typename T>
class GptDecoder: public BaseLayer {
private:
protected:
    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);
    virtual void initialize();
    // buffer handling
    size_t max_batch_size_ = 0;
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t hidden_units_;

    T* decoder_normed_input_;
    T* self_attn_output_;
    T* normed_self_attn_output_;
    T* decoder_layer_output_;
    BaseAttentionLayer<T>* self_attention_layer_;
    FfnLayer<T>* ffn_layer_;

public:
    GptDecoder(size_t max_batch_size,
               size_t head_num,
               size_t size_per_head,
               size_t inter_size,
               size_t num_layer,
               cudaStream_t stream,
               cublasMMWrapper* cublas_wrapper,
               IAllocator* allocator,
               bool is_free_buffer_after_forward,
               bool sparse);

    GptDecoder(GptDecoder<T> const& decoder);

    virtual ~GptDecoder();

    virtual void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const std::vector<GptDecoderLayerWeight<T>>* decoder_layer_weights);
};

}  // namespace fastertransformer
