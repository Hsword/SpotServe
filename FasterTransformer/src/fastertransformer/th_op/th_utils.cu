/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/th_op/th_utils.h"

namespace torch_ext {

std::vector<size_t> convert_shape(torch::Tensor tensor)
{
    std::vector<size_t> v_shape;
    for (int i = 0; i < tensor.dim(); i++) {
        v_shape.push_back(tensor.size(i));
    }
    return v_shape;
}

template<typename T>
fastertransformer::Tensor convert_tensor(torch::Tensor tensor)
{
    return convert_tensor<T>(tensor, fastertransformer::MEMORY_GPU);
}

template fastertransformer::Tensor convert_tensor<float>(torch::Tensor tensor);
template fastertransformer::Tensor convert_tensor<half>(torch::Tensor tensor);
#ifdef ENABLE_BF16
template fastertransformer::Tensor convert_tensor<__nv_bfloat16>(torch::Tensor tensor);
#endif
template fastertransformer::Tensor convert_tensor<int>(torch::Tensor tensor);
template fastertransformer::Tensor convert_tensor<unsigned long long int>(torch::Tensor tensor);
template fastertransformer::Tensor convert_tensor<unsigned int>(torch::Tensor tensor);

template<typename T>
fastertransformer::Tensor convert_tensor(torch::Tensor tensor, fastertransformer::MemoryType memory_type)
{
    return fastertransformer::Tensor{
        memory_type, fastertransformer::getTensorType<T>(), convert_shape(tensor), get_ptr<T>(tensor)};
}

template fastertransformer::Tensor convert_tensor<float>(torch::Tensor                 tensor,
                                                         fastertransformer::MemoryType memory_type);
template fastertransformer::Tensor convert_tensor<half>(torch::Tensor                 tensor,
                                                        fastertransformer::MemoryType memory_type);
#ifdef ENABLE_BF16
template fastertransformer::Tensor convert_tensor<__nv_bfloat16>(torch::Tensor                 tensor,
                                                                 fastertransformer::MemoryType memory_type);
#endif
template fastertransformer::Tensor convert_tensor<int>(torch::Tensor tensor, fastertransformer::MemoryType memory_type);
template fastertransformer::Tensor convert_tensor<unsigned long long int>(torch::Tensor                 tensor,
                                                                          fastertransformer::MemoryType memory_type);
template fastertransformer::Tensor convert_tensor<unsigned int>(torch::Tensor                 tensor,
                                                                fastertransformer::MemoryType memory_type);

}  // namespace torch_ext
