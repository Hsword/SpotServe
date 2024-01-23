/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <string>
#include <vector>

namespace fastertransformer {

int read_start_ids(int               batch_size,
                   std::vector<int>* v_start_lengths,
                   std::vector<int>* v_start_ids,
                   int&              max_input_len,
                   const int         end_id,
                   const int         beam_width,
                   std::string       file_name);

int read_start_ids_iter(int               batch_size,
                        std::vector<std::vector<int>>* v_start_lengths_list,
                        std::vector<std::vector<int>>* v_start_ids_list,
                        std::vector<int>*              max_input_lens,
                        const int         end_id,
                        const int         beam_width,
                        std::string       file_name);

}  // namespace fastertransformer
