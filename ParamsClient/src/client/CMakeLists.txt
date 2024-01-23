add_executable(param_client main.cc layerConfig.cc TensorStorage.cc TensorStorageLayer.cc)
target_link_libraries(param_client PUBLIC  -lcublas -lcublasLt -lcudart
    cJSON mpi_utils nccl_utils memory_utils tcp_utils matrix_transpose_kernels)