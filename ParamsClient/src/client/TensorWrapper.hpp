#pragma once
#include <vector>

#include "src/client/p2pOp.hpp"
#include "src/utils/tcp_utils.h"
#include "src/utils/memory_utils.h"
#include "src/kernels/matrix_transpose_kernels.h"

using namespace fastertransformer;

template<class T>
class TensorWrapper{
private:
    bool is_trans;
    bool now_trans; // at communication time, now_trans=true

    cudaStream_t cuda_stream;

    T* devPtr;

    int offset = 0; // should always be zero after reconstruction
    int st_size;

    std::vector<int> shape; // shape is for communication time

    // is valid only if is_trans==false || is_modified==false
    TensorMetaTransit_t tensor_metadata;
    void update_metadata(bool get_handle=true, T* ptr = nullptr);

// -------------------------------------
    bool is_buffer;
    bool free_when_destory;

public:
    TensorWrapper(T* _devPtr, const std::vector<int>& _shape, cudaStream_t _cuda_stream, 
        bool _is_trans=false, bool _now_trans=true, bool _is_buffer=false, bool _free_when_destory=true);
    TensorWrapper(const TensorWrapper&) = delete;
    TensorWrapper& operator=(const TensorWrapper&) = delete;

    void send_metadata(const TcpAgent& tcp_agent, int flag = 0);
    T* get_sliced_tensor(int tp_stage, int tp_grain, int current_grain, int& tensor_size);
    void assign_sliced_tensor(int tp_stage, int tp_grain, int current_grain, T* tensor_devPtr, int tensor_size);
    void slice_tensor_and_delete(int tp_stage, int tp_grain, int current_grain, T* del_dst_ptr=nullptr);

    void transpose_inplace(T* buf, bool for_compute);
    void transpose_to(T* dst);

    ~TensorWrapper();

    void print_data(int verbose);
    void assign_test_value(T base_value);
    void load_init_value(const std::string& weight_path, bool print_debug_info = false);

    bool get_is_buffer() const {return is_buffer; }
    bool get_is_trans() const {return is_trans; }
    int get_size() const {return st_size; }
    const std::vector<int>& get_shape() const {return shape; }
    T* get_tensor() const { return devPtr ? devPtr + offset : devPtr; }

    void allocate_mem();
};

template<class T>
TensorWrapper<T>::TensorWrapper(T* _devPtr, const std::vector<int>& _shape, cudaStream_t _cuda_stream,
    bool _is_trans, bool _now_trans, bool _is_buffer, bool _free_when_destory)
    : devPtr(_devPtr)
    , shape(_shape)
    , is_trans(_is_trans)
    , cuda_stream(_cuda_stream)
    , now_trans(_now_trans)
    , is_buffer(_is_buffer)
    , free_when_destory(_free_when_destory)
{
    st_size = 1;
    for(int l : shape)
        st_size *= l;
    
    update_metadata(devPtr && free_when_destory);
}

// only call when dePtr is null (for lazy alloc)
template<class T>
void TensorWrapper<T>::allocate_mem(){
    FT_CHECK(devPtr == nullptr);
    deviceMalloc(&devPtr, st_size, false);
    update_metadata(free_when_destory);
}

template<class T>
void TensorWrapper<T>::send_metadata(const TcpAgent& tcp_agent, int flag){
    FT_CHECK(free_when_destory == true);
    // FT_LOG_DEBUG("send ptr: %p", devPtr);
    
    tcp_agent.tcpSend((void*)(&tensor_metadata), sizeof(tensor_metadata));
    tensor_metadata.flag = 1;
}

template<class T>
void TensorWrapper<T>::update_metadata(bool get_handle, T* ptr){
    tensor_metadata.stoage_size = st_size * sizeof(T);
    tensor_metadata.storage_offset = offset * sizeof(T);

    if(get_handle){
        if(ptr && ptr != devPtr){
            // check_cuda_error(cudaIpcCloseMemHandle(devPtr));
            deviceFree(devPtr);
            devPtr = ptr;
        }
        check_cuda_error(cudaIpcGetMemHandle(&tensor_metadata.recv_tensor_handle, (void*)devPtr));
        tensor_metadata.flag = 0;
    }
}

template<class T>
T* TensorWrapper<T>::get_sliced_tensor(int tp_stage, int tp_grain, int current_grain, int& tensor_size){
    int factor = tp_grain / current_grain;
    int stage = tp_stage % factor;
    tensor_size = st_size / factor;

    return (devPtr + offset) + (tensor_size * stage);
}

template<class T>
void TensorWrapper<T>::assign_sliced_tensor(int tp_stage, int tp_grain, int current_grain, T* tensor_devPtr, int tensor_size){
    int factor = tp_grain / current_grain;
    int stage = tp_stage % factor;
    int _tensor_size = st_size / factor;

    // FT_LOG_DEBUG("assign %d == %d", _tensor_size, tensor_size);
    assert(_tensor_size == tensor_size);

    cudaD2Dcpy((devPtr + offset) + (_tensor_size * stage), tensor_devPtr, _tensor_size);
    // if(is_trans) is_modified = true;
}

template<class T>
void TensorWrapper<T>::slice_tensor_and_delete(int tp_stage, int tp_grain, int current_grain, T* del_dst_ptr){
    int factor = tp_grain / current_grain;
    int stage = tp_stage % factor;
    int tensor_size = st_size / factor;

    // malloc a new and delete rest
    // offset += tensor_size * stage;
    st_size /= factor;
    shape[0] /= factor;

    const bool no_malloc = del_dst_ptr != nullptr;

    T* sliced_ptr = del_dst_ptr;
    if(!no_malloc) deviceMalloc(&sliced_ptr, st_size, false);
    cudaD2Dcpy(sliced_ptr, devPtr + tensor_size * stage, st_size);

    if(!no_malloc) update_metadata(true, sliced_ptr);
    else{
        // free old ptr, and dont need to free when destory
        deviceFree(devPtr);
        devPtr = sliced_ptr;
        free_when_destory = false;
    }
}


template<class T>
void TensorWrapper<T>::transpose_inplace(T* buf, bool for_compute){
    if(!is_trans || shape.size() < 2 || shape[0] == 1 || shape[1] == 1) return;
    // comm -> compute
    // now_trans = true, for_compute = true
    // if(is_buffer){
    //     FT_LOG_DEBUG("transfer a buffer shape[%d, %d]", shape[0], shape[1]);
    // }
    if(for_compute ^ now_trans){
        FT_LOG_WARNING("Mismatch status: for_compute = %d, now_trans = %d", for_compute, now_trans);
    }
    invokeMatrixTransposeInplace(buf, devPtr + offset, shape[now_trans ? 0 : 1], shape[now_trans ? 1 : 0], cuda_stream);
    now_trans = !now_trans;
}

template<class T>
void TensorWrapper<T>::transpose_to(T* dst){
    invokeMatrixTransposeInplace(dst, devPtr + offset, shape[now_trans ? 0 : 1], shape[now_trans ? 1 : 0], cuda_stream);
}

template<class T>
TensorWrapper<T>::~TensorWrapper(){
    if(free_when_destory)
        deviceFree(devPtr);
}

template<class T>
void TensorWrapper<T>::assign_test_value(T base_value){
    T* host_ptr = new T[st_size];
    int x = shape[0], y = shape.size() > 1 ? shape[1] : 1;

    for(int i = 0; i < x; i++)
        for(int j = 0; j < y; j++)
            host_ptr[i * y + j] = base_value + (i * 100 + j);
    cudaH2Dcpy(devPtr + offset, host_ptr, st_size);

    delete[] host_ptr;
}

template<class T>
void TensorWrapper<T>::print_data(int verbose){
    printf("dim: %d\t", shape.size());
    if(shape.size() == 1) printf("shape: [%d]\t", shape[0]);
    else printf("shape: [%d, %d]\t", shape[0], shape[1]);

    if(verbose > 1) print_to_screen(devPtr + offset, st_size > verbose ? verbose : st_size);
    else putchar('\n');
}

template<class T>
void TensorWrapper<T>::load_init_value(const std::string& weight_path, bool print_debug_info){
    if(!is_trans || shape.size() < 2){
        if(shape.size() == 1)
            loadWeightFromBin<T>(devPtr, {(size_t)shape[0]}, weight_path);
        else
            loadWeightFromBin<T>(devPtr, {(size_t)shape[0], (size_t)shape[1]}, weight_path);
    }else{
        // T* temp_tensor;
        // deviceMalloc(&temp_tensor, st_size, false);
        // loadWeightFromBin<T>(temp_tensor, {(size_t)shape[1], (size_t)shape[0]}, weight_path);
        // invokeMatrixTranspose(devPtr, temp_tensor, shape[1], shape[0], cuda_stream);
        // sync_check_cuda_error();
        // deviceFree(temp_tensor);

        loadWeightFromBin<T>(devPtr, {(size_t)shape[1], (size_t)shape[0]}, weight_path);
    }

    if(print_debug_info){
        printf("is_trans %d\n", is_trans);
        print_to_screen(devPtr, 10);
    }
   
}