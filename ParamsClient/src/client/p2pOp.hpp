#pragma once
#include "src/utils/nccl_utils.h"

using namespace fastertransformer;

template<class T>
class P2pOp{
    cudaStream_t cuda_stream;
    int peer, data_size;
    T* devPtr;
    bool prior;
public:
    enum p2pOpType {isend, irecv} op_type;
    P2pOp(p2pOpType _op_type, T* devPtr, int _peer, int _data_size, cudaStream_t _cuda_stream, bool _prior);
    void do_op(const NcclParam& ncclParam) const;
    bool get_is_prior() const {return prior; }
    int get_peer() const {return peer; }
    std::string toString() const{
        return std::string(op_type==isend ? "p2pOp: send peer=" : "p2pOp: recv peer=") + std::to_string(peer) + ", size="+std::to_string(data_size);
    }
};

template<class T>
P2pOp<T>::P2pOp(p2pOpType _op_type, T* _devPtr, int _peer, int _data_size, cudaStream_t _cuda_stream, bool _prior)
    : op_type(_op_type)
    , devPtr(_devPtr)
    , peer(_peer)
    , cuda_stream(_cuda_stream)
    , data_size(_data_size)
    , prior(_prior)
{}

template<class T>
void P2pOp<T>::do_op(const NcclParam& ncclParam) const {
    if(op_type == p2pOpType::isend)
        ftNcclSend(devPtr, data_size, peer, ncclParam, cuda_stream);
    else
        ftNcclRecv(devPtr, data_size, peer, ncclParam, cuda_stream);
}