#pragma once
#include <assert.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

class Request {
public:
    Request(std::ifstream& infile, int id, int offset);
    ~Request();

    float getScheduleLatency();
    float getInferenceLatency();
    int getSeqLen() { return seq_len_; }

    void setStart() { start_ = std::chrono::system_clock::now(); }
    void setEnd() { end_ = std::chrono::system_clock::now(); }

    int id_;
    int offset_;
    int seq_len_;
    int start_step_;
    int end_step_;
    std::vector<int> input_ids_;
    std::chrono::system_clock::time_point submit_;  // when this request is created
    std::chrono::system_clock::time_point start_;   // when this request is scheduled
    std::chrono::system_clock::time_point end_;     // when this request is completed
};


class RequestPool {
public:
    RequestPool();
    ~RequestPool();

    int numPendingRequests() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (init_batch_size_ > 0 && num_interrupt_reqs_ != init_batch_size_) return 0;
        return req_queue_.size();
    }
    void putRequest(std::ifstream& infile, int id, int offset);
    void putRequest(std::ifstream& infile, int id, int offset, int start_step);
    void putBatchRequest(std::ifstream& infile, std::vector<int>& ids, std::vector<int>& offsets);
    void setInitBatchSize(int batch_size) { init_batch_size_ = batch_size; }
    int getBatchRequests(int request_batch_size, int& start_step, int& end_step,
                         std::vector<std::shared_ptr<Request>>& requests_batch);
    std::shared_ptr<Request> popRequest();
    std::shared_ptr<Request> getRequest();
    void putResponse(std::shared_ptr<Request> req);
    std::shared_ptr<Request> getResponse();

    void setEstimateXferCost(float cost);
    void setNoMoreRequests(int remain_reqs, int num_tokens);
    bool isRequestFinished() {
        std::lock_guard<std::mutex> lock(mtx_);
        return no_more_reqs_ && (req_queue_.size() == 0);
    }
    void setFinished() {
        std::lock_guard<std::mutex> lock(mtx_);
        no_more_reqs_ = true;
        num_reqs_ -= req_queue_.size();
    }
    bool isFinished() {
        std::lock_guard<std::mutex> lock(mtx_);
        return no_more_reqs_ && (num_reqs_ == num_resps_);
    }
    void setNotifyApiServer() { notify_api_server_ = true; }
    bool needNotifyApiServer() { return notify_api_server_; }
    void setReqConnFinish() {
        std::lock_guard<std::mutex> lock(mtx_);
        req_conn_finish_ = true;
    }
    void setRespConnFinish() {
        std::lock_guard<std::mutex> lock(mtx_);
        resp_conn_finish_ = true;
    }
    bool isConnFinish() {
        std::lock_guard<std::mutex> lock(mtx_);
        return req_conn_finish_ && resp_conn_finish_;
    }

    // mutex for requests and responses, maybe we need two mutexes
    std::mutex mtx_;
    int num_reqs_;
    int num_resps_;
    int init_batch_size_;
    int last_num_tokens_;
    bool no_more_reqs_;
    bool notify_api_server_;
    bool req_conn_finish_;
    bool resp_conn_finish_;

    int num_interrupt_reqs_;

    std::chrono::system_clock::time_point signal_time_;
    float estimate_xfer_cost_;

    // build a priority queue for requests, the priority is determined by the start step and the request id
    // the larger the start step and the smaller the id, the higher the priority
    struct ReqCompare {
        bool operator()(const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            if (a->start_step_ == b->start_step_) {
                return a->id_ > b->id_;
            } else {
                return a->start_step_ < b->start_step_;
            }
        }
    };
    std::priority_queue<std::shared_ptr<Request>, std::vector<std::shared_ptr<Request>>, ReqCompare> req_queue_;
    std::queue<std::shared_ptr<Request>> resp_queue_;
    std::queue<std::shared_ptr<Request>> interrupt_resp_queue_;
};
