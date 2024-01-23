#include "src/fastertransformer/utils/request_pool.h"
#include <iostream>

Request::Request(std::ifstream& infile, int id, int offset)
    : id_(id),
      offset_(offset),
      seq_len_(0),
      start_step_(0),
      end_step_(-1),
      input_ids_(std::vector<int>()),
      submit_(std::chrono::system_clock::now()) {
    infile.seekg(std::ios::beg);
    infile.seekg(offset);
    std::string line;
    std::getline(infile, line);
    std::stringstream line_stream(line);
    std::string vals;
    while (std::getline(line_stream, vals, ',')) {
        vals.erase(std::remove_if(vals.begin(), vals.end(), [](char c) { return std::isspace(c); }),
                   vals.end());
        input_ids_.push_back(std::stoi(vals));
        seq_len_++;
    }
}

Request::~Request() {}

float Request::getScheduleLatency() {
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(start_ - submit_);
    return diff.count();
}

float Request::getInferenceLatency() {
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
    return diff.count();
}

RequestPool::RequestPool()
    : num_reqs_(0),
      num_resps_(0),
      init_batch_size_(-1),
      last_num_tokens_(-1),
      no_more_reqs_(false),
      notify_api_server_(false),
      req_conn_finish_(false),
      resp_conn_finish_(false),
      num_interrupt_reqs_(0),
      estimate_xfer_cost_(0) {}

RequestPool::~RequestPool() {}

void RequestPool::putRequest(std::ifstream& infile, int query_id, int query_offset) {
    if (no_more_reqs_) {
        return;
    }
    auto req_ptr = std::make_shared<Request>(infile, query_id, query_offset);
    std::lock_guard<std::mutex> lock(mtx_);
    req_queue_.push(req_ptr);
    num_reqs_ += 1;
}

void RequestPool::putRequest(std::ifstream& infile, int query_id, int query_offset, int start_step) {
    if (no_more_reqs_) {
        return;
    }
    auto req_ptr = std::make_shared<Request>(infile, query_id, query_offset);
    req_ptr->start_step_ = start_step;
    std::lock_guard<std::mutex> lock(mtx_);
    req_queue_.push(req_ptr);
    num_interrupt_reqs_ += 1;
    num_reqs_ += 1;
}

void RequestPool::putBatchRequest(std::ifstream& infile, std::vector<int>& ids, std::vector<int>& offsets) {
    if (no_more_reqs_) {
        return;
    }
    assert(ids.size() == offsets.size());
    std::lock_guard<std::mutex> lock(mtx_);
    for (int i = 0; i < ids.size(); i++) {
        auto req_ptr = std::make_shared<Request>(infile, ids[i], offsets[i]);
        req_queue_.push(req_ptr);
        num_reqs_ += 1;
    }
}

int RequestPool::getBatchRequests(int request_batch_size, int& start_step, int& end_step,
                                  std::vector<std::shared_ptr<Request>>& requests_batch) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (req_queue_.empty()) return 0;

    if (init_batch_size_ > 0) {
        assert(init_batch_size_ <= request_batch_size);
        if (num_interrupt_reqs_ == init_batch_size_) {
            start_step = req_queue_.top()->start_step_;
            while (requests_batch.size() < init_batch_size_) {
                auto request = popRequest();
                assert(request->start_step_ == start_step);
                requests_batch.push_back(request);
            }
            init_batch_size_ = -1;
            return requests_batch.size();
        }
        return 0;
    }

    if (no_more_reqs_ && req_queue_.size() <= request_batch_size) {
        end_step = last_num_tokens_ > 0 ? last_num_tokens_ : -1;
        while (!req_queue_.empty()) {
            requests_batch.push_back(popRequest());
        }
        return requests_batch.size();
    }

    while (requests_batch.size() < request_batch_size) {
        auto request = popRequest();
        if (request == nullptr) break;
        requests_batch.push_back(request);
    }
    return requests_batch.size();
}

std::shared_ptr<Request> RequestPool::popRequest() {
    if (req_queue_.empty()) {
        return nullptr;
    }
    std::shared_ptr<Request> req = req_queue_.top();
    req_queue_.pop();
    return req;
}

std::shared_ptr<Request> RequestPool::getRequest() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (req_queue_.empty()) {
        return nullptr;
    }
    std::shared_ptr<Request> req = req_queue_.top();
    req_queue_.pop();
    return req;
}

void RequestPool::putResponse(std::shared_ptr<Request> req) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (req->end_step_ > 0) {
        interrupt_resp_queue_.push(req);
    } else {
        resp_queue_.push(req);
    }
    num_resps_ += 1;
}

std::shared_ptr<Request> RequestPool::getResponse() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (resp_queue_.empty()) {
        return nullptr;
    }
    std::shared_ptr<Request> resp = resp_queue_.front();
    resp_queue_.pop();
    return resp;
}

void RequestPool::setEstimateXferCost(float cost) {
    signal_time_ = std::chrono::system_clock::now();
    estimate_xfer_cost_ = cost;
}

void RequestPool::setNoMoreRequests(int remain_reqs, int num_tokens) {
    std::lock_guard<std::mutex> lock(mtx_);
    no_more_reqs_ = true;

    std::cout << "remain_reqs: " << remain_reqs << " num_tokens: " << num_tokens
              << " but size: " << req_queue_.size() << std::endl;
    if (remain_reqs >= 0) {
        std::queue<std::shared_ptr<Request>> tmp_queue;
        while (remain_reqs > 0 && !req_queue_.empty()) {
            tmp_queue.push(req_queue_.top());
            req_queue_.pop();
            remain_reqs--;
        }

        num_reqs_ -= req_queue_.size();
        while (!req_queue_.empty()) {
            req_queue_.pop();
        }
        while (!tmp_queue.empty()) {
            req_queue_.push(tmp_queue.front());
            tmp_queue.pop();
        }
        last_num_tokens_ = num_tokens;
    }
}
