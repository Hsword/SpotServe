#include <fstream>

#include "examples/cpp/multi_gpu_gpt/gpt_estimation_utils.h"
#include "3rdparty/json.hpp"

using json = nlohmann::json;

double CostEstimator::real_t1 = -1;
double CostEstimator::real_t2 = -1;

CostEstimator::CostEstimator(const std::string& file_name, int tp, int pp, int _M1, int _M2, int output_seqlen){
    output_seq_len = output_seqlen;
    tp_degree = tp;
    pp_degree = pp;
    M1 = _M1;
    M2 = _M2;

    std::ifstream f(file_name);
    json data = json::parse(f);

    data.at("pp_init_comm").get_to<double>(pp_init_comm);
    data.at("pp_incr_comm").get_to<double>(pp_incr_comm);
    data.at("final_fix_factor").get_to<double>(final_fix_factor);

    data.at("layer_num").get_to<int>(real_layer_num);
    data.at("padded_layer_num").get_to<int>(padded_layer_num);
    data.at("embed").get_to<std::vector<double>>(embedding);

    for(int b = 1; b <= 8; b *= 2){
        inits.push_back(data.at("tp-"+std::to_string(tp)).at("bsz-"+std::to_string(b)).at("init").get<double>());
        incrs.push_back(data.at("tp-"+std::to_string(tp)).at("bsz-"+std::to_string(b)).at("incr").get<double>());
    }

    f.close();
}

double CostEstimator::get_t1(int bsz, bool conserve) const{
    // in sec
    if(real_t1 > 0) return real_t1;

    int nM_1;
    if (bsz % M1 == 0)
        nM_1 = M1;
    else
        nM_1 = 1;
    int factor = (real_layer_num + pp_degree - 1) / pp_degree;
    double init = inits[get_index(bsz / nM_1)];
    double t1 = init * factor * (pp_degree + nM_1 - 1) / nM_1;
    t1 += pp_init_comm * (pp_degree - 1) / nM_1;
    if(conserve)
        t1 *= final_fix_factor / 1000;  // in sec
    else
        t1 /= final_fix_factor * 1000;  // in sec

    return t1;
}

double CostEstimator::get_t2(int bsz, bool conserve) const{
    if(real_t2 > 0) return real_t2;

    int nM_2;
    if (bsz % M2 == 0)
        nM_2 = M2;
    else
        nM_2 = 1;
    int factor = (real_layer_num + pp_degree - 1) / pp_degree;
    double incr = incrs[get_index(bsz / nM_2)];
    double t2 = incr * factor * (pp_degree + nM_2 - 1);
    t2 += pp_incr_comm * (pp_degree - 1) / nM_2;
    if(conserve)
        t2 *= final_fix_factor / 1000;  // in sec
    else
        t2 /= final_fix_factor * 1000;  // in sec
    return t2;
}

int CostEstimator::calc_remain_step(double slot, int bsz, bool conserve) const {
    if (bsz <= 0) return -1;

    double t1 = get_t1(bsz, conserve);
    double t2 = get_t2(bsz, conserve);

    int step = (int)((slot - t1) / t2) + (conserve ? -1 : 1);
    if(step <= 0) return 0;
    if(step >= output_seq_len) return -1;
    return step;
}

void CostEstimator::printInfo() const{
    printf("inits:\n");
    for(auto iter : inits){
        printf("%.3f ", iter);
    }
    printf("\nincr:\n");
    for(auto iter : incrs){
        printf("%.3f ", iter);
    }
    printf("\n");
}
