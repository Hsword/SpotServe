#include <string>
#include <vector>

class CostEstimator{
private:
    int output_seq_len;
    int tp_degree;
    int pp_degree;
    int real_layer_num;
    int padded_layer_num;
    int M1, M2;

    std::vector<double> embedding;
    std::vector<double> inits;
    std::vector<double> incrs;

    double final_fix_factor;
    double pp_init_comm;
    double pp_incr_comm;

    static double real_t1;
    static double real_t2;

public:
    CostEstimator(const std::string& file_name, int tp, int pp, int M1, int M2, int output_seqlen=128);
    int calc_remain_step(double slot, int bsz, bool conserve) const;
    int get_index(int bsz) const {
        switch (bsz)
        {
        case 1:
            return 0;
        case 2:
            return 1;
        case 4:
            return 2;
        case 8:
            return 3;
        }
        return -1;
    }
    void printInfo() const;
    static void set_real_t1(double t_in_sec){real_t1 = t_in_sec;}
    static void set_real_t2(double t_in_sec){real_t2 = t_in_sec;}

    double get_t1(int bsz, bool conserve) const;
    double get_t2(int bsz, bool conserve) const;
};
