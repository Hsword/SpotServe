import os
import numpy as np
import json
from functools import cmp_to_key

class StrategySolver:
    def __init__(self, model):
        self.model = model

    def solve(self, prev_strategy, nnodes):
        # manually set strategy
        if nnodes == 0:
            return (0, 0, 0)
        elif nnodes == 1:
            return (1, 1, 1)

        # FIXME: fixed strategy for latency test
        if os.environ.get('FIXED_STRATEGY', None):
            strategy = tuple(map(int, os.environ['FIXED_STRATEGY'].split(',')))
            assert nnodes == np.prod(strategy)
            return strategy

        if nnodes == 8:
            dp, tp, pp = 2, 2, 2
            return (dp, tp, pp)
        else:
            dp, tp = 1, 2
        pp = nnodes // (dp * tp)
        strategy = (dp, tp, pp)
        return strategy

class StrategyOptimizer:
    def __init__(self, model_profile_path, min_world_size, output_seq_len) -> None:
        with open(model_profile_path, 'r') as f:
            self.profile_data = json.load(f)
        self.min_ws = min_world_size
        self.output_seq_len = output_seq_len
        self.max_tp = self.profile_data["max_tp"]

    def solve(self, prev_strategy, nnodes, max_bsz, min_tpt, output_seq_len=None, debug=False):
        if self.min_ws > nnodes:
            return (0, ) * 8

        if output_seq_len is None:
            output_seq_len = self.output_seq_len
        res = []

        bsz_sets = []
        bsz = 1
        while bsz <= max_bsz:
            bsz_sets.append(bsz)
            bsz *= 2

        max_tpt, mtpt_res = 0, None
        tp = 1
        while tp <= self.max_tp and tp <=nnodes:
            for bsz_idx in range(len(bsz_sets)):
                bsz = bsz_sets[bsz_idx]
                for M1_i in range(bsz_idx + 1):
                    for M2_i in range(bsz_idx + 1):
                        M1, M2 = bsz_sets[M1_i], bsz_sets[M2_i]
                        pp = (self.min_ws + tp - 1) // tp
                        while pp * tp <= nnodes:
                            if self.profile_data['padded_layer_num'] % pp != 0:
                                pp += 1
                                continue
                            latency, _, _ = self.get_approx_latency(tp, pp, bsz, M1, M2, output_seq_len)
                            dp = nnodes // (pp * tp)
                            tpt = dp * bsz / latency * 1000
                            if tpt >= min_tpt:
                                res.append((dp, tp, pp, bsz, M1, M2, latency, tpt))
                            if tpt > max_tpt:
                                mtpt_res = (dp, tp, pp, bsz, M1, M2, latency, tpt)
                                max_tpt = tpt
                            pp += 1
            tp *= 2

        min_lat = float('inf')
        idx = -1
        for i, tup in enumerate(res):
            if tup[6] < min_lat:
                min_lat = tup[6]
                idx = i

        # for debug
        if debug:
            def res_cmp(a, b):
                # return b[7] - a[7] if a[7] != b[7] else a[6] - b[6]
                return a[6] - b[6] if a[6] != b[6] else b[7] - a[7]
            sort_res = sorted(res, key=cmp_to_key(res_cmp))
            for tup in sort_res:
                print("dp=%d tp=%d pp=%d bsz=%d M1=%d M2=%d latency=%f tpt=%f"%(tup))

        return res[idx] if idx >= 0 else mtpt_res

    def get_approx_latency(self, tp, pp, bsz, M1, M2, output_seq_len=None):
        if output_seq_len is None:
            output_seq_len = self.output_seq_len
        factor = (self.profile_data['layer_num'] + pp - 1) // pp # floor
        init = self.profile_data[f"tp-{tp}"][f"bsz-{bsz//M1}"]['init']
        incr = self.profile_data[f"tp-{tp}"][f"bsz-{bsz//M2}"]['incr']

        t1 = init * factor * (pp + M1 - 1) / M1
        t1 += self.profile_data['pp_init_comm']/M1 * (pp-1)
        t2 = incr * factor * (pp + M2 - 1)
        t2 += self.profile_data['pp_incr_comm']/M2 * (pp-1)

        lat = t1+t2*output_seq_len + 2*self.profile_data['embed'][int(bsz).bit_length()-1]
        lat *= self.profile_data['final_fix_factor']
        return lat, t1 * self.profile_data['final_fix_factor'], t2 * self.profile_data['final_fix_factor']

    def estimate_remain_steps(self, slot, tp, pp, bsz, M1, M2):
        lat, t1, t2 = self.get_approx_latency(tp, pp, bsz, M1, M2)
        t1, t2 = t1/1000, t2/1000
        step = int((slot - t1) / t2 + 1)
        if step <= 0:
            return 0
        elif step >= self.output_seq_len:
            return -1
        else:
            return step

    def dp_predict(self, prev_strategy, s0, future_nodes, max_tp, max_bsz, min_tpt, arrival_rate, output_seq_len=None, debug=False):
        # s0, number of request which is not finished now
        # future_nodes: list[3], availble nodes when [now, 1min later, 2min later]
        # min_tpt, arrival_rate: given in req/sec
        if output_seq_len is None:
            output_seq_len = self.output_seq_len
        assert(len(future_nodes) == 3)
        last_cfg = self.solve(None, future_nodes[2], max_bsz, output_seq_len, min_tpt)
        min_lat, res = self._do_dp(0, prev_strategy, s0, last_cfg, future_nodes, max_tp, max_bsz, output_seq_len, arrival_rate, debug=debug)
        return res


    def _do_dp(self, step, cfg, s, last_cfg, future_nodes, max_tp, max_bsz, arrival_rate, output_seq_len=None, debug=False):
        if output_seq_len is None:
            output_seq_len = self.output_seq_len
        # arrival_rate in req / s
        if step == 2:
            return s / last_cfg[7], []

        min_lat = float('inf')
        min_res = []

        bsz_sets = []
        bsz = 1
        while bsz <= max_bsz:
            bsz_sets.append(bsz)
            bsz *= 2

        if step == 1:
            s_list = [last_cfg]
        else:
            s_list = self.gen_all_strat(max_tp, max_bsz, future_nodes[step + 1])

        for dp, tp, pp, bsz, M1, M2, *rest in s_list:

            l, _, _ = self.get_approx_latency(tp, pp, bsz, M1, M2, output_seq_len) #s
            print(tp, pp, bsz, M1, M2, output_seq_len, l)
            tpt = dp * bsz / l * 1000 # res/s
            cur_cfg = (dp, tp, pp, bsz, M1, M2, l, tpt)

            qi = (30 - self.get_swich_time(cfg[:3], (dp, tp, pp), debug=debug)) / tpt
            next_s = max(s + 30 * arrival_rate - qi, 0)
            if debug:
                print(step, cur_cfg, qi)

            next_dp, res_next = self._do_dp(step+1, cur_cfg, next_s, last_cfg, future_nodes, max_tp, max_bsz, output_seq_len, arrival_rate)
            val = next_dp + qi * (l / 1000)
            if min_lat > val:
                min_lat = val
                min_res = [cur_cfg] + res_next

        return min_lat, min_res

    def get_swich_time(self, old_cfg, new_cfg, debug=False):
        if old_cfg == new_cfg:
            return 0
        tot = self.profile_data['hidden_size'] ** 2
        p_n = min(old_cfg[1] * old_cfg[2], new_cfg[1] * new_cfg[2])
        param_n = tot / p_n * self.profile_data['padded_layer_num']
        param_n *= 4 * 12 * 1e-6 # MB

        res = param_n * 8 / (50000 / 2)
        if debug:
            print("estimate from", old_cfg, "to", new_cfg, res)
        # return res
        return 10.0

    def gen_all_strat(self, max_tp, max_bsz, n_nodes):
        res = []
        tp = 1
        bsz_sets = []
        bsz = 1
        while bsz <= max_bsz:
            bsz_sets.append(bsz)
            bsz *= 2
        while tp <= max_tp and tp <=self.min_ws:
            pp = (self.min_ws + tp - 1) // tp
            while pp * tp <= n_nodes:
                if self.profile_data['padded_layer_num'] % pp != 0:
                    pp += 1
                    continue
                dp = n_nodes // (pp * tp)
                for bsz_idx in range(len(bsz_sets)):
                    bsz = bsz_sets[bsz_idx]
                    for M1_i in range(bsz_idx):
                        for M2_i in range(bsz_idx):
                            M1, M2 = bsz_sets[M1_i], bsz_sets[M2_i]
                            res.append((dp, tp, pp, bsz, M1, M2))
                pp += 1
            tp *= 2
        return res



if __name__ == '__main__':
    # optim = StrategyOptimizer("../../profile/T4-4x/megatron_345M_profile.json", 1, 128)
    # optim = StrategyOptimizer("../../profile/T4-4x/megatron_6.7B_profile.json", 4, 128)
    optim = StrategyOptimizer("../../profile/T4-4x/megatron_h6144_profile.json", 12, 128)
    # optim = StrategyOptimizer("../../profile/T4-4x/megatron_h7168_profile.json", 16, 128)

    # print('*****res', optim.solve(None, 23, 4, 4, 32, 0, debug=True))

    tpt = 0.8
    # print(optim.dp_predict((2, 8, 2), 3, [32, 36, 32], 16, 4, tpt, tpt/2, output_seq_len=128, debug=True))

    print('Required TPT=', tpt)
    for avail_nodes in range(48, 0, -4):
        print("nodes:", avail_nodes ,"dp=%d tp=%d pp=%d bsz=%d M1=%d M2=%d latency=%f tpt=%f"%(optim.solve(None, avail_nodes, 4, tpt, debug=False)))
