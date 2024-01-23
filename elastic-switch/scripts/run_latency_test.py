import os
import argparse
import json
import time
import subprocess
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    # trace replay args
    parser.add_argument('--parse-log', type=str, default=None)
    return parser.parse_args()


def run_exp(model, tpt, bs, strategy):
    # generate a trace and hostfile
    ngpu = np.prod(strategy)
    with open('trace/test_tmp.txt', 'w') as f:
        event = [0, 'add', {'nodes': [f'node-{idx}' for idx in range(ngpu)]}]
        f.write(f'{json.dumps(event)}\n')
        event = [3600000, 'DONE', {'nodes': []}]
        f.write(f'{json.dumps(event)}\n')

    # generate query trace
    cmd = f'python trace/gen_query.py --query trace/query/query_seq512.csv --seq-len 512 --n 100 --tpt {tpt} \
           --trace trace/query/lat_test.txt'
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    time.sleep(0.05)

    # run latency test
    current_env = os.environ.copy()
    current_env['FIXED_STRATEGY'] = ','.join(map(str, strategy))
    current_env['SLEEP_DUR'] = '50' if model == '6.7b' else '5'
    cmd = f'sh scripts/test_latency.sh {model} {tpt:.1f} {bs} {ngpu}'
    print(cmd)
    proc = subprocess.Popen(cmd, env=current_env, shell=True)
    proc.wait()
    time.sleep(1)


def main():
    # list strategies
    total_nnodes = 16
    strategy_candidates = []
    for tp in [1, 2, 4, 8, 16]:
        for pp in [1, 2, 4, 8, 16]:
            if tp * pp <= total_nnodes:
                strategy_candidates.append((1, tp, pp))
    strategy_candidates = sorted(strategy_candidates, key=lambda x: -np.prod(x))

    est_speed_plain = {
        '345m': {
            1: [537.523,581.929,543.346,609.616,559.563,547.266,713.119,569.111,552.767,562.137,687.905,786.111,737.823,701.639,554.133],
            2: [713.208,681.558,824.696,573.104,860.220,725.850,599.994,883.743,698.220,742.845,782.243,1213.815,824.600,846.337,792.879],
            4: [761.100,757.619,1094.104,612.100,1045.604,1016.478,641.485,884.158,1059.161,749.185,862.944,1340.050,1414.660,948.286,803.87]
        },
        '6.7b': {
            1: [3544.835,2029.303,3552.802,1341.460,2029.333,3535.864,992.716,1350.445,2039.859,3551.220,1058.501,1273.705,1653.504,2264.283,3612.004],
            2: [4131.125,2798.507,5340.588,1728.840,3070.579,4115.454,1349.395,2045.647,2800.056,4158.632,1358.785,1940.839,2101.705,3102.163,4246.391],
            4: [5035.266,3256.359,6235.908,1972.450,4205.414,6384.706,1468.904,2635.831,3692.508,4439.243,1562.050,2669.851,3216.160,3471.066,4473.937]
        }
    }
    plain_st_list = sorted(strategy_candidates, key=lambda x: np.prod(x))

    for bs in [1, 2, 4]:
        for model in ['6.7b', '345m']:
            for st in strategy_candidates:
                speed = est_speed_plain[model][bs][plain_st_list.index(st)] / 1000
                tpt = bs / speed

                print(f'{model}, {tpt:.1f}, {bs}, {st}')
                run_exp(model, tpt, bs, st)


def read_latencies(comp_lats, slo_lats, strategy, logdir):
    # read slo latencies
    with open(f'{logdir}/inference_service.log', 'r') as f:
        cur_lats = []
        for line in f.readlines():
            if 'latency' in line:
                service_latency = float(line.split()[5][:-1])
                cur_lats.append(service_latency)
        slo_lats.extend(cur_lats[40:])

    # read mini-batch latencies
    hist_lats = []
    with open(f'{logdir}/ft_inference_0.log', 'r') as f:
        for line in f.readlines():
            if 'Finish request' in line:
                lat = int(line.strip().split()[-2])
                hist_lats.append(lat)
    with open(f'{logdir}/ft_inference_0.log', 'r') as f:
        for line in f.readlines():
            if 'Processing request ids' in line:
                bs = len(line.strip().split(',')) - 1
                lat = np.mean(hist_lats[:bs])
                hist_lats = hist_lats[bs:]
                comp_lats[bs].append(lat)
        assert len(hist_lats) == 0


def parse_log(logdir):
    for model in os.listdir(logdir):
        print(f'Model: {model}')
        comp_lats = {}
        slo_lats = {}
        for log in os.listdir(f'{logdir}/{model}'):
            tpt = float(log.strip().split('_')[1][3:])
            mini_bs = int(log.strip().split('_')[2][2:])
            strategy = tuple(map(int, log.strip().split('_')[-1][2:].split(',')))

            if strategy not in slo_lats:
                slo_lats[strategy] = {bs: {} for bs in [1, 2, 3, 4]}
                comp_lats[strategy] = {bs: [] for bs in [1, 2, 3, 4]}
            if tpt not in slo_lats[strategy][mini_bs]:
                slo_lats[strategy][mini_bs][tpt] = []

            sub_logdir = f'{logdir}/{model}/{log}'
            read_latencies(comp_lats[strategy], slo_lats[strategy][mini_bs][tpt], strategy, sub_logdir)

        # print comp latency table
        strategies = sorted(list(comp_lats.keys()), key=lambda x: np.prod(x))
        for ptype in ['avg', 'max', 'p90']:
            for st in strategies:
                print(f'{st[0]}~{st[1]}~{st[2]}', end=', ')
            print()
            for bs in [1, 2, 4]:
                for st in strategies:
                    if ptype == 'max':
                        lat = max(sorted(comp_lats[st][bs]))
                    else:
                        nood = int(len(comp_lats[st][bs]) * 0.1) // 2 if ptype == 'p90' else 1
                        lat = np.mean(sorted(comp_lats[st][bs])[nood:-nood])
                    print(f'{lat:.3f}', end=',')
                print()
            print()

        # print slo latency table
        for bs in [1, 2, 4]:
            for st in strategies:
                print(f'{st[0]}~{st[1]}~{st[2]}', end=', ')
            print()
            arrivals = ''
            for slo in [0.8, 0.9, 0.95, 0.99]:
                for st in strategies:
                    min_lat = np.inf
                    min_tpt = np.inf
                    for tpt, serve_lats in slo_lats[st][bs].items():
                        serve_lats = sorted(serve_lats)
                        nslo = int(len(serve_lats) * slo)
                        if min_lat > serve_lats[nslo]:
                            min_lat = serve_lats[nslo]
                            min_tpt = tpt
                    print(f'{min_lat:.3f}', end=', ')
                    if slo == 0.8:
                        arrivals += f'{min_tpt:.2f}, '
                print()
            print(arrivals)
            print()


if __name__ == '__main__':
    args = parse_args()
    if args.parse_log:
        parse_log(args.parse_log)
    else:
        main()
