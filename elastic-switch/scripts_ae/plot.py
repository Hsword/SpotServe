import numpy as np
import argparse
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ablation_plot import plot_tpt
from tradeoff_plot import plot_all

LOG_DIR = './log'
OUTPUT_DIR = './graphs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PERCENTS = [90, 95, 96, 97, 98, 99]
XTICKS = ['AVG'] + [f'P{p}' for p in PERCENTS]
APPROACHES = 'reparallelization', 'rerouting', 'spotserve'
APPROACHES_LABEL = 'baseline', 'baseline-triton', 'naive'
map_str = {
    'baseline': 'Reparallelization',
    'baseline-triton': 'Rerouting',
    'naive': 'SpotServe',
}
map_model = {
    '6.7B': '6.7B',
    '20B' : 'h6144',
    '30B' : 'h7168'
}

color_palette = ["#05b9e2", "#54B345", "#C76DA2"]

def plot_lat(ax, data, title):
    ax.set_xticks(np.arange(len(XTICKS) + 1))
    ax.set_xticklabels(XTICKS + ['o'], fontsize=15, rotation=45)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    

    for i, (label, data) in enumerate(data.items()):
        if data is None:
            continue
        ls = None
        label = map_str[label]
        ax.plot(data, label=label, lw=2.0, ls=ls, marker='o', markersize=3, color=color_palette[i])

    ax.set_title(title, fontsize=16)

def plot_latency(filename, ax, label, max_lat, req_id=False, color=None, end_time=True):
    datas = []
    int_datas = []
    
    if not os.path.isfile(filename):
        print('WARNING: file not found:', filename)
        return
    
    with open(filename, 'r') as f:
        for line in f.readlines():
            if 'latency' in line:
                if req_id:
                    x_id = int(line.split()[3])
                else:
                    x_id = float(line.split()[5])
                lat = float(line.split()[7][:-1]) / 1000
                if lat < max_lat:
                    if end_time:
                        datas.append((x_id + lat, lat))
                    else:
                        datas.append((x_id, lat))
            if 'interrupt query' in line:
                query_id = int(line.split()[4])
                int_datas.append(query_id)

    datas.sort(key=lambda x: x[0])
    xs, lats, max_lats = [], [], []
    # cluster nearby points
    x_seg, lat_seg = [], []
    for x_id, lat in datas:
        if len(x_seg) == 0:
            x_seg.append(x_id)
            lat_seg.append(lat)
            continue
        if x_id - x_seg[-1] <= 0.5:
            x_seg.append(x_id)
            lat_seg.append(lat)
        else:
            xs.append(np.mean(x_seg))
            lats.append(np.mean(lat_seg))
            max_lats.append(max(lat_seg))
            x_seg, lat_seg = [], []
            x_seg.append(x_id)
            lat_seg.append(lat)
    if len(x_seg) > 0:
        xs.append(np.mean(x_seg))
        lats.append(np.mean(lat_seg))
        max_lats.append(max(lat_seg))
    
    # xs = [e[0] for e in datas]
    # lats = [e[1] for e in datas]
    
    new_xs, new_lats = [], []
    all_pieces, cur_piece = [], []
    smooth_type = 1
    if smooth_type == 1:
        # manuall split
        for x, lat in zip(xs, lats):
            if len(cur_piece) == 0:
                cur_piece.append((x, lat))
                continue
            if lat <= cur_piece[-1][1]:
                cur_piece.append((x, lat))
            else:
                all_pieces.append(cur_piece)
                cur_piece = [(x, lat)]
        if len(cur_piece) > 0:
            all_pieces.append(cur_piece)
        for piece in all_pieces:
            new_xs.append(np.mean([x[0] for x in piece]))
            new_lats.append(np.mean([x[1] for x in piece]))
    elif smooth_type == 2:
        # window smooth
        window_size = 4
        block_size = 2
        for i in range(len(xs)):
            x = xs[i]
            lat = np.mean(lats[max(0, i-window_size):i+window_size])
            if i % block_size == 0:
                new_xs.append(x)
                new_lats.append(lat)
    else:
        new_xs = xs
        new_lats = lats

    ax.axhline(np.mean([x[1] for x in datas]), lw=0.5, ls='-.', color=color)
    line, = ax.plot(new_xs, new_lats, label=label, lw=1.2, color=color)
    return line

def stat_latency(filename):
    datas = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if 'latency' in line:
                lat = float(line.split()[7][:-1])
                datas.append(lat)

    datas.sort()
    px_indexes = [int(len(datas) * p / 100) for p in PERCENTS]
    avg_lat = np.mean(datas)
    px_lat = list(map(lambda x: datas[x], px_indexes))
    return list(map(lambda x: x/1000 , [avg_lat] + px_lat))

def get_e2e_data(model, subdir, dirname):
    data = {'baseline':None, 'baseline-triton':None, 'naive':None}
    for approach_label in APPROACHES_LABEL:
        filename = os.path.join(LOG_DIR, map_model[model], subdir, f'{approach_label}_{dirname}', 'inference_service.log')
        if not os.path.exists(filename):
            print(f'Cannot find log file {filename}, skipped.')
            continue
        data[approach_label] = stat_latency(filename)
    return data

def do_e2e(model, trace):
    assert model in ('6.7B', '20B', '30B'), 'model must be one of 6.7B, 20B, 30B'
    assert trace in ('As', 'Bs', 'As+o', 'Bs+o'), 'trace must be one of As, Bs, As+o, Bs+o'
    
    trace_name = '0304' if 'A' in trace else '0506'
    subdir = 'ondemand' if 'o' in trace else 'real'
    if model == '6.7B':
        config_str = 'tpt1.5_cv6'
    elif model == '20B':
        config_str = 'tpt0.35_cv6'
    elif model== '30B':
        config_str = 'tpt0.2_cv6'
    dirname = f'{config_str}-{trace_name}'
    
    data = get_e2e_data(model, subdir, dirname)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_lat(ax, data, f'{model} on {trace}')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylabel('Latency (s)', fontsize=14)
    fig.savefig(os.path.join(OUTPUT_DIR, f'e2e-{model}_{trace}.png'), bbox_inches='tight', dpi=300)
    print(f'Figure saved to {OUTPUT_DIR}/e2e-{model}_{trace}.png')

def do_workload_e2e(trace):
    assert trace in ('A', 'B'), 'trace must be one of A, B'
    
    trace_name = '0304' if trace == 'A' else '0506'
    subdir = 'workload'
    model = '20B'
    config_str = 'realAr_cv6'
    dirname = f'{config_str}-{trace_name}'
    
    data = get_e2e_data(model, subdir, dirname)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_lat(ax, data, f'{model} on ${trace}' + "_{s+o}'$")
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylabel('Latency (s)', fontsize=14)
    fig.savefig(os.path.join(OUTPUT_DIR, f'workload_e2e-{trace}.png'), bbox_inches='tight', dpi=300)
    print(f'Figure saved to {OUTPUT_DIR}/workload_e2e-{trace}.png')

def do_workload_case(trace):
    assert trace in ('A', 'B'), 'trace must be one of A, B'
    
    trace_name = '0304' if trace == 'A' else '0506'
    subdir = 'workload'
    model = '20B'
    config_str = 'realAr_cv6'
    dirname = f'{config_str}-{trace_name}'
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for i, approach_label in enumerate(APPROACHES_LABEL):
        filename = os.path.join(LOG_DIR, map_model[model], subdir, f'{approach_label}_{dirname}', 'inference_service.log')
        if not os.path.exists(filename):
            print(f'Cannot find log file {filename}, skipped.')
            continue
        plot_latency(filename, ax, map_str[approach_label], 120000, color=color_palette[i])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylabel('Latency (s)', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=14)
    fig.savefig(os.path.join(OUTPUT_DIR, f'workload_case-{trace}.png'), bbox_inches='tight', dpi=300)
    print(f'Figure saved to {OUTPUT_DIR}/workload_case-{trace}.png')

def do_price():
    opt_20b_avg = {}
    opt_20b_p99 = {}
    for trace_name in ['0304', '0506']:
        for trace_type in ['real', 'ondemand']:
            opt_20b_avg[f'{trace_name}-{trace_type}'] = [0, 0, 0]
            opt_20b_p99[f'{trace_name}-{trace_type}'] = [0, 0, 0]
            for i, approach_label in enumerate(APPROACHES_LABEL):
                dirname = f'tpt0.35_cv6-{trace_name}'
                filename = os.path.join(LOG_DIR, 'h6144', trace_type, f'{approach_label}_{dirname}', 'inference_service.log')
                if not os.path.exists(filename):
                    print(f'Cannot find log file {filename}, skipped.')
                    continue
                res = stat_latency(filename)
                opt_20b_avg[f'{trace_name}-{trace_type}'][i] = res[0]
                opt_20b_p99[f'{trace_name}-{trace_type}'][i] = res[-1]
    
    opt_20b_ondemand_avg = []
    opt_20b_ondemand_p99 = []
    for nnode in [3, 4, 6, 8]:
        dirname = f'tpt0.35_cv6-node{nnode}'
        filename = os.path.join(LOG_DIR, 'h6144', 'ondemand', f'naive_{dirname}', 'inference_service.log')
        if not os.path.exists(filename):
            print(f'Cannot find log file {filename}, skipped drawing on-demand dashed line.')
            opt_20b_ondemand_avg = opt_20b_ondemand_p99 = None
            break
        res = stat_latency(filename)
        opt_20b_ondemand_avg.append(res[0])
        opt_20b_ondemand_p99.append(res[-1])
    else:
        opt_20b_ondemand_avg = np.array(opt_20b_ondemand_avg)
        opt_20b_ondemand_p99 = np.array(opt_20b_ondemand_p99)
    
    plot_all(opt_20b_avg, opt_20b_ondemand_avg, opt_20b_p99, opt_20b_ondemand_p99)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['e2e', 'price', 'workload-e2e', 'workload-case', 'ablation'])
    parser.add_argument('-m', '--model', type=str, default='???')
    parser.add_argument('-t', '--trace', type=str, default='???')
    
    args = parser.parse_args()
    if args.mode == 'e2e':
        do_e2e(args.model, args.trace)
    elif args.mode == 'workload-e2e':
        if args.model != '???':
            print('WARNING: model is not used in this mode.')
        do_workload_e2e(args.trace)
    elif args.mode == 'workload-case':
        if args.model != '???':
            print('WARNING: model is not used in this mode.')
        do_workload_case(args.trace)
    elif args.mode == 'ablation':
        if args.model != '???':
            print('WARNING: model is not used in this mode.')
        if args.trace != '???':
            print('WARNING: trace is not used in this mode.')
        plot_tpt(1000000)
    elif args.mode == 'price':
        if args.model != '???':
            print('WARNING: model is not used in this mode.')
        if args.trace != '???':
            print('WARNING: trace is not used in this mode.')
        do_price()
    else:
        print('Unknown mode:', args.mode)
        sys.exit(1)
