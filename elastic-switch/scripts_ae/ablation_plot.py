import os
import argparse
import seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ABLATION_RES_DIR = './log/h6144/ablation/'
E2E_RES_DIR = './log/h6144/real/'
OUTPUT_FILE = './graphs/ablation.png'

# color_palette = seaborn.color_palette('cubehelix', 5).as_hex()
# color_palette = '#72aa91', "#f1c586", '#fee9bc', '#f6b3bc', '#e88290'
# color_palette = "#C76DA2", "#F27970", "#BB9727", "#54B345", "#05b9e2"
color_palette = "#C76DA2", "#F5B0B5", "#91D1EC", "#F7DAF0", '#4D81C1'
# color_palette = seaborn.xkcd_palette(["windows blue", "amber", "greyish", "faded green", "dusty purple"])
hatches = ['', '///', '', '\\\\\\', '']

def percentile(datas, percents):
    px_indexes = int(len(datas) * percents / 100)
    return datas[px_indexes]


def plot_latency(filename, ax, label, max_lat, req_id=False):
    datas = []
    int_datas = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if 'latency' in line:
                if req_id:
                    x_id = int(line.split()[3])
                else:
                    x_id = float(line.split()[5])
                lat = float(line.split()[7][:-1]) / 1000
                if lat < max_lat:
                    datas.append((x_id, lat))
            if 'interrupt query' in line:
                query_id = int(line.split()[4])
                int_datas.append(query_id)

    datas.sort(key=lambda x: x[0])
    xs, lats = [], []
    for x_id, lat in datas:
        xs.append(x_id)
        lats.append(lat)

    # ax.plot(xs, lats, label=label, lw=1.0)

    # for query_id in int_datas:
    #     if query_id >= len(datas):
    #         continue
    #     query_arrival = datas[query_id][0]
    #     ax.axvline(query_arrival, color='lightgray', linestyle='--', linewidth=0.2)

    # avg latency
    avg_lat = sum(lats) / len(lats)
    tail_lat = percentile(sorted(lats), 99)
    # print(f'{label} avg latency: {avg_lat:.3f} ms, tail lat: {tail_lat:3f}, std: {np.std(lats):.3f}')
    return avg_lat, tail_lat

def axvline(query_trace, ax):
    t = 1
    with open(query_trace, 'r') as f:
        for line in f.readlines():
            req_id = int(line.split(',')[0])
            tstamp = float(line.split(',')[1])
            if tstamp >= t:
                ax.axvline(req_id, color='lightgray', linestyle='--', linewidth=0.5)
                t += 1


def plot_tpt(max_lat):
    plt.rcParams['hatch.color'] = 'white'
    mpl.rcParams['hatch.color'] = 'white'
    plt.rcParams["font.family"] = "Calibri"
    mpl.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1, 2, dpi=600, tight_layout=True)
    fig.set_size_inches(6, 2.5)

    x_ticks = ['$A_S$', '$B_S$']
    avg_lats, tail_lats = {}, {}

    map_str = {
        'plain': '$-$ Device Mapper',
        'match': '$-$ Interruption Arranger',
        'cache': '$-$ Migration Planner',
        'overlap': '$-$ Controller',
        'naive': 'SpotServe',
    }

    tpt = 0.35
    cv = 6
    figname = OUTPUT_FILE
    # figname = f'{model}_tpt{tpt}_cv{cv}_{suffix}_ablation.pdf'
    for approach in ['naive', 'overlap', 'cache', 'match', 'plain']:
        avg_lats[approach] = []
        tail_lats[approach] = []
        for suffix in ['0304', '0506']:
            if approach == 'naive':
                filename = os.path.join(E2E_RES_DIR, f'naive_tpt{tpt}_cv{cv}-{suffix}/inference_service.log')
            else:
                filename = os.path.join(ABLATION_RES_DIR, f'naive_tpt{tpt}_cv{cv}-{suffix}-{approach}/inference_service.log')
            if os.path.exists(filename):
                avg_lat, tail_lat = plot_latency(filename, ax, f'{approach.upper()}-TPT{tpt}', max_lat)
                avg_lats[approach].append(avg_lat)
                tail_lats[approach].append(tail_lat)
            else:
                avg_lats[approach].append(0)
                tail_lats[approach].append(0)
                print(f'File not found: {filename}, skipped.')

    # draw avg latency
    x = np.arange(len(x_ticks))
    width = 0.15
    for i, approach in enumerate(['naive', 'overlap', 'cache', 'match', 'plain']):
        if len(avg_lats[approach]) == 0:
            continue
        # print(approach, avg_lats[approach])
        ax[1].bar(x + i * width, avg_lats[approach], width, label=map_str[approach], color=color_palette[i], hatch=hatches[i])

    ax[1].set_xticks(x + width * 2)
    ax[1].set_xticklabels(x_ticks, fontsize=14)
    # ax[1].set_ylabel('Latency (s)')
    ax[1].set_title('Average Latency', fontsize=14)
    ax[1].set_ylim(0, 90)

    # draw tail latency
    for i, approach in enumerate(['naive', 'overlap', 'cache', 'match', 'plain']):
        if len(tail_lats[approach]) == 0:
            continue
        ax[0].bar(x + i * width, tail_lats[approach], width, label=map_str[approach], color=color_palette[i], hatch=hatches[i])
    ax[0].set_xticks(x + width * 2)
    ax[0].set_xticklabels(x_ticks)
    ax[0].set_ylim(0, 190)
    ax[0].set_ylabel('Latency (s)', fontsize=14)
    ax[0].set_title('P99 Tail Latency', fontsize=14)

    hadles, labels = ax[0].get_legend_handles_labels()
    fig.legend(hadles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.18), frameon=False, fontsize=14, columnspacing=0.1)
    fig.savefig(figname, bbox_inches='tight', pad_inches=0.05)
    print('Figure saved to', figname)

if __name__ == '__main__':
    plot_tpt(1000000)
