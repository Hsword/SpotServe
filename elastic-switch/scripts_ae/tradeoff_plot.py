import os
import json
import numpy as np
import pandas as pd
import seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec


E2E_RESULT_DIR = './log/h6144/'
OD_RESULT_DIR = './log/h6144/ondemand/'
OUTPUT_FILE = './graphs/price.png'

# color_palette = seaborn.color_palette().as_hex()
color_palette = "#e08e51", "#f16fac", "#8850ac", "#315b70"
markers = ['x', '+', 'o', '^']

trace_tags = ['0304-real', '0506-real', '0304-ondemand', '0506-ondemand']
trace_names = [f'Trace ${s}$' for s in ('A_S', 'B_S', 'A_{S+O}', 'B_{S+O}')]
approaches = 'baseline', 'baseline-triton', 'naive'

# ondemand node price: 3.912 dollar/hour
# spot node price: 1.9046 dollar/hour
trace_price = {
    '0304-real': (203 / 60) * 1.9046,
    '0506-real': (137 / 60) * 1.9046,
    '0304-ondemand': (210 / 60) * 1.9046 + (27 / 60) * 3.912,
    '0506-ondemand': (92 / 60) * 1.9046 + (26 / 60) * 3.912,
}

map_str = {
    'baseline': 'Reparallelization',
    'baseline-triton': 'Rerouting',
    'naive': 'SpotServe',
}

trace_map_str = {
    '0304-real': '$A_S$',
    '0506-real': '$B_S$',
    '0304-ondemand': '$A_{S+O}$',
    '0506-ondemand': '$B_{S+O}$',
}

verbose = False

def request_latency(filename, req_id=False):
    if not os.path.isfile(filename):
        if verbose:
            print('WARNING: file not found:', filename)
        return 0
    datas = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if 'latency' in line:
                if req_id:
                    x_id = int(line.split()[3])
                else:
                    x_id = float(line.split()[5])
                lat = float(line.split()[7][:-1])
                datas.append((x_id, lat))

    datas.sort(key=lambda x: x[0])
    return len(datas)


def plot_price(ax, xticks, trace_tags, trace_price, datas, num_reqs_data, ondemand_prices, ondemand_lats):
    line, = ax.plot(ondemand_prices, ondemand_lats, linestyle='--', linewidth=1.2, label='Ondemand', color='#7797e1')
    for trace_tag, data in datas.items():
        for i in range(len(data)):
            ti = trace_tags.index(trace_tag)
            # print(ti, trace_tag, price, data[i])
            num_reqs = num_reqs_data[xticks[i]][ti] if num_reqs_data is not None else 1
            if num_reqs == 0 or data[i] <= 0:
                continue
            price = trace_price[trace_tag] * 1e5 / (num_reqs*640)
            ax.scatter(price, data[i], marker=markers[i], color=color_palette[ti], s=15)
            if i == 2:
                print(f'Price of trace {trace_tag}:', price)
    return line


def plot_pl(opt_20b_num_reqs, opt_20b_res, opt_20b_ondemand_lats, ax):
    xticks = ['Reparallelization', 'Rerouting', 'SpotServe']

    
    ngpus = [12, 16, 24, 32]
    opt_20b_ondemand_prices_pt = [] # per time
    
    if opt_20b_ondemand_lats is not None:
        for ngpu in ngpus:
            nnode = ngpu//4
            filename = os.path.join(OD_RESULT_DIR, f'naive_tpt0.35_cv6-node{nnode}/inference_service.log')
            nrequests = request_latency(filename)
            opt_20b_ondemand_prices_pt.append(nnode * 3.912 * (18/60) * 1e5 / (nrequests*640))
    print(opt_20b_ondemand_prices_pt)
    
    line = plot_price(ax, xticks, trace_tags, trace_price, opt_20b_res, opt_20b_num_reqs, opt_20b_ondemand_prices_pt, opt_20b_ondemand_lats)
    ax.set_xlabel(r'Cost($\times1e^{-5}$ USD/token)')
    return line

def plot_all(opt_20b_avg, opt_20b_ondemand_avg, opt_20b_p99, opt_20b_ondemand_p99):
    plt.rcParams['hatch.color'] = 'white'
    mpl.rcParams['hatch.color'] = 'white'
    plt.rcParams["font.family"] = "Calibri"
    mpl.rcParams.update({'font.size': 14})
    
    fig, axs = plt.subplots(1, 2, dpi=600, tight_layout=True)
    fig.set_size_inches(6.5, 2.4)
    batch_bias = -4.5
    
    opt_20b_num_reqs = {}
    for approach in ['baseline', 'baseline-triton', 'naive']:
        opt_20b_num_reqs[map_str[approach]] = []

    for trace_tag in trace_tags:
        trace_type = 'real' if 'real' in trace_tag else 'ondemand'
        suffix = trace_tag.split('-')[0]
        for approach in ['baseline', 'baseline-triton', 'naive']:
            filename =  os.path.join(E2E_RESULT_DIR, f"{trace_type}/{approach}_tpt0.35_cv6-{suffix}/inference_service.log")
            nrequests = request_latency(filename)
            opt_20b_num_reqs[map_str[approach]].append(nrequests)

    if opt_20b_ondemand_avg is not None:
        opt_20b_ondemand_avg += batch_bias
    
    plot_pl(opt_20b_num_reqs, opt_20b_avg, opt_20b_ondemand_avg, axs[1])
    axs[0].set_ylabel('Latency(s)', fontsize=14)
    axs[1].set_title('Average Latency', fontsize=14, y=0.82)

    if opt_20b_ondemand_p99 is not None:
        opt_20b_ondemand_p99 += batch_bias * 2
    od_line = plot_pl(opt_20b_num_reqs, opt_20b_p99, opt_20b_ondemand_p99, axs[0])
    axs[0].set_title('P99 tail Latency', fontsize=14, y=0.82)
    
    linepatches = [
        mlines.Line2D([], [], marker=marker, linestyle='None', markersize=10, label=map_str[approach], color='black')
        for approach, marker in zip(approaches, markers)
    ] + [od_line]
    
    patches = [mpatches.Patch(color=color_palette[i], label=trace_name) for i, trace_name in enumerate(trace_names)]
    
    all_patches = []
    for i in range(len(linepatches)):
        all_patches.append(linepatches[i])
        if i < len(patches):
            all_patches.append(patches[i])
            
    legend = fig.legend(handles=all_patches, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=4, 
                        fontsize=14, frameon=False, columnspacing=0.4, labelspacing=0.3, handletextpad=0.3)
    
    fig.savefig(OUTPUT_FILE, bbox_inches='tight', pad_inches=0.1)
    print('Figure saved to', OUTPUT_FILE)


if __name__ == '__main__':
    verbose = True
    
    opt_20b_avg = {
        '0304-real': [32.045, 26.333, 23.518],
        '0506-real': [27.068, 47.57, 23.315],
        '0304-ondemand': [32.663, 30.373, 23.416],
        '0506-ondemand': [31.915, 24.58, 20.724],
    }
    opt_20b_ondemand_avg = np.array([141.482, 98.037, 29.146, 24.623]) # 3, 4, 6, 8
    
    opt_20b_p99 = {
        '0304-real': [61.287, 51.008, 46.108],
        '0506-real': [56.284, 99.115, 37.083],
        '0304-ondemand': [65.816, 50.732, 49.005],
        '0506-ondemand': [73.257, 43.239, 33.388],
    }
    opt_20b_ondemand_p99 = np.array([253.343, 175.346, 43.47, 34.45]) # 3, 4, 6, 8
    plot_all(opt_20b_avg, opt_20b_ondemand_avg, opt_20b_p99, opt_20b_ondemand_p99)

