# Deprecated file
import numpy as np

def parse_log(filename):
    req_lat = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            if 'latency' in line:
                seq_len = int(line.split()[-1])
                comp_latency = float(line.split()[-5][:-1])

                if seq_len not in req_lat:
                    req_lat[seq_len] = []
                req_lat[seq_len].append(comp_latency)

    print(f'seq_len | avg_latency (ms)')
    for seq_len in req_lat:
        lats = req_lat[seq_len]
        lats = sorted(lats)
        outlier = int(len(lats) * 0.1) // 2
        p90_lats = lats[outlier:-outlier]
        avg_lat = np.mean(p90_lats)
        print(f'{seq_len}, {avg_lat:.3f}')


def main_seqlen_test(logdir):
    for out in [32, 64, 128, 256, 512]:
        filename = f'{logdir}_out{out}/inference_service.log'
        print(f'out seqlen: {out}, logfile: {filename}')
        parse_log(filename)


if __name__ == '__main__':
    logdir = 'log/6.7b/naive_tpt8'
    main_seqlen_test(logdir)
