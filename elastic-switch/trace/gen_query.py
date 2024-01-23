import argparse
import csv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str)
    parser.add_argument('--seq-len', nargs='+', type=int, default=128)
    parser.add_argument('--n', type=int, default=256)
    parser.add_argument('--vocab-size', type=int, default=50304)
    parser.add_argument('--tpt', type=float, default=2.5)
    parser.add_argument('--cv', type=float, default=1)
    parser.add_argument('--trace', type=str, default=None)
    parser.add_argument('--dist', type=str, default='gamma', choices=['gamma', 'possion', 'uniform'])
    parser.add_argument('--gen-query', action='store_true')
    return parser.parse_args()


def generate_queries(outfile, seq_len_list, n, vocab_size):
    with open(outfile, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for seq_len in seq_len_list:
            for i in range(n):
                seq = np.random.randint(0, vocab_size, seq_len)
                csvwriter.writerow(seq)


def main(query_file, num_query, arrival_rate, trace_file, cv, dist='possion'):
    queries = []
    query_id_to_len = {}
    with open(query_file, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')
        for i, row in enumerate(csvreader):
            seq = list(map(lambda x: int(x.strip()), row))
            query_id_to_len[i] = len(seq)
            queries.append(seq)

            if len(queries) > num_query:
                break

    if dist == 'possion':
        x = np.random.exponential(1 / arrival_rate, len(queries))
    elif dist == 'gamma':
        k = (1000 / arrival_rate) / cv
        x = np.random.gamma(k, cv, len(queries))
        x = x / 1000
        print(f'gamma mean arrival rate: {1 / np.mean(x):.3f}, std: {np.std(x):.3f}')
    elif dist == 'uniform':
        x = [0] + [1 / arrival_rate] * (len(queries) - 1)

    # calculate offset
    query_id_to_offset = {}
    with open(query_file, 'r') as f:
        offset = 0
        for i, line in enumerate(f.readlines()):
            query_id_to_offset[i] = offset
            offset += len(line) + 1

    # with open(query_file, 'w') as f:
    #     csvwriter = csv.writer(f, delimiter=',')
    #     for i, seq in enumerate(data):
    #         csvwriter.writerow(seq)

    with open(trace_file, 'w') as f:
        tstamp = 0
        for i, t in enumerate(x):
            tstamp += t
            f.write(f'{i},{tstamp},{query_id_to_len[i]},{query_id_to_offset[i]}\n')

if __name__ == '__main__':
    args = parse_args()
    if args.gen_query:
        generate_queries(args.query, args.seq_len, args.n, args.vocab_size)

    main(args.query, args.n, args.tpt, args.trace, args.cv, args.dist)
