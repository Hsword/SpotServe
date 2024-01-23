import argparse
import csv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arrival', type=str)
    parser.add_argument('--query', type=str)
    parser.add_argument('--seq-len', nargs='+', type=int, default=128)
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--vocab-size', type=int, default=50304)
    parser.add_argument('--cv', type=float, default=1)
    parser.add_argument('--trace', type=str, default=None)
    parser.add_argument('--dist', type=str, default='gamma', choices=['gamma'])
    parser.add_argument('--gen-query', action='store_true')
    return parser.parse_args()


def generate_queries(outfile, seq_len_list, n, vocab_size):
    with open(outfile, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for seq_len in seq_len_list:
            for i in range(n):
                seq = np.random.randint(0, vocab_size, seq_len)
                csvwriter.writerow(seq)


def main(query_file, num_query, arrival_rates, trace_file, cv, dist='possion'):
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
    
    enum_time = 0
    enum_x = []
    
    for cur_t, arrival_rate in enumerate(arrival_rates):
        assert dist == 'gamma'
        k = (1000 / arrival_rate) / cv
        x = np.random.gamma(k, cv, int(120 * arrival_rate))
        x = x / 1000
        for interval in x:
            if enum_time >= (cur_t + 1) * 60:
                break
            enum_x.append(interval)
            enum_time += interval
    
    assert len(enum_x) < num_query, f'len({len(enum_x)}) shoue be less than n({num_query})'
    x = enum_x

    # calculate offset
    query_id_to_offset = {}
    with open(query_file, 'r') as f:
        offset = 0
        for i, line in enumerate(f.readlines()):
            query_id_to_offset[i] = offset
            offset += len(line) + 1

    with open(trace_file, 'w') as f:
        tstamp = 0
        for i, t in enumerate(x):
            tstamp += t
            f.write(f'{i},{tstamp},{query_id_to_len[i]},{query_id_to_offset[i]}\n')

if __name__ == '__main__':
    args = parse_args()
    
    with open(args.arrival, 'r') as f:
        arrival_rates = list(map(float, f.readline().strip().split(',')))
    print(f'read {len(arrival_rates)} arrival rates.')
    
    if args.gen_query:
        generate_queries(args.query, args.seq_len, args.n, args.vocab_size)

    main(args.query, args.n, arrival_rates, args.trace, args.cv, args.dist)
    
