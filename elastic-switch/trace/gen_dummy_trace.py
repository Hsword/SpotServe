import json
import numpy as np


def gen(nnodes=16, interval=120, nevent=20):
    node_trace = np.random.randint(1, nnodes, size=nevent)
    interval = interval * 1000 # convert to ms

    fp = open('trace/test_tmp.txt', 'w')
    cur_nodes = []
    prev_nnode = 0
    node_id = 0
    for idx, node_num in enumerate(node_trace):
        nodes_chg = []
        if node_num > prev_nnode:
            for _ in range(node_num - prev_nnode):
                nodes_chg.append(f'node-{node_id}')
                cur_nodes.append(node_id)
                node_id += 1
            event = [idx * interval, 'add', {'nodes': nodes_chg}]
        elif node_num < prev_nnode:
            for _ in range(prev_nnode - node_num):
                nodes_chg.append(f'node-{cur_nodes.pop(0)}')
            event = [idx * interval, 'remove', {'nodes': nodes_chg}]
        else:
            continue
        prev_nnode = node_num
        fp.write(f'{json.dumps(event)}\n')

    event = [(idx + 1) * interval, 'DONE', {'nodes': []}]
    fp.write(f'{json.dumps(event)}\n')

if __name__ == '__main__':
    gen()
