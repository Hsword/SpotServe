import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduler.parallel.solver import StrategyOptimizer


class TraceGen:
    def __init__(self, trace_file, opt):
        self.trace_file = trace_file
        self.opt = opt

        self.trace_events = []
        cur_nodes = []
        with open(self.trace_file, 'r') as f:
            last_tstamp = None
            for line in f.readlines():
                tstamp, operation, node_id = line.strip().split(',')
                tstamp = int(tstamp)
                if last_tstamp is None:
                    last_tstamp = tstamp

                if tstamp != last_tstamp:
                    self.trace_events.append((last_tstamp, len(cur_nodes), tuple(cur_nodes)))
                    last_tstamp = tstamp

                if operation == 'add':
                    cur_nodes.append(node_id)
                else:
                    cur_nodes.remove(node_id)
            if len(cur_nodes) > 0:
                self.trace_events.append((tstamp, len(cur_nodes), tuple(cur_nodes)))

    def optimize(self, max_bsz, min_tpt):
        print(f'Time(min),#Nodes,DP,TP,PP,BS,M1,M2,lat,TPT')
        for i, (tstamp, num_nodes, nodes) in enumerate(self.trace_events):
            cur_minutes = tstamp / 1000 / 60
            strategy = self.opt.solve(None, num_nodes * 4, max_bsz, min_tpt)

            print(f'{cur_minutes:.1f},{num_nodes},{",".join(map(str, strategy))}')

    def split_trace(self, start, end):
        # convert to ms
        start = int(start * 60 * 1000)
        end = int(end * 60 * 1000)

        new_trace = []
        node_map, node_id = {}, 0
        start_tstamp = None
        for tstamp, num_nodes, e_nodes in self.trace_events:
            if tstamp >= start and tstamp <= end:
                if start_tstamp is None:
                    start_tstamp = tstamp
                nodes = []
                for vnode in e_nodes:
                    if vnode not in node_map:
                        node_map[vnode] = f'node-{node_id}'
                        node_id += 1
                    nodes.append(node_map[vnode])
                new_trace.append((tstamp, num_nodes, nodes))

        last_nnodes, last_nodes = 0, []
        for event in new_trace:
            tstamp, num_nodes, cur_nodes = event
            node_chg = []
            if num_nodes > last_nnodes:
                for node in cur_nodes:
                    if node not in last_nodes:
                        node_chg.append(node)
                operation = 'add'
            else:
                for node in last_nodes:
                    if node not in cur_nodes:
                        node_chg.append(node)
                operation = 'remove'
            last_nnodes, last_nodes = num_nodes, cur_nodes
            tstamp = tstamp - start_tstamp
            new_event = [tstamp, operation, {"nodes": node_chg}]
            print(f'{json.dumps(new_event)}')
        # final event last for 1 minutes
        final_event = [new_trace[-1][0] - start_tstamp + 60 * 1000, 'DONE', {"nodes": []}]
        print(f'{json.dumps(final_event)}')


if __name__ == '__main__':
    model_cfgs = {
        '6.7B': ('profile/T4-4x/megatron_6.7B_profile.json', 4, 0, 16),
        # 20B
        'h6144': ('profile/T4-4x/megatron_h6144_profile.json', 12, 1.2, 16),
        # 30B
        'h7168': ('profile/T4-4x/megatron_h7168_profile.json', 24, 0.8, 8),
    }
    model = '6.7B'
    trace_file = 'trace/trace_full/g4dn.csv'
    profile_path, min_world_size, min_tpt, max_tp = model_cfgs[model]
    output_seq_len = 128
    max_bsz = 8

    opt = StrategyOptimizer(profile_path, min_world_size, output_seq_len, max_tp=max_tp)
    generator = TraceGen(trace_file, opt)
    generator.optimize(max_bsz, min_tpt)

    # split trace
    generator.split_trace(313, 336)
