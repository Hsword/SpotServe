from hashlib import new
import sys
import json

from util.util import TcpServer, TcpAgent, timestamp
from TcpThread import TcpThread
from scheduler.trace_replayer import VNode
from global_server.Switch import Switch

GLOBAL_SERVER_PORT = 10040

agent_dict = {}

def main():
    tcp_server = TcpServer('localhost', GLOBAL_SERVER_PORT)

    tcp_thread = TcpThread(tcp_server, agent_dict)
    tcp_thread.setDaemon(True)
    tcp_thread.start()
    
    dp = 1 if len(sys.argv) <= 1 else int(sys.argv[1])
    tp = 1 if len(sys.argv) <= 2 else int(sys.argv[2])
    pp = 1 if len(sys.argv) <= 3 else int(sys.argv[3])
    
    strat = dp, tp, pp
    print('initial strat', strat)
    
    swicher = Switch(24)
    world_size = 16
    ip = '127.0.0.1'
    
    vnodes = [VNode(ip, i) for i in range(world_size)]
    for vnode in vnodes:
        vnode.pc_rank = vnode.gpu_id
    old_nodes = []
    
    ws_old = strat[0] * strat[1] * strat[2]
    for i in range(ws_old):
        vnodes[i].prev_strategy = strat
        vnodes[i].prev_coordinate = i // (strat[1] * strat[2]), i % strat[1], (i // strat[1]) % strat[2]
        old_nodes.append(vnodes[i])
        print(vnodes[i], vnodes[i].prev_coordinate)
    
    while True:
        s = input()
        if s == 'quit':
            break

        new_s = tuple(map(int, s.split()))
        timestamp('gs', f'switching from {strat} to {new_s}')
        ws_new = new_s[0] * new_s[1] * new_s[2]

        switch_buffer =  len(new_s) > 3
        buffer_dicts = None
        
        if switch_buffer:
            ## test value
            buffer_dict = {
                'batchxbeam': 8,
                'beam' : 1,         # batch = batchxbeam // beam
                'mem_len': 40,      # = input_len + output_len
                'session_len': 40,  # = mem_len
                'new_layers_per_device': swicher.layer_num // new_s[2]
            }
            buffer_dicts = [buffer_dict] * strat[0]
            
        
        # disable km match for debugging PC
        switch_dict = swicher.do_switch(strat, vnodes[:ws_old], new_s[:3], vnodes[:ws_new], list(range(new_s[0])), buffer_dicts, km_match=False, verbose=True, memory_tolarance=0.2)

        for k in switch_dict.keys():
            agent_dict[k].send_string(json.dumps(switch_dict[k]))

        strat = new_s
        ws_old = ws_new
        
        for vnode in vnodes[:ws_old]:
            vnode.prev_strategy = vnode.strategy
            vnode.prev_coordinate = vnode.coordinate
            vnode.strategy = None
            vnode.coordinate = None
    
if __name__ == '__main__':
    main()