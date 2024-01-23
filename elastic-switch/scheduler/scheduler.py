import os
import copy
import configparser
import json
import signal
import time
import subprocess
import numpy as np
from datetime import datetime
from multiprocessing import Process, shared_memory, Lock

import scheduler.commands as sched_cmd
from scheduler.api_server import deploy_api_server
from scheduler.constants import OLD_REPLICAS_NAME, NEW_REPLICAS_NAME, BUFFER_PARAM_NAME, ESTIMATE_COST_NAME
from scheduler.parallel.solver import StrategyOptimizer
from scheduler.utils import encode_tuple, load_shared_mem, dump_shared_mem, clean_shared_mem
from global_server.Switch import Switch
from global_server.TcpThread import TcpThread
from util.util import TcpServer


class Scheduler:
    """Inference job and cluster scheduler.

    The scheduler is responsible for inference job scheduling.
    1. It tracks and controls the inference cluster of nodes.
    2. It schedules inference jobs to the cluster.
    """
    def __init__(self, model_cfg, trace_replayer, checkpoint_path, profile_path, query_trace,
                 query_file, mbs, approach, required_tpt=None, init_strategy=None,
                 min_world_size=1, grace_period=30, ablation=None, batch_requests=True):
        self.model_cfg = model_cfg
        self.trace_replayer = trace_replayer
        self.checkpoint_path = checkpoint_path
        self.profile_path = profile_path
        self.query_trace = query_trace
        self.query_file = query_file
        self.mbs = mbs
        self.approach = approach
        self.required_tpt = required_tpt
        self.init_strategy = init_strategy
        self.min_world_size = min_world_size
        self.grace_period = grace_period
        self.ablation = ablation
        self.batch_requests = batch_requests

        self.is_profiling = (os.environ.get('ENABLE_PROFILER', None) is not None)
        self.read_ini_config()

        self.group_tp = True
        self.strategy_solver = StrategyOptimizer(self.profile_path, self.min_world_size, self.output_seq_len)
        self.inspect_init_strategy()

        self.proc_handlers = {}

        self.lock = Lock()
        if len(self.init_strategy) == 3:
            self.prev_strategy = (*self.init_strategy, self.mbs, 1, 1)
        else:
            self.prev_strategy = tuple(self.init_strategy)
        self.strategy_mem = shared_memory.SharedMemory(name='strategy_mem', create=True, size=20)
        self.strategy_mem.buf[:] = encode_tuple(-1, self.prev_strategy, self.strategy_mem) # init
        self.state_mem = shared_memory.SharedMemory(name='state_mem', create=True, size=1)
        self.state_mem.buf[:] = b'1'
        self.is_initialized = False
        self.switcher = Switch(self.num_layer, self.vocab_size, self.hidden_size, self.max_seq_len)

        self.init_nodes = []
        self.max_world_size = len(self.trace_replayer.get_available_hosts())
        self.num_hist_replica = 0
        self.is_finished = False

    def read_ini_config(self):
        config = configparser.ConfigParser()
        config.read(self.model_cfg)
        model_name = config.get('ft_instance_hyperparameter', 'model_name')
        self.num_layer = config.getint(model_name, 'decoder_layers')
        self.vocab_size = config.getint(model_name, 'vocab_size')
        self.head_num = config.getint(model_name, 'head_num')
        self.head_size = config.getint(model_name, 'size_per_head')
        self.hidden_size = self.head_num * self.head_size
        self.output_seq_len = int(config.get('request', 'request_output_len').split()[0])
        self.max_seq_len = int(config.get('ft_instance_hyperparameter', 'max_seq_len').split()[0])

        # get input seq len
        with open(self.query_trace, 'r') as f:
            for line in f.readlines():
                _, _, req_seq_len, _ = map(float, line.strip().split(','))
                self.input_seq_len = int(req_seq_len)
                break
        if self.required_tpt is None:
            self.query_tpt = float(self.query_trace.split('_')[-1][3:-4])
        else:
            self.query_tpt = self.required_tpt

    def init_timestamp(self):
        self.begin_timestamp = time.time()

    def cur_timestamp(self):
        return time.time() - self.begin_timestamp

    def wait_till(self, timestamp):
        while self.cur_timestamp() < timestamp:
            if 'api_server' in self.proc_handlers and not self.proc_handlers['api_server'].is_alive():
                self.is_finished = True
                return
            self.sleep(0.2)

    def sleep(self, sec):
        time.sleep(sec)

    def __del__(self):
        self.json_out_file.close()
        self.strategy_mem.close()
        self.strategy_mem.unlink()
        self.state_mem.close()
        self.state_mem.unlink()
        self.release_shm('est_cost')

    def release_shm(self, name):
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except:
            pass

    def create_shm(self, name, value):
        if isinstance(value, (int, float)):
            value_str = str(value).encode('utf-8')
        else:
            value_str = json.dumps(value).encode('utf-8')
        self.release_shm(name)
        shm = shared_memory.SharedMemory(name=name, create=True, size=len(value_str))
        shm.buf[:] = value_str

    def deploy_api_server(self):
        server_ip, server_port =  sched_cmd.get_api_server_node()

        xfer_buffer = self.approach == 'naive' and (self.ablation is None or self.ablation in ['cache', 'overlap'])
        profiler_args = (self.profile_path, self.min_world_size, self.output_seq_len)
        self.proc_handlers['api_server'] = Process(
            target=deploy_api_server,
            name='api_server',
            args=(self.query_trace, self.model_cfg, profiler_args, self.approach, server_ip, server_port, self.lock, self.strategy_mem, self.state_mem, xfer_buffer, self.batch_requests),
        )
        self.proc_handlers['api_server'].start()

    def deploy_global_server(self):
        all_hosts = self.trace_replayer.get_available_hosts()

        server_ip, server_port = sched_cmd.get_master_node()
        tcp_server = TcpServer(server_ip, server_port)

        tcp_thread = TcpThread(tcp_server, {})
        tcp_thread.setDaemon(True)
        tcp_thread.start()
        while len(tcp_thread.agent_dict) != self.max_world_size:
            # FIXME: naive way to wait for all agents to connect
            self.sleep(0.05)

        # assign pc_rank to VNodes
        ip_to_start_rank = {}
        for rank, agent in tcp_thread.agent_dict.items():
            if agent.address[0] not in ip_to_start_rank:
                ip_to_start_rank[agent.address[0]] = int(rank)
            else:
                ip_to_start_rank[agent.address[0]] = min(int(rank), ip_to_start_rank[agent.address[0]])
        for vnode in all_hosts:
            vnode.pc_rank = ip_to_start_rank[vnode.ip] + vnode.local_rank
            if vnode.pc_rank < np.prod(self.init_strategy[:3]):
                dp = vnode.pc_rank // (self.init_strategy[1] * self.init_strategy[2])
                tp = vnode.pc_rank % self.init_strategy[1]
                pp = (vnode.pc_rank // self.init_strategy[1]) % self.init_strategy[2]
                vnode.strategy = tuple(self.init_strategy)
                vnode.coordinate = (dp, tp, pp)
                self.init_nodes.append(vnode)

        self.tcp_thread = tcp_thread

    def restart_param_client(self, cur_nodes, restart_strategy):
        # close param client
        print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) begin to restart param clients...', flush=True)
        for agent in self.tcp_thread.agent_dict.values():
            agent.send_string(json.dumps({'job_done': True}))
        self.proc_handlers['param_client'].wait()
        print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) close param clients.', flush=True)

        # restart param client and updata agent_dict
        all_keys = list(self.tcp_thread.agent_dict.keys())
        for k in all_keys:
            del self.tcp_thread.agent_dict[k]
        self.deploy_param_client(cur_nodes=cur_nodes, start_strategy=restart_strategy, clean_log=False)
        while len(self.tcp_thread.agent_dict) != len(cur_nodes):
            self.sleep(0.05)

        # update pc_rank
        ip_to_start_rank = {}
        for rank, agent in self.tcp_thread.agent_dict.items():
            if agent.address[0] not in ip_to_start_rank:
                ip_to_start_rank[agent.address[0]] = int(rank)
            else:
                ip_to_start_rank[agent.address[0]] = min(int(rank), ip_to_start_rank[agent.address[0]])
        self.init_nodes = []
        # get new local_rank map
        node_visible_devs = {}
        for vnode in cur_nodes:
            if vnode.ip not in node_visible_devs:
                node_visible_devs[vnode.ip] = []
            node_visible_devs[vnode.ip].append(vnode.local_rank)
        for ip in node_visible_devs:
            node_visible_devs[ip] = sorted(node_visible_devs[ip])

        for vnode in cur_nodes:
            local_rank = node_visible_devs[vnode.ip].index(vnode.local_rank)
            vnode.pc_rank = ip_to_start_rank[vnode.ip] + local_rank
            if vnode.pc_rank < np.prod(restart_strategy):
                dp = vnode.pc_rank // (restart_strategy[1] * restart_strategy[2])
                tp = vnode.pc_rank % restart_strategy[1]
                pp = (vnode.pc_rank // restart_strategy[1]) % restart_strategy[2]
                vnode.strategy = tuple(restart_strategy)
                vnode.coordinate = (dp, tp, pp)
                self.init_nodes.append(vnode)
            else:
                vnode.prev_strategy = None
                vnode.prev_coordinate = None
                vnode.strategy = None
                vnode.coordinate = None
        print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) restart param clients finished!', flush=True)

    def deploy_param_client(self, cur_nodes=None, start_strategy=None, clean_log=True):
        # first remove all logfiles in logdir
        if clean_log:
            sched_cmd.clean_logfiles()
            self.json_out_file = open(sched_cmd.get_logfile(f'switcher.json'), 'a')
            print(f'[{datetime.now()}] logfiles in {sched_cmd.LOG_PATH}', flush=True)
        if start_strategy is None:
            start_strategy = self.init_strategy
        if cur_nodes is None:
            # experimental: start a param client for all hosts
            cur_nodes = self.trace_replayer.get_available_hosts()

        cmd = sched_cmd.get_param_client_cmd(cur_nodes, self.model_cfg, self.checkpoint_path, start_strategy)

        current_env = os.environ.copy()
        logfn = sched_cmd.get_logfile('param_client')
        out_file = open(logfn, 'a')
        self.proc_handlers['param_client'] = subprocess.Popen(
            cmd, env=current_env, shell=True,
            stdout=out_file, stderr=subprocess.STDOUT
        )

    def close_ft_servers(self, replicas_to_close=None):
        if 'ft_server' not in self.proc_handlers:
            self.proc_handlers['ft_server'] = {}
            self.close_ft_hook_fn = lambda : None
            return

        # print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) begin to close ft servers...')
        # for ft_proc in self.proc_handlers['ft_server']:
        #     if not self.is_profiling:
        #         os.killpg(ft_proc.pid, signal.SIGUSR1)

        ft_procs = []
        if replicas_to_close is None:
            while len(self.proc_handlers['ft_server']) > 0:
                replica_id, ft_proc = self.proc_handlers['ft_server'].popitem()
                ft_procs.append(ft_proc)
        else:
            for replica_id in replicas_to_close:
                ft_proc = self.proc_handlers['ft_server'].pop(replica_id)
                ft_procs.append(ft_proc)

        def fallback_hook():
            for ft_proc in ft_procs:
                ft_proc.wait()

        self.close_ft_hook_fn = fallback_hook

    def start_ft_servers(self, available_nodes, next_strategy, replicas_to_start=None):
        replica, tp, pp, bsz, M1, M2 = next_strategy
        if replicas_to_start is None:
            replicas_to_start = list(range(replica))
        # split available nodes into different replicas
        replica_to_nodes = {i: [] for i in replicas_to_start}
        for node in available_nodes:
            if node.coordinate is not None and node.coordinate[0] in replicas_to_start:
                replica_to_nodes[node.coordinate[0]].append(node)

        # logfn = sched_cmd.get_logfile(f'ft_inference')
        # out_file = open(logfn, 'a')
        need_notify = (self.approach == 'baseline-triton')
        need_block = (self.ablation in ['plain', 'match', 'cache'])
        for replica_id in replicas_to_start:
            server_ip, server_port = sched_cmd.get_api_server_node()
            hosts = replica_to_nodes[replica_id]
            cmd = sched_cmd.get_ft_cmd(hosts, self.model_cfg, replica_id, tp, pp, bsz, M1, M2,
                                       server_ip, server_port, self.query_file, self.profile_path,
                                       need_notify=need_notify, need_block=need_block)

            current_env = os.environ.copy()
            logfn = sched_cmd.get_logfile(f'ft_inference_{self.num_hist_replica}')
            self.num_hist_replica += 1
            out_file = open(logfn, 'a')
            ft_proc = subprocess.Popen(
                cmd, env=current_env, shell=True,
                stdout=out_file, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
            self.proc_handlers['ft_server'][replica_id] = ft_proc

    def triton_transfer(self, prev_nodes, cur_nodes, next_strategy, replicas_to_close, replicas_to_start):
        switch_dict = {}
        for vnode in prev_nodes:
            if vnode.coordinate is not None and vnode.coordinate[0] in replicas_to_close:
                switch_dict[vnode.pc_rank] = self.switcher.node_clean_weight()
                vnode.strategy = None
                vnode.coordinate = None
        new_coordinates = []
        for replica_id in replicas_to_start:
            for pp in range(next_strategy[2]):
                for tp in range(next_strategy[1]):
                    new_coordinates.append((replica_id, tp, pp))

        idle_nodes = []
        for vnode in cur_nodes:
            if vnode.coordinate is None:
                idle_nodes.append(vnode)
        idle_nodes = sorted(idle_nodes, key=lambda x: x.pc_rank)
        for vnode in idle_nodes:
            if len(new_coordinates) <= 0:
                break
            vnode.strategy = tuple(next_strategy[:3])
            vnode.coordinate = new_coordinates.pop(0)
            switch_info = self.switcher.node_load_from_disk(next_strategy, *vnode.coordinate[1:3])
            if vnode.pc_rank in switch_dict:
                switch_dict[vnode.pc_rank].update(switch_info)
            else:
                switch_dict[vnode.pc_rank] = switch_info

        # send switch info to all nodes
        for k in switch_dict:
            self.tcp_thread.agent_dict[str(k)].send_string(json.dumps(switch_dict[k]))

    def estimate_transfer_cost(self, prev_strategy, prev_nodes, next_strategy, next_nodes):
        # init coordinate, switcher will set coordinate for nodes
        for vnode in prev_nodes:
            vnode.prev_strategy = vnode.strategy
            vnode.prev_coordinate = vnode.coordinate
            vnode.strategy = None
            vnode.coordinate = None
        for vnode in next_nodes:
            if vnode.strategy is not None:
                vnode.strategy = None
                vnode.coordinate = None

        buffer_param = {}
        for replica_id in range(prev_strategy[0]):
            buffer_param[replica_id] = {
                'batchxbeam': self.prev_strategy[3],
                'beam': 1,
                'mem_len': self.input_seq_len + self.output_seq_len,
                'session_len': self.input_seq_len + self.output_seq_len,
                'new_layers_per_device': self.num_layer // next_strategy[2],
            }
        replicas_stay = list(range(next_strategy[0]))
        # set default values
        kwargs = {
            'buffer_param': buffer_param,
            'memory_tolarance': 0,
            'group_tp': self.group_tp,
        }
        if self.approach == 'naive':
            if self.ablation == 'plain':
                kwargs['buffer_param'] = None
                kwargs['km_match'] = False
                kwargs['group_tp'] = False
                kwargs['memory_tolarance'] = 10
                kwargs['enable_no_lock'] = False
                kwargs['try_dont_stop_ft'] = False
                kwargs['slpt'] = 0
            elif self.ablation == 'match':
                kwargs['buffer_param'] = None
                kwargs['memory_tolarance'] = 10
                kwargs['enable_no_lock'] = False
                kwargs['try_dont_stop_ft'] = False
                kwargs['slpt'] = 0
            elif self.ablation == 'cache':
                kwargs['memory_tolarance'] = 10
                kwargs['enable_no_lock'] = False
                kwargs['try_dont_stop_ft'] = False
                kwargs['slpt'] = 0
        switch_dict = self.switcher.do_switch(prev_strategy, prev_nodes, next_strategy, next_nodes, replicas_stay,
                                              verbose=False, json_out_file=self.json_out_file, **kwargs)
        transfer_cost = self.switcher.estimate_last_switch_transfer_time()
        # transfer cost + overhead time of switching strategy
        return transfer_cost + 0.5

    def switch_strategy(self, prev_nodes, cur_nodes, next_strategy, preempted_pipelines, ddl=None):
        # Naive way to switch strategy: first close old ft servers
        if self.approach == 'baseline-triton':
            replicas_to_close = list(preempted_pipelines)
            num_new_pipelines = next_strategy[0] - (self.prev_strategy[0] - len(preempted_pipelines))
            replicas_to_start = list(range(self.num_hist_replica, self.num_hist_replica + num_new_pipelines))
        else:
            replicas_to_close = None
            replicas_to_start = None
            if self.approach == 'naive':
                clean_shared_mem(OLD_REPLICAS_NAME)
                clean_shared_mem(NEW_REPLICAS_NAME)

        # FIXME: suppose replay ends when len(cur_nodes) == 0
        if next_strategy[0] == 0 and next_strategy[1] == 0 and next_strategy[2] == 0:
            est_cost = 0.05
            dump_shared_mem(est_cost, ESTIMATE_COST_NAME)

            if self.approach == 'baseline-triton':
                dump_shared_mem(replicas_to_close, OLD_REPLICAS_NAME)
                dump_shared_mem(replicas_to_start, NEW_REPLICAS_NAME)

            # set strategy to shared memory
            with self.lock:
                self.strategy_mem.buf[:] = encode_tuple(1, next_strategy, self.strategy_mem)
                print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) set new strategy {next_strategy} to shared memory, '
                      f'estimate transfer cost: {est_cost:.3f}', flush=True)

            # wait for previous finish
            while True:
                with self.lock:
                    state = bytes(self.state_mem.buf[:]).decode()
                    if state == '1':
                        self.state_mem.buf[:] = b'0'
                        print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) previous strategy finished', flush=True)
                        break
                self.sleep(0.01)

            for agent in self.tcp_thread.agent_dict.values():
                agent.send_string(json.dumps({'job_done': True}))
            print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) send "job_done" signal to param clients.', flush=True)
            return

        work_prev_nodes = sorted([vnode.id for vnode in prev_nodes if vnode.strategy is not None])
        work_cur_nodes = sorted([vnode.id for vnode in cur_nodes if vnode.strategy is not None])
        prev_hash = ','.join([f'{tuple(self.prev_strategy)}'] + [str(vid) for vid in work_prev_nodes])
        next_hash = ','.join([f'{tuple(next_strategy)}'] + [str(vid) for vid in work_cur_nodes])
        if prev_hash != next_hash:
            print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) switching from {self.prev_strategy} to {next_strategy}', flush=True)

            # pure parallelization strategy
            pure_prev_strategy = tuple(self.prev_strategy[:3])
            pure_next_strategy = tuple(next_strategy[:3])

            # estimates switch transfer cost
            cp_node_map = {}
            prev_nodes_cp = copy.deepcopy(prev_nodes)
            for vnode_cp in prev_nodes_cp:
                cp_node_map[vnode_cp.id] = vnode_cp
            cur_nodes_cp = []
            for vnode in cur_nodes:
                cur_nodes_cp.append(cp_node_map[vnode.id] if vnode.id in cp_node_map else copy.deepcopy(vnode))
            if self.approach in ['baseline', 'baseline-triton']:
                est_cost = 0.5
                select_ft_start = False
            else:
                est_cost = self.estimate_transfer_cost(pure_prev_strategy, prev_nodes_cp, pure_next_strategy, cur_nodes_cp)
                select_ft_start = self.switcher.dont_stop_ft
            if ddl is not None:
                print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) find ddl, original estimate transfer cost: {est_cost:.3f}', flush=True)
                est_cost = est_cost + max(self.grace_period - (ddl - self.cur_timestamp()), 0)
            if len(cur_nodes) > len(prev_nodes):
                est_cost = -est_cost
            dump_shared_mem(est_cost, ESTIMATE_COST_NAME)

            if self.approach == 'baseline-triton':
                dump_shared_mem(replicas_to_close, OLD_REPLICAS_NAME)
                dump_shared_mem(replicas_to_start, NEW_REPLICAS_NAME)
            elif self.approach == 'naive' and select_ft_start:
                replicas_to_close = []
                replicas_to_start = list(range(self.prev_strategy[0], next_strategy[0]))
                dump_shared_mem(replicas_to_close, OLD_REPLICAS_NAME)
                dump_shared_mem(replicas_to_start, NEW_REPLICAS_NAME)
                print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) select ft start: {replicas_to_start}', flush=True)

            # FIX: wait for new instances arrive
            if self.approach == 'baseline-triton' and len(replicas_to_close) == 0 and len(replicas_to_start) > 0:
                self.sleep(self.grace_period)
            elif self.approach == 'naive' and select_ft_start:
                self.sleep(self.grace_period)

            self.close_ft_servers(replicas_to_close)

            # set strategy to shared memory
            with self.lock:
                self.strategy_mem.buf[:] = encode_tuple(1, next_strategy, self.strategy_mem)
                print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) set new strategy {next_strategy} to shared memory, '
                      f'estimate transfer cost: {est_cost:.3f}', flush=True)

            # wait for previous finish
            while True:
                with self.lock:
                    state = bytes(self.state_mem.buf[:]).decode()
                    if state == '1':
                        self.state_mem.buf[:] = b'0'
                        print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) previous strategy finished', flush=True)
                        break
                self.sleep(0.01)

            # set prev and new coordinate
            if self.approach == 'baseline':
                self.close_ft_hook_fn()
                self.restart_param_client(cur_nodes, pure_next_strategy)
                prev_nodes = list(self.init_nodes)
                self.close_ft_hook_fn = lambda: None
                # start new ft servers
                self.start_ft_servers(cur_nodes, next_strategy)
                self.is_initialized = True
                return
            elif self.approach == 'baseline-triton':
                self.close_ft_hook_fn()
                # send signals to PC
                if self.is_initialized:
                    self.triton_transfer(prev_nodes, cur_nodes, next_strategy, replicas_to_close, replicas_to_start)
                self.start_ft_servers(cur_nodes, next_strategy, replicas_to_start=replicas_to_start)
                self.is_initialized = True
                return

            # init coordinate, switcher will set coordinate for nodes
            for vnode in prev_nodes:
                vnode.prev_strategy = vnode.strategy
                vnode.prev_coordinate = vnode.coordinate
                vnode.strategy = None
                vnode.coordinate = None
            for vnode in cur_nodes:
                if vnode.strategy is not None:
                    vnode.strategy = None
                    vnode.coordinate = None

            # must wait for signal from self.state_mem
            if self.is_initialized:
                replica_to_batch = load_shared_mem(BUFFER_PARAM_NAME)
                buffer_param = {} if len(replica_to_batch) > 0 else None
                for replica_id, (batch, seq_len, start_step) in replica_to_batch.items():
                    replica_id = int(replica_id)

                    buffer_param[replica_id] = {
                        'batchxbeam': batch,
                        'beam': 1,
                        'mem_len': seq_len + self.output_seq_len,
                        'session_len': seq_len + self.output_seq_len,
                        'new_layers_per_device': self.num_layer // next_strategy[2],
                    }
                # first fill in empty positions
                if len(replica_to_batch) > 0:
                    for replica_id in range(self.prev_strategy[0]):
                        if replica_id not in buffer_param:
                            buffer_param[replica_id] = None

                if len(replica_to_batch) <= next_strategy[0]:
                    replicas_stay = list(map(int, replica_to_batch.keys()))
                    for replica_id in range(next_strategy[0]):
                        if len(replicas_stay) == next_strategy[0]:
                            break
                        if replica_id not in replicas_stay:
                            replicas_stay.append(replica_id)
                    replicas_stay = sorted(replicas_stay)
                else:
                    replicas_stay = list(map(int, sorted(replica_to_batch.keys(), key=lambda x: replica_to_batch[x][2], reverse=True)))
                    replicas_stay = sorted(replicas_stay[:next_strategy[0]])
            else:
                buffer_param = None
                replicas_stay = list(range(next_strategy[0]))

            print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) replicas_stay: {replicas_stay}', flush=True)
            print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) buffer_param: {buffer_param}', flush=True)
            # set default values
            kwargs = {
                'buffer_param': buffer_param,
                'memory_tolarance': 0,
                'group_tp': self.group_tp,
            }
            if self.approach == 'naive':
                if self.ablation == 'plain':
                    kwargs['buffer_param'] = None
                    kwargs['km_match'] = False
                    kwargs['group_tp'] = False
                    kwargs['memory_tolarance'] = 10
                    kwargs['enable_no_lock'] = False
                    kwargs['try_dont_stop_ft'] = False
                    kwargs['slpt'] = 0
                elif self.ablation == 'match':
                    kwargs['buffer_param'] = None
                    kwargs['memory_tolarance'] = 10
                    kwargs['enable_no_lock'] = False
                    kwargs['try_dont_stop_ft'] = False
                    kwargs['slpt'] = 0
                elif self.ablation == 'cache':
                    kwargs['memory_tolarance'] = 10
                    kwargs['enable_no_lock'] = False
                    kwargs['try_dont_stop_ft'] = False
                    kwargs['slpt'] = 0
            switch_dict = self.switcher.do_switch(
                pure_prev_strategy, prev_nodes, pure_next_strategy, cur_nodes, replicas_stay,
                json_out_file=self.json_out_file, verbose=False, **kwargs
            )

            for k in switch_dict:
                self.tcp_thread.agent_dict[k].send_string(json.dumps(switch_dict[k]))

            # start new ft servers
            self.start_ft_servers(cur_nodes, next_strategy, replicas_to_start=replicas_to_start)

            print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) Wait for FT proc exit.', flush=True)
            self.close_ft_hook_fn()
        elif not self.is_initialized:
            self.close_ft_servers()

            # set strategy to shared memory
            with self.lock:
                self.strategy_mem.buf[:] = encode_tuple(1, next_strategy, self.strategy_mem)
                print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) **init** set new strategy {next_strategy} to shared memory', flush=True)
                self.state_mem.buf[:] = b'0'

            # start new ft servers
            self.start_ft_servers(cur_nodes, next_strategy)

        self.is_initialized = True

    def inspect_init_strategy(self):
        if self.init_strategy is None:
            tstamp, operation, event_info = self.trace_replayer[0]
            if self.approach == 'baseline-triton':
                nnodes = len(self.trace_replayer.get_available_hosts())
                # for triton baseline, we will only change dp!
                init_strategy = list(self.strategy_solver.solve(None, nnodes, self.mbs, self.query_tpt))
                # init_strategy = (1, 8, 2, 4, 4, 1, 0, 0) # TODO: hard code for now
                num_dp = len(event_info['nodes']) // (init_strategy[1] * init_strategy[2])
                self.init_strategy = (num_dp, *init_strategy[1:-2])
            else:
                init_strategy = self.strategy_solver.solve(None, len(event_info['nodes']), self.mbs, self.query_tpt)
                self.init_strategy = (*init_strategy[:3], self.mbs, *init_strategy[4:-2])
                # init_strategy = (1, 1, 2, 4, 2, 1, 0, 0) # TODO: hard code for now
                # num_dp = len(event_info['nodes']) // (init_strategy[1] * init_strategy[2])
                # self.init_strategy = (num_dp, *init_strategy[1:-2])

    def run(self):
        # 1. start a param client for each host to hold checkpoints
        self.deploy_param_client()
        # 2. start a global server to manage the param client
        self.deploy_global_server()
        # 3. start an api server to serve the inference requests
        self.deploy_api_server()

        # replay the trace
        self.init_timestamp()
        print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) system initialized.', flush=True)

        last_event_id = -1
        available_nodes = set()
        for event_id, event in enumerate(self.trace_replayer):
            if event_id <= last_event_id:
                continue
            
            if self.approach == 'naive':
                # NOTE: disable merge two events, seems no need
                tstamp, operation, event_info = event
                nxt_event = self.trace_replayer[event_id + 1] if event_id + 1 < len(self.trace_replayer) else None
                if nxt_event is not None:
                    nxt_tstamp, nxt_operation, nxt_event_info = nxt_event
                    if operation == 'add' and nxt_operation == 'remove' and tstamp + 30_000 >= nxt_tstamp:
                        # merge two event
                        event = (tstamp, 'MERGE', {'add': event_info['nodes'], 'remove': nxt_event_info['nodes']})
                        last_event_id = event_id + 1

            # update event
            tstamp, operation, event_info = event
            print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) next-event: {operation} at {tstamp / 1000:.3f}, waiting...', flush=True)
            noti_tstamp = tstamp / 1000
            # if operation in ['remove', 'DONE']:
            noti_tstamp -= self.grace_period
            self.wait_till(noti_tstamp)

            preempted_pipelines = set()
            if self.is_finished:
                self.switch_strategy(None, None, (0, 0, 0, 0, 0, 0), preempted_pipelines)
                break

            if event_id > 0:
                prev_nodes = []
                for vnode in available_nodes:
                    if vnode.strategy is not None:
                        prev_nodes.append(vnode)
            else:
                prev_nodes = list(self.init_nodes)

            if operation == 'add':
                for vnode in event_info['nodes']:
                    # init a vnode
                    if vnode not in prev_nodes:
                        vnode.prev_strategy = None
                        vnode.prev_coordinate = None
                        vnode.strategy = None
                        vnode.coordinate = None

                    available_nodes.add(vnode)
                ddl = None
            elif operation in ['remove', 'DONE']:
                for vnode in event_info['nodes']:
                    available_nodes.remove(vnode)
                    if vnode.coordinate is not None:
                        preempted_pipelines.add(vnode.coordinate[0])
                ddl = tstamp/1000
            elif operation == 'MERGE':
                for vnode in event_info['add']:
                    # init a vnode
                    if vnode not in prev_nodes:
                        vnode.prev_strategy = None
                        vnode.prev_coordinate = None
                        vnode.strategy = None
                        vnode.coordinate = None

                    available_nodes.add(vnode)
                for vnode in event_info['remove']:
                    available_nodes.remove(vnode)
                    if vnode.coordinate is not None:
                        preempted_pipelines.add(vnode.coordinate[0])
                ddl = tstamp/1000

            cur_nnodes = len(available_nodes)
            if 'required_tpt' in event_info:
                self.query_tpt = event_info['required_tpt']
                ddl = None

            # strategy format: (dp, tp, pp, bsz, M1, M2, latency, tpt)
            if self.approach == 'baseline-triton' or self.ablation is not None:
                num_dp = cur_nnodes // (self.init_strategy[1] * self.init_strategy[2])
                next_strategy = (num_dp, *self.init_strategy[1:])
            else:
                next_strategy = self.strategy_solver.solve(self.prev_strategy, cur_nnodes, self.mbs, self.query_tpt)
                # strategy format: (dp, tp, pp, bsz, M1, M2)
                next_strategy = (*next_strategy[:3], self.mbs, *next_strategy[4:-2])

            if operation == 'DONE' or cur_nnodes == 0:
                next_strategy = (0, 0, 0, 0, 0, 0)

            print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) cur_nnodes: {cur_nnodes}, next_strategy: {next_strategy}, requited_tpt={self.query_tpt}', flush=True)

            self.switch_strategy(prev_nodes, list(available_nodes), next_strategy, preempted_pipelines, ddl=ddl)
            self.prev_strategy = tuple(next_strategy)

        print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) replay finished, wait for param client to exit.', flush=True)
        # closes all processes
        self.proc_handlers['param_client'].wait()
        print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) scheduler exits.', flush=True)
