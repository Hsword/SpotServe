import asyncio
import bisect
import configparser
import json
import struct
import time
import numpy as np
from datetime import datetime
from multiprocessing import shared_memory

import scheduler.commands as sched_cmd
from scheduler.constants import OLD_REPLICAS_NAME, NEW_REPLICAS_NAME, BUFFER_PARAM_NAME, ESTIMATE_COST_NAME
from scheduler.parallel.solver import StrategyOptimizer
from scheduler.utils import encode_tuple, decode_tuple, dump_shared_mem, load_shared_mem
from util.util import TcpServer, TcpAgent


class Request:
    def __init__(self, query_id, tstamp, seq_len, query_offset):
        self.query_id = query_id
        self.tstamp = tstamp
        self.seq_len = seq_len
        self.query_offset = query_offset
        self.start_step = 0

        # latency info, in seconds time stamp
        self.submit = self.tstamp # when this request is created
        self.end = None # when this request is completed

        # latency breakdown in ms
        self.schedule_lat = None
        self.inference_lat = None
        self.e2e_lat = None

    def __repr__(self):
        return f'Query({self.query_id})'

    def __lt__(self, other):
        return self.tstamp < other.tstamp

    def info(self):
        return (self.query_id, self.tstamp, self.seq_len, self.query_offset)

    def set_latency(self, end_tstamp, schedule_lat, inference_lat):
        self.end = end_tstamp
        if self.schedule_lat is not None:
            self.schedule_lat += schedule_lat
        else:
            self.schedule_lat = schedule_lat
        if self.inference_lat is not None:
            self.inference_lat += inference_lat
        else:
            self.inference_lat = inference_lat
        self.e2e_lat = (self.end - self.submit) * 1000

    def tcp_overhead(self):
        return self.e2e_lat - self.schedule_lat - self.inference_lat


# apireplcia state enum class
class ReplicaState:
    INIT = 0
    READY = 1
    DONE = 2


class APIReplica:
    id = 0

    def __init__(self, replica_id, strategy, approach):
        self.replica_id = replica_id
        self.strategy = strategy
        self.approach = approach

        self.request_batch_size = strategy[3]
        self.M1 = strategy[4]
        self.M2 = strategy[5]
        self.div_bs = self.request_batch_size // self.M1

        self.request_agent = None
        self.request_rank = None
        self.response_agent = None
        self.response_rank = None

        if approach == 'baseline-triton' or approach == 'naive':
            # disable currently
            self.state = ReplicaState.INIT
        else:
            self.state = ReplicaState.READY

        self.id = APIReplica.id
        APIReplica.id += 1

        # track requests status
        self.last_resp_tstamp = 0
        self.past_latencies = []
        self.request_pool_size = 0
        self.query_onflight = set()
        self.interrupt_queries = []

    def is_ready(self):
        if self.request_agent is None or self.response_agent is None:
            return False
        return self.state == ReplicaState.READY

    def get_min_bsz(self, pool_size):
        # new batching
        min_bsz = 1
        while min_bsz * 2 <= pool_size and min_bsz * 2 <= self.request_batch_size:
            min_bsz *= 2
        # old batching
        # if self.request_pool_size == 0 or self.request_pool_size < self.request_batch_size:
        #     min_bsz = 1
        #     while min_bsz * 2 <= pool_size and min_bsz * 2 <= self.request_batch_size:
        #         min_bsz *= 2
        # else:
        #     remain_bs = min(0, self.request_pool_size - self.request_batch_size)
        #     min_bsz = 1
        #     while min_bsz * 2 <= pool_size + remain_bs and min_bsz * 2 <= self.request_batch_size:
        #         min_bsz *= 2
        #     min_bsz = min(min_bsz - remain_bs, self.request_batch_size)
        return min_bsz

    def push_record(self, request):
        self.last_resp_tstamp = request.end
        self.past_latencies.append((request.inference_lat, request.e2e_lat - request.inference_lat))

    def get_avg_latency(self, n=10, comp_only=False):
        comp_lats = [lat[0] for lat in self.past_latencies[-n:]]
        sched_lats = [lat[1] for lat in self.past_latencies[-n:]]
        lat = np.mean(comp_lats)
        if not comp_only:
            lat += np.mean(sched_lats)
        return lat


class APIServer:
    def __init__(self, query_trace, model_cfg, profiler_args, approach, server_ip, server_port,
                lock, strategy_mem, state_mem, xfer_buffer=True, batch_requests=True):
        self.query_trace = query_trace
        self.model_cfg = model_cfg
        self.profiler_args = profiler_args
        self.approach = approach
        self.server_port = server_port
        self.lock = lock
        self.strategy_mem = strategy_mem
        self.state_mem = state_mem
        self.xfer_buffer = xfer_buffer

        # read model config
        self.read_ini_config()

        self.batch_requests = batch_requests
        print(f'[{datetime.now()}] batch_requests: {self.batch_requests}', flush=True)
        self.queries = self.read_query_trace()
        self.query_id_to_request = {query.query_id: query for query in self.queries}
        self.interrupt_queries = {} # Dict(replica_id: List[query_id])
        self.query_onflight = set()

        self.logfile = sched_cmd.get_logfile('inference_service')
        self.logfn = open(self.logfile, 'a')

        self.req_tcp_server = TcpServer(server_ip, server_port)
        self.resp_tcp_server = TcpServer(server_ip, server_port + 1)

        self.api_agent = {} # Dict(replica_id: APIReplica)
        self.tcp_agent_lock = asyncio.Lock()
        self.tcp_agent_initialized = False

        self.cur_strategy = None
        self.is_timestamp_initialized = False
        self.is_finished = False

        self.strategy_solver = StrategyOptimizer(*self.profiler_args)

    def read_ini_config(self):
        config = configparser.ConfigParser()
        config.read(self.model_cfg)
        model_name = config.get('ft_instance_hyperparameter', 'model_name')
        self.num_layer = config.getint(model_name, 'decoder_layers')

        self.request_batch_size = int(config.get('request', 'request_batch_size').split()[0])
        self.output_seq_len = int(config.get('request', 'request_output_len').split()[0])

    def read_query_trace(self):
        queries = []
        with open(self.query_trace, 'r') as f:
            for line in f.readlines():
                query_id, tstamp, seq_len, query_ofset = map(float, line.strip().split(','))
                queries.append(Request(int(query_id), tstamp, int(seq_len), int(query_ofset)))
        queries = sorted(queries, key=lambda x: x.tstamp)
        return queries

    def init_timestamp(self):
        if self.is_timestamp_initialized:
            return
        self.begin_timestamp = time.time()
        self.is_timestamp_initialized = True

    def cur_timestamp(self):
        return time.time() - self.begin_timestamp

    def wait_till(self, timestamp):
        while self.cur_timestamp() < timestamp:
            self.sleep(0.25)

    def sleep(self, sec):
        time.sleep(sec)

    def __del__(self):
        self.logfn.close()
        self.strategy_mem.close()
        self.state_mem.close()

        try:
            shm = shared_memory.SharedMemory(BUFFER_PARAM_NAME)
            shm.close()
            shm.unlink()
        except:
            pass

    def log(self, logstr):
        self.logfn.write(logstr + '\n')
        self.logfn.flush()

    async def tcp_accept_req_client(self):
        conn, address = await self.req_tcp_server.async_accept()
        agent = TcpAgent(conn, address, blocking=False)

        data = await agent.async_recv(8)
        replica_id, rank = struct.unpack('II', data)

        self.api_agent[replica_id].request_agent = agent
        self.api_agent[replica_id].request_rank = rank
        unique_id = self.api_agent[replica_id].id
        print(f'[{datetime.now()}] request client accept from {address} with replica id {replica_id} (unique id: {unique_id})', flush=True)

    async def tcp_accept_resp_client(self):
        conn, address = await self.resp_tcp_server.async_accept()
        agent = TcpAgent(conn, address, blocking=False)

        data = await agent.async_recv(8)
        replica_id, rank = struct.unpack('II', data)

        self.api_agent[replica_id].response_agent = agent
        self.api_agent[replica_id].response_rank = rank
        unique_id = self.api_agent[replica_id].id
        print(f'[{datetime.now()}] response client accept from {address} with replica id {replica_id} (unique id {unique_id})', flush=True)

    async def tcp_init(self, replica_ids):
        async with self.tcp_agent_lock:
            for replica_id in replica_ids:
                assert replica_id not in self.api_agent
                self.api_agent[replica_id] = APIReplica(replica_id, self.cur_strategy, self.approach)
        print(f'[{datetime.now()}] api_server init tcp connections (replicas {", ".join(map(str, replica_ids))})...', flush=True)
        await asyncio.gather(*[self.tcp_accept_req_client() for _ in range(len(replica_ids))],
                             *[self.tcp_accept_resp_client() for _ in range(len(replica_ids))])
        async with self.tcp_agent_lock:
            self.tcp_agent_initialized = True

    async def tcp_close(self, replica_id):
        api_replica = self.api_agent[replica_id]
        print(f'[{datetime.now()}] api_server close replica {replica_id} (unique id: {api_replica.id})...', flush=True)
        async with self.tcp_agent_lock:
            api_replica.state = ReplicaState.DONE
        await api_replica.request_agent.async_send(struct.pack('ii', -1, -1))

    async def send_xfer_cost(self, xfer_cost, replicas_to_close):
        print(f'[{datetime.now()}] api_server send xfer_cost {xfer_cost}...', flush=True)
        async with self.tcp_agent_lock:
            for replica_id in replicas_to_close:
                await self.api_agent[replica_id].request_agent.async_send(struct.pack('ii', -2, -2))
                await self.api_agent[replica_id].request_agent.async_send(struct.pack('f', xfer_cost))

    async def watch_cluster(self):
        self.running_replica_coroutines = {}
        cluster_initialized = False
        while True:
            # first wait for cluster change
            while True:
                with self.lock:
                    state, strategy = decode_tuple(self.strategy_mem)
                    if state > 0 or not cluster_initialized:
                        num_replica, tp, pp, bsz, M1, M2 = strategy
                        self.strategy_mem.buf[:] = encode_tuple(0, strategy, self.strategy_mem)
                        break
                await asyncio.sleep(0.13)

            if state != 0:
                print(f'[{datetime.now()}] cluster strategy changed, state: {state}, strategy: {strategy}, is_finished: {self.is_finished}', flush=True)

            if not cluster_initialized:
                self.cur_strategy = strategy
                self.init_timestamp()

            if state == 1:
                # start new replicas
                replicas_to_close = load_shared_mem(OLD_REPLICAS_NAME)
                replicas_to_create = load_shared_mem(NEW_REPLICAS_NAME)
                if replicas_to_close is not None:
                    assert replicas_to_create is not None
                else:
                    # close all replicas
                    replicas_to_close = list(self.api_agent.keys())
                    replicas_to_create = list(range(num_replica))

                # close replicas
                if len(replicas_to_close) > 0:
                    xfet_cost = load_shared_mem(ESTIMATE_COST_NAME)
                    print(f'[{datetime.now()}] api_server close replicas {", ".join(map(str, replicas_to_close))}...', flush=True)
                    await self.send_xfer_cost(xfet_cost, replicas_to_close)
                    await asyncio.gather(*[self.running_replica_coroutines[replica_id] for replica_id in replicas_to_close])

                # handle onflight queries
                replica_to_batch = {}
                async with self.tcp_agent_lock:
                    for replica_id in replicas_to_close:
                        replica_agent = self.api_agent.pop(replica_id)

                        assert replica_id not in self.interrupt_queries
                        if len(replica_agent.interrupt_queries) > 0:
                            self.interrupt_queries[replica_id] = list(replica_agent.interrupt_queries)
                            int_query_id = replica_agent.interrupt_queries[0]
                            seq_len = self.query_id_to_request[int_query_id].seq_len
                            start_step = self.query_id_to_request[int_query_id].start_step
                            replica_to_batch[str(replica_id)] = (len(replica_agent.interrupt_queries), seq_len, start_step)
                        for query_id in replica_agent.query_onflight:
                            bisect.insort(self.queries, self.query_id_to_request[query_id])

                    if len(replica_to_batch) > 0:
                        self.tcp_agent_initialized = False
                # send interrupt requests info to scheduler
                dump_shared_mem(replica_to_batch, BUFFER_PARAM_NAME)

                # inform scheduler
                if cluster_initialized:
                    with self.lock:
                        self.state_mem.buf[:] = b'1'
                else:
                    cluster_initialized = True

                if strategy[0] == 0 and strategy[1] == 0 and strategy[2] == 0:
                    self.is_finished = True
                    return

                # start new replicas
                self.cur_strategy = strategy
                await self.tcp_init(replicas_to_create)
                print(f'[{datetime.now()}] start {len(replicas_to_create)} new replicas: {replicas_to_create}', flush=True)
                for replica_id in replicas_to_create:
                    replica_coroutine = asyncio.create_task(self.serve_replica(replica_id))
                    self.running_replica_coroutines[replica_id] = replica_coroutine

    async def serve_replica(self, replica_id):
        replica_agent = self.api_agent[replica_id]

        # accept response loop
        resp_agent = replica_agent.response_agent
        self.log(f'[{datetime.now()}] get response from replica id {replica_id} (address: {resp_agent.address})')
        print(f'[{datetime.now()}] get response from replica id {replica_id} (address: {resp_agent.address})', flush=True)
        while True:
            # try to accept 3 float numbers as response
            data = await resp_agent.async_recv(12)
            if len(data) == 0:
                break
            # comm protocol:
            # query_id >= 0: (query_id, schedule_lat, inference_lat)
            # query_id = -1: interrupt request: (-1, replica_size, inference_lat)
            # query_id = -2: finish request init
            query_id, schedule_lat, inference_lat = struct.unpack('fff', data)
            query_id = int(query_id)
            if query_id == -1:
                # start a coroutine to close the request agent
                close_request_task = asyncio.create_task(self.tcp_close(replica_id))
                # handle interrupt request
                replica_size = int(schedule_lat)
                if replica_size > 0:
                    for _ in range(replica_size):
                        data = await resp_agent.async_recv(12)
                        if len(data) == 0:
                            break
                        query_id, interrupt_step, schedule_lat = struct.unpack('fff', data)
                        query_id = int(query_id)
                        interrupt_step = int(interrupt_step)

                        req = self.query_id_to_request[query_id]
                        req.start_step = interrupt_step
                        req.set_latency(self.cur_timestamp(), schedule_lat, inference_lat)

                        if self.approach != 'baseline' and self.approach != 'baseline-triton' and self.xfer_buffer:
                            async with self.tcp_agent_lock:
                                self.query_onflight.remove(query_id)
                                replica_agent.query_onflight.remove(query_id)
                                replica_agent.interrupt_queries.append(query_id)
                        string = f'[{datetime.now()}] interrupt query {query_id} at step {interrupt_step} '
                        string += f'[schedule {req.schedule_lat:.3f}, computation {req.inference_lat:.3f}, overhead {req.tcp_overhead():.3f}]'
                        self.log(string)
                self.log(f'[{datetime.now()}] response replica rank {replica_id} close connection')
                print(f'[{datetime.now()}] response replica rank {replica_id} close connection', flush=True)
                await close_request_task
                return
            elif query_id == -2:
                async with self.tcp_agent_lock:
                    replica_agent.state = ReplicaState.READY
                    # print(f'[{datetime.now()}] replica {replica_id} is ready!', flush=True)
                continue

            req = self.query_id_to_request[query_id]
            req.set_latency(self.cur_timestamp(), schedule_lat, inference_lat)
            # record past requests latencies
            replica_agent.push_record(req)
            async with self.tcp_agent_lock:
                self.query_onflight.remove(query_id)
                replica_agent.query_onflight.remove(query_id)
                replica_agent.request_pool_size -= 1
            self.log(f'[{datetime.now()}] Request {query_id} arrival {req.tstamp:.3f} latency: {req.e2e_lat:.3f}, [schedule {req.schedule_lat:.3f},'
                     f' computation {req.inference_lat:.3f}, overhead {req.tcp_overhead():.3f}] seq_len: {req.seq_len}, replica id: {replica_id}')

    def select_replica(self, num_queries):
        # check if there are ready replicas
        ready_replicas = [replica for replica in self.api_agent.values() if replica.is_ready()]
        if len(ready_replicas) == 0:
            return None, 0
        # prioritize replicas that donot have batch
        for replica in ready_replicas:
            if replica.request_pool_size == 0:
                num_reqs = replica.get_min_bsz(num_queries)
                return replica, num_reqs
        # if batch requests, we only send one batch at a time
        if self.batch_requests:
            return None, 0
        # if all replicas have batch
        ready_replicas = sorted(ready_replicas, key=lambda x: x.request_pool_size)
        # control FT pool size
        pool_sz = 1
        if 'baseline' in self.approach:
            pool_sz = 2
        if ready_replicas[0].request_pool_size >= pool_sz * ready_replicas[0].request_batch_size:
            return None, 0
        num_reqs = ready_replicas[0].get_min_bsz(num_queries)
        return ready_replicas[0], num_reqs

    async def dispatch_request(self):
        while True:
            # wait till query available or all query finished
            while len(self.queries) == 0 or self.is_finished:
                async with self.tcp_agent_lock:
                    if len(self.query_onflight) == 0 or self.is_finished:
                        self.is_finished = True
                        return
                await asyncio.sleep(0.03)

            tstamp = 0 if len(self.interrupt_queries) > 0 else self.queries[0].info()[1]

            # wait till tstamp in short intervals
            while self.cur_timestamp() < tstamp:
                dur = min(tstamp - self.cur_timestamp(), 0.01, 0)
                await asyncio.sleep(dur)
                tstamp = self.queries[0].info()[1]

            # dispatch request
            need_wait = False
            async with self.tcp_agent_lock:
                if not self.tcp_agent_initialized:
                    need_wait = True
                else:
                    # check if there is any interrupt query
                    if len(self.interrupt_queries) > 0:
                        def get_start_step(replica_id):
                            return self.query_id_to_request[self.interrupt_queries[replica_id][0]].start_step
                        int_replica_ids = sorted(self.interrupt_queries.keys(), key=get_start_step, reverse=True)
                        int_replica_ids = int_replica_ids[:len(self.api_agent)]
                        for replica_id in range(len(self.api_agent)):
                            if len(int_replica_ids) == len(self.api_agent):
                                break
                            if replica_id not in int_replica_ids:
                                int_replica_ids.append(replica_id)
                        int_replica_ids = sorted(int_replica_ids)

                        for replica_id, prev_replica_id in enumerate(int_replica_ids):
                            if prev_replica_id not in self.interrupt_queries:
                                continue
                            int_query_ids = sorted(self.interrupt_queries[prev_replica_id])
                            request_agent = self.api_agent[replica_id].request_agent
                            await request_agent.async_send(struct.pack('ii', -1, -2))
                            start_step = self.query_id_to_request[int_query_ids[0]].start_step
                            await request_agent.async_send(struct.pack('ii', len(int_query_ids), start_step))
                            for query_id in int_query_ids:
                                query_offset = self.query_id_to_request[query_id].info()[3]
                                print(f'[{datetime.now()}] >>>>> ({self.cur_timestamp():.3f}) dispatch **int** query {query_id} to replica {replica_id}', flush=True)
                                await request_agent.async_send(struct.pack('ii', query_id, query_offset))
                                self.query_onflight.add(query_id)
                                self.api_agent[replica_id].query_onflight.add(query_id)
                                self.api_agent[replica_id].request_pool_size += 1
                            replica_id += 1
                        self.interrupt_queries = {}
                        continue

                    query_id, tstamp, _, query_offset = self.queries[0].info()
                    num_queries = 0
                    for query in self.queries:
                        if query.info()[1] <= self.cur_timestamp():
                            num_queries += 1
                        else:
                            break
                    api_replica, num_queries = self.select_replica(num_queries)
                    if api_replica is not None:
                        request_agent = api_replica.request_agent
                        if self.batch_requests and num_queries > 0:
                            await request_agent.async_send(struct.pack('ii', -3, num_queries))
                        for _ in range(num_queries):
                            query_id, tstamp, _, query_offset = self.queries[0].info()
                            print(f'[{datetime.now()}] >>>>> ({self.cur_timestamp():.3f}) -- ({tstamp:.3f}) dispatch query {query_id} to replica {api_replica.replica_id}, n: {num_queries}, pool: {api_replica.request_pool_size}', flush=True)
                            await request_agent.async_send(struct.pack('ii', query_id, query_offset))

                            self.queries.pop(0)
                            self.query_onflight.add(query_id)
                            api_replica.query_onflight.add(query_id)
                            api_replica.request_pool_size += 1
                    else:
                        need_wait = True

                    # check if all requests have been finished
                    if len(self.queries) == 0 and len(self.query_onflight) == 0:
                        self.is_finished = True
                        return

            if need_wait:
                await asyncio.sleep(0.023)

    async def run(self):
        await asyncio.gather(self.watch_cluster(), self.dispatch_request())
        print(f'[{datetime.now()}] API server finished', flush=True)


def deploy_api_server(query_trace, model_cfg, profiler_args, approach, server_ip, server_port, lock, strategy_mem, state_mem, xfer_buffer=True, batch_requests=True):
    api_server = APIServer(query_trace, model_cfg, profiler_args, approach, server_ip, server_port, lock, strategy_mem, state_mem, xfer_buffer=xfer_buffer, batch_requests=batch_requests)
    asyncio.run(api_server.run())
