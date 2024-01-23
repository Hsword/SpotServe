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
from scheduler.parallel.solver import StrategyOptimizer
from scheduler.utils import encode_tuple, decode_tuple
from util.util import TcpServer, TcpAgent


class APIReplica:
    id = 0

    def __init__(self, replica_id, strategy, approach):
        self.replica_id = replica_id
        self.request_agent = None
        self.request_rank = None
        self.response_agent = None
        self.response_rank = None

        if 'baseline' in approach:
            # disable currently
            self.request_agent_ready = True #False
        else:
            self.request_agent_ready = True

        self.id = APIReplica.id
        APIReplica.id += 1

        # track requests status
        self.last_resp_tstamp = 0
        self.past_latencies = []
        self.request_pool_size = 0
        self.strategy = strategy
        self.request_batch_size = strategy[3]
        self.M1 = strategy[4]
        self.M2 = strategy[5]
        self.div_bs = self.request_batch_size // self.M1

    def is_ready(self):
        return self.request_agent_ready

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

    def estimate_remain_request_pool_info(self, cur_tstamp, remain_slot, approach, cost_model):
        # convert ms to seconds
        avg_lat = self.get_avg_latency(comp_only=True) / 1000
        iter_remain = max(avg_lat - (cur_tstamp - self.last_resp_tstamp), 0)
        total_time = iter_remain + ((self.request_pool_size - self.request_batch_size) // self.request_batch_size) * avg_lat
        if total_time < remain_slot:
            return 0, 0
        remain_slot = remain_slot - iter_remain
        remain_pool_size = self.request_pool_size
        ret_num_req, ret_num_iter = 0, -1
        while remain_slot > avg_lat and remain_pool_size > 0:
            remain_slot -= avg_lat
            remain_pool_size -= self.request_batch_size
            ret_num_req += self.request_batch_size
        # estimate remain size
        if approach != 'baseline':
            # FIXME: current assume request a full final batch
            final_bs = self.request_batch_size
            ret_num_iter = cost_model(remain_slot, final_bs, self.M1, self.M2)
            if ret_num_iter == 0:
                # previous batch can be finished
                ret_num_iter = -1
            else:
                ret_num_req += final_bs
            print(f'[{datetime.now()}] +Replica {self.replica_id} remain pool size: {remain_pool_size}, estimate req: {ret_num_req}, {ret_num_iter}')
        return ret_num_req, ret_num_iter


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


class APIServer:
    def __init__(self, query_trace, model_cfg, profiler_args, approach, server_ip, server_port, lock, strategy_mem, state_mem):
        self.query_trace = query_trace
        self.model_cfg = model_cfg
        self.profiler_args = profiler_args
        self.approach = approach
        self.server_port = server_port
        self.lock = lock
        self.strategy_mem = strategy_mem
        self.state_mem = state_mem

        # read model config
        self.read_ini_config()

        self.queries = self.read_query_trace()
        self.query_id_to_request = {query.query_id: query for query in self.queries}
        self.onflight_queues = {}
        self.onlight_queue_id = -1
        self.query_onflight = None
        self.interrupt_queries = {} # Dict(replica_rank: List[query_id])

        self.logfile = sched_cmd.get_logfile('inference_service')
        self.logfn = open(self.logfile, 'a')

        self.req_tcp_server = TcpServer(server_ip, server_port)
        self.resp_tcp_server = TcpServer(server_ip, server_port + 1)

        self.api_agent = {}
        self.tcp_agent_lock = asyncio.Lock()
        self.tcp_agent_initialized = False

        self.cur_strategy = None
        self.cur_replica_id = -1
        self.is_timestamp_initialized = False
        self.response_gathered = False
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
            shm = shared_memory.SharedMemory('buffer_param')
            shm.close()
            shm.unlink()
        except:
            pass

    def dump_replica(self):
        replica_ids = sorted([replica.id for replica in self.api_agent.values()])
        string = f'{len(self.api_agent)} replicas: ' + ', '.join(map(str, replica_ids))
        return string

    def log(self, logstr):
        self.logfn.write(logstr + '\n')
        self.logfn.flush()

    def select_replica(self):
        # check if there are ready replicas
        ready_replicas = [replica for replica in self.api_agent.values() if replica.is_ready()]
        if len(ready_replicas) == 0:
            return None
        # if there are replicas that have not fulfill one-batch
        div_bs = self.api_agent[0].div_bs
        for replica in ready_replicas:
            if replica.request_pool_size % div_bs != 0:
                return replica
        # if all replicas have fulfilled one-batch
        ready_replicas = sorted(ready_replicas, key=lambda x: x.request_pool_size)
        # control pool size
        if ready_replicas[0].request_pool_size > 100 * div_bs:
            return None
        return ready_replicas[0]

    async def tcp_accept_req_client(self):
        conn, address = await self.req_tcp_server.async_accept()
        agent = TcpAgent(conn, address, blocking=False)

        data = await agent.async_recv(8)
        replica_rank, rank = struct.unpack('II', data)

        self.api_agent[replica_rank].request_agent = agent
        self.api_agent[replica_rank].request_rank = rank
        print(f'[{datetime.now()}] request client accept from {address} with replica rank {replica_rank} '
              f'(replica id {self.api_agent[replica_rank].id})', flush=True)

    async def tcp_accept_resp_client(self):
        conn, address = await self.resp_tcp_server.async_accept()
        agent = TcpAgent(conn, address, blocking=False)

        data = await agent.async_recv(8)
        replica_rank, rank = struct.unpack('II', data)

        self.api_agent[replica_rank].response_agent = agent
        self.api_agent[replica_rank].response_rank = rank
        print(f'[{datetime.now()}] response client accept from {address} with replica rank {replica_rank} '
              f'(replica id {self.api_agent[replica_rank].id})', flush=True)

    async def tcp_init(self, replica):
        for replica_id in range(replica):
            self.api_agent[replica_id] = APIReplica(replica_id, self.cur_strategy, self.approach)
        print(f'[{datetime.now()}] api_server init tcp connections ({self.dump_replica()})...', flush=True)
        await asyncio.gather(*[self.tcp_accept_req_client() for _ in range(replica)],
                             *[self.tcp_accept_resp_client() for _ in range(replica)])
        async with self.tcp_agent_lock:
            self.tcp_agent_initialized = True

            self.onlight_queue_id += 1
            self.onflight_queues[self.onlight_queue_id] = set()
            self.query_onflight = self.onflight_queues[self.onlight_queue_id]

    async def tcp_close(self, replica_id, replica_remain_info):
        api_replica = self.api_agent[replica_id]
        print(f'[{datetime.now()}] api_server close replica rank {replica_id} (replica id {api_replica.id})...', flush=True)
        request_agent = api_replica.request_agent
        await request_agent.async_send(struct.pack('ii', -1, -1))
        # TODO: set how many requests to wait for
        num_req, num_iter = replica_remain_info.get(replica_id, (-1, -1))
        await request_agent.async_send(struct.pack('ii', num_req, num_iter))
        del self.api_agent[replica_id]

    async def watch_cluster(self):
        cluster_initialized = False
        while True:
            if cluster_initialized:
                with self.lock:
                    state, strategy = decode_tuple(self.strategy_mem)
                    # set strategy have been read
                    self.strategy_mem.buf[:] = encode_tuple(0, strategy, self.strategy_mem)
                    if 0 in strategy:
                        self.is_finished = True
                num_replica, tp, pp, bsz, M1, M2 = strategy

                if self.is_finished:
                    self.tcp_agent_initialized = False
                    return
            else:
                # wait for vaild strategy
                while True:
                    with self.lock:
                        state, strategy = decode_tuple(self.strategy_mem)
                        if state >= 0:
                            num_replica, tp, pp, bsz, M1, M2 = strategy
                            self.strategy_mem.buf[:] = encode_tuple(0, strategy, self.strategy_mem)
                            break
                    await asyncio.sleep(0.023)

            if state != 0:
                print(f'[{datetime.now()}] cluster strategy changed, state: {state}, strategy: {strategy}')

            if not cluster_initialized:
                self.cur_strategy = strategy
                await self.tcp_init(num_replica)
                self.init_timestamp()
                cluster_initialized = True
            elif state == 1:
                # wait until we can close the tcp connections
                shm = shared_memory.SharedMemory('est_cost')
                remain_slot = float(bytes(shm.buf[:]).decode('utf-8'))
                shm.close()
                persist_tstamp = self.cur_timestamp()
                while True:
                    cur_tstamp = self.cur_timestamp()
                    remain_slot = remain_slot - (cur_tstamp - persist_tstamp)
                    replica_remain_info = {}
                    for replica_id, replica_agent in self.api_agent.items():
                        dp, tp, pp, _, _, _ = replica_agent.strategy
                        cost_model = lambda slot, bsz, M1, M2: self.strategy_solver.estimate_remain_steps(slot, tp, pp, bsz, M1, M2)
                        num_req, num_iter = replica_agent.estimate_remain_request_pool_info(cur_tstamp, remain_slot, self.approach, cost_model)
                        if num_req >= 0:
                            replica_remain_info[replica_id] = (num_req, num_iter)
                    if len(replica_remain_info) > 0 or remain_slot <= 1:
                        break

                # first stop current tcp connection, then init new tcp connection
                async with self.tcp_agent_lock:
                    self.tcp_agent_initialized = False
                    print(f'[{datetime.now()}] api_server is closing all tcp connections ({self.dump_replica()})...', flush=True)
                    await asyncio.gather(*[self.tcp_close(replica_id, replica_remain_info) for replica_id in range(len(self.api_agent))])
                    print(f'[{datetime.now()}] api_server closed all tcp connections (#replica {len(self.api_agent)})', flush=True)

                while not self.response_gathered:
                    await asyncio.sleep(0.03)
                with self.lock:
                    self.state_mem.buf[:] = b'1'
                self.response_gathered = False
                self.cur_strategy = strategy
                await self.tcp_init(num_replica)
            else:
                await asyncio.sleep(0.13)

    async def close_request_agent(self, replica_rank):
        request_agent = self.api_agent[replica_rank].request_agent
        await request_agent.async_send(struct.pack('ii', -1, -1))
        # a fake message to finish the protocol
        await request_agent.async_send(struct.pack('ii', -1, -1))
        print(f'[{datetime.now()}] All requests finished!!!! resquest replica rank {replica_rank} close connection', flush=True)

    async def dispatch_request(self):
        while True:
            # wait till tcp initialized
            while True:
                async with self.tcp_agent_lock:
                    if self.tcp_agent_initialized:
                        break
                await asyncio.sleep(0.03)

            # wait till query available or all query finished
            while len(self.queries) == 0 or self.is_finished:
                async with self.tcp_agent_lock:
                    if len(self.query_onflight) == 0 or self.is_finished:
                        # all query finished
                        await asyncio.gather(*[self.close_request_agent(replica_id) for replica_id in range(len(self.api_agent))])
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
            async with self.tcp_agent_lock:
                need_wait = False
                if not self.tcp_agent_initialized:
                    continue

                # check if there is any interrupt query
                if len(self.interrupt_queries) > 0:
                    replica_id = 0
                    for prev_replica_id in sorted(self.interrupt_queries.keys()):
                        int_query_ids = sorted(self.interrupt_queries[prev_replica_id])
                        request_agent = self.api_agent[replica_id].request_agent
                        await request_agent.async_send(struct.pack('ii', -1, -2))
                        start_step = self.query_id_to_request[int_query_ids[0]].start_step
                        await request_agent.async_send(struct.pack('ii', len(int_query_ids), start_step))
                        for query_id in int_query_ids:
                            query_offset = self.query_id_to_request[query_id].info()[3]
                            await request_agent.async_send(struct.pack('II', query_id, query_offset))
                            self.query_onflight.add(query_id)
                        replica_id += 1
                    self.interrupt_queries = {}
                    continue

                query_id, tstamp, _, query_offset = self.queries[0].info()
                api_replica = self.select_replica()
                if api_replica is not None:
                    request_agent = api_replica.request_agent
                    # print(f'[{datetime.now()}] ({self.cur_timestamp():.3f}) -- ({tstamp:.3f}) dispatch query {query_id} to replica {api_replica.replica_id}')
                    await request_agent.async_send(struct.pack('II', query_id, query_offset))

                    self.queries.pop(0)
                    self.query_onflight.add(query_id)
                    api_replica.request_pool_size += 1
                else:
                    need_wait = True
                # check if all requests have been finished
                if len(self.queries) == 0 and len(self.query_onflight) == 0:
                    await asyncio.gather(*[self.close_request_agent(replica_id) for replica_id in range(len(self.api_agent))])
                    self.is_finished = True
                    return

            if need_wait:
                await asyncio.sleep(0.03)

    async def get_response(self, replica_id, replica_agent):
        resp_agent = replica_agent.response_agent
        self.log(f'get response from replica id {replica_id} (address: {resp_agent.address})')
        print(f'[{datetime.now()}] get response from replica rank {replica_id} (address: {resp_agent.address})')
        while True:
            # try to accept 3 float numbers as response
            data = await resp_agent.async_recv(12)
            if len(data) == 0:
                break
            query_id, schedule_lat, inference_lat = struct.unpack('fff', data)
            query_id = int(query_id)
            if query_id == -1:
                # handle interrupt request
                replica_size = int(schedule_lat)
                if replica_size > 0:
                    assert replica_id not in self.interrupt_queries
                    self.interrupt_queries[replica_id] = []
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
                        async with self.tcp_agent_lock:
                            self.query_onflight.remove(query_id)
                        self.interrupt_queries[replica_id].append(query_id)
                        string = f'[{datetime.now()}] interrupt query {query_id} at step {interrupt_step}'
                        self.log(string)
                        print(string)
                print(f'[{datetime.now()}] response replica rank {replica_id} close connection', flush=True)
                return
            elif query_id == -2:
                replica_agent.request_agent_ready = True
                continue

            # print(f'[{datetime.now()}] response {query_id}', flush=True)
            req = self.query_id_to_request[query_id]
            req.set_latency(self.cur_timestamp(), schedule_lat, inference_lat)
            # record past requests latencies
            replica_agent.push_record(req)
            async with self.tcp_agent_lock:
                self.query_onflight.remove(query_id)
                replica_agent.request_pool_size -= 1
            self.log(f'[{datetime.now()}] Request {query_id} latency: {req.e2e_lat:.3f}, [schedule {schedule_lat:.3f},'
                     f' computation {inference_lat:.3f}, overhead {req.tcp_overhead():.3f}] seq_len: {req.seq_len}')

    async def gather_response(self):
        self.past_api_agent_ids = set()
        last_onflight_queue_id = -1
        while True:
            # wait till tcp initialized
            while True:
                async with self.tcp_agent_lock:
                    if self.tcp_agent_initialized:
                        # check if getting old api_replica
                        cur_api_replicas = [api_replica for api_replica in self.api_agent.values()]
                        if len(cur_api_replicas) > 0 and cur_api_replicas[0].id not in self.past_api_agent_ids:
                            for replica in cur_api_replicas:
                                self.past_api_agent_ids.add(replica.id)

                            resp_agents = [(replica.replica_id, replica) for replica in cur_api_replicas]
                            # insert onflight requests to queries
                            onflight_queue = self.onflight_queues.pop(last_onflight_queue_id, set())
                            for query_id in onflight_queue:
                                bisect.insort(self.queries, self.query_id_to_request[query_id])
                            if len(self.queries) == 0:
                                return
                            break
                    if self.is_finished:
                        return
                await asyncio.sleep(0.05)

            # gather response
            await asyncio.gather(*[self.get_response(*resp_ag) for resp_ag in resp_agents])
            # send interrupt requests info to scheduler
            replica_to_batch = {}
            for replica_id, values in self.interrupt_queries.items():
                seq_len = self.query_id_to_request[values[0]].seq_len
                start_step = self.query_id_to_request[values[0]].start_step
                replica_to_batch[str(replica_id)] = (len(values), seq_len, start_step)
            replica_to_batch_str = json.dumps(replica_to_batch).encode('utf-8')
            try:
                shm = shared_memory.SharedMemory('buffer_param')
                shm.close()
                shm.unlink()
            except:
                pass
            shm = shared_memory.SharedMemory('buffer_param', create=True, size=len(replica_to_batch_str))
            shm.buf[:] = replica_to_batch_str
            shm.close()

            last_onflight_queue_id += 1
            self.response_gathered = True

    async def serve(self):
        await asyncio.gather(self.watch_cluster(), self.dispatch_request(), self.gather_response())

    def run(self):
        asyncio.run(self.serve())
        print(f'[{datetime.now()}] API server finished', flush=True)


def deploy_api_server(query_trace, model_cfg, profiler_args, approach, server_ip, server_port, lock, strategy_mem, state_mem):
    api_server = APIServer(query_trace, model_cfg, profiler_args, approach, server_ip, server_port, lock, strategy_mem, state_mem)
    api_server.run()
