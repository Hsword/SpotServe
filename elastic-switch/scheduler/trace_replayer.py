import json
import socket


class VNode:
    id = 0

    def __init__(self, ip, gpu_id=None):
        self.ip = ip
        self.gpu_id = gpu_id
        self.local_rank = -1
        self.pc_rank = -1

        self.id = VNode.id
        VNode.id += 1

        # for strategy switch communication, format: (dp, tp, pp)
        self.prev_strategy = None
        self.prev_coordinate = None
        self.strategy = None
        self.coordinate = None

    def __hash__(self):
        return hash((self.ip, self.gpu_id))

    def __repr__(self) -> str:
        if self.gpu_id is None:
            return f'VNode({self.ip})'
        return f'VNode({self.ip}:{self.gpu_id})'


class TraceReplayer:
    def __init__(self, trace_file, hostfile, nnodes, gpu_per_node=1, dry_run=False):
        self.trace_file = trace_file
        self.hostfile = hostfile
        self.nnodes = nnodes
        self.gpu_per_node = gpu_per_node
        self.dry_run = dry_run

        self.read_replay_info()

    def read_replay_info(self):
        self.hosts = {}
        self.trace = []
        self.physical_node_to_vnodes = {}

        # read hostfile
        ip_to_gpus = {}
        with open(self.hostfile, 'r') as f:
            per_gpu_vm, host_id, vhost_id = False, 0, 0
            for i, line in enumerate(f.readlines()):
                if i == 0 and line.startswith('#'):
                    per_gpu_vm = True
                    continue

                gpu_id = None
                if self.dry_run:
                    ip = line.strip()
                else:
                    if per_gpu_vm:
                        ip, gpu_id = line.strip().split(':')
                        gpu_id = int(gpu_id)
                    else:
                        ip = line.strip()
                    ip = socket.gethostbyname(ip)
                if ip not in ip_to_gpus:
                    ip_to_gpus[ip] = []

                if per_gpu_vm:
                    self.hosts[vhost_id] = VNode(ip, gpu_id)
                    self.physical_node_to_vnodes[host_id] = [vhost_id]
                    vhost_id += 1
                    ip_to_gpus[ip].append(gpu_id)
                else:
                    if host_id not in self.physical_node_to_vnodes:
                        self.physical_node_to_vnodes[host_id] = []
                    for gpu_id in range(self.gpu_per_node):
                        self.hosts[vhost_id] = VNode(ip, gpu_id)
                        self.physical_node_to_vnodes[host_id].append(vhost_id)
                        vhost_id += 1
                        ip_to_gpus[ip].append(gpu_id)
                host_id += 1

                if len(self.physical_node_to_vnodes) == self.nnodes:
                    break

        # assign local rank
        for ip in ip_to_gpus:
            ip_to_gpus[ip].sort()
        for vhost_id, vnode in self.hosts.items():
            if vnode.gpu_id is None:
                vnode.local_rank = 0
            else:
                vnode.local_rank = ip_to_gpus[vnode.ip].index(vnode.gpu_id)

        # read trace event file
        remain_nodes = list(self.physical_node_to_vnodes.keys())
        node_map = {}
        with open(self.trace_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # self-defined comment for trace file
                if line.startswith('#'):
                    continue
                event = json.loads(line)
                nodes = []
                if event[1] == 'add':
                    for node_id in event[2]['nodes']:
                        host_id = remain_nodes.pop(0)
                        node_map[node_id] = host_id
                        for vhost_id in self.physical_node_to_vnodes[host_id]:
                            nodes.append(self.hosts[vhost_id])
                elif event[1] == 'remove':
                    for node_id in event[2]['nodes']:
                        host_id = node_map.pop(node_id)
                        for vhost_id in self.physical_node_to_vnodes[host_id]:
                            nodes.append(self.hosts[vhost_id])
                        remain_nodes.append(host_id)
                elif event[1] == 'DONE':
                    for node_id in node_map:
                        host_id = node_map[node_id]
                        for vhost_id in self.physical_node_to_vnodes[host_id]:
                            nodes.append(self.hosts[vhost_id])
                event[2]['nodes'] = nodes
                self.trace.append(event)

    def get_available_hosts(self, timestamp=None):
        if timestamp is None:
            return list(self.hosts.values())
        assert False, 'Not Implemented'

    def __iter__(self):
        self.event_id = 0
        return self

    def __next__(self):
        if self.event_id >= len(self.trace):
            raise StopIteration
        event = self.trace[self.event_id]
        self.event_id += 1
        return event

    def __getitem__(self, index):
        return self.trace[index]

    def __len__(self):
        return len(self.trace)
