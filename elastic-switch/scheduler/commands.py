import os
import json
from datetime import datetime

MPI_EXEC = os.environ.get('MPI_EXEC', 'mpirun')

DEFAULT_GPU_PER_NODE = 1
PARAM_CLIENT_EXEC = os.environ.get('PARAM_CLIENT_EXEC', None)
FT_INFER_EXEC = os.environ.get('FT_INFER_EXEC', None)

MASTER_IP = os.environ.get('MASTER_IP', '127.0.0.1')
MASTER_PORT = int(os.environ.get('MASTER_PORT', 10024))
API_SERVER_PORT = int(os.environ.get('API_SERVER_PORT', 14041))
LOG_PATH = os.environ.get('LOG_PATH', '/tmp')


def get_hosts_info(hosts):
    nodes_to_slots = {}
    node_visible_dev = {}
    for vnode in hosts:
        if vnode.ip not in nodes_to_slots:
            nodes_to_slots[vnode.ip] = 0
            node_visible_dev[vnode.ip] = []
        if vnode.gpu_id is not None:
            nodes_to_slots[vnode.ip] += 1
            node_visible_dev[vnode.ip].append(vnode.gpu_id)
        else:
            nodes_to_slots[vnode.ip] += DEFAULT_GPU_PER_NODE

    hosts, nprocs = [], 0
    visible_dev = ''
    for ip, slots in nodes_to_slots.items():
        hosts.append(f'{ip}:{slots}')
        nprocs += slots
        if len(node_visible_dev[ip]) > 0:
            local_devs = sorted(node_visible_dev[ip])
            visible_dev += ' -x CUDA_VISIBLE_DEVICES=' + ','.join(map(str, local_devs)) + ' '
    hosts = ','.join(hosts)
    return hosts + visible_dev, nprocs


def get_coord_map(hosts):
    coord_map = {}
    node_visible_devs = {}
    for host in hosts:
        dev_id = 0 if host.gpu_id is None else host.gpu_id
        if host.ip not in node_visible_devs:
            node_visible_devs[host.ip] = []
        node_visible_devs[host.ip].append(dev_id)
    for ip in node_visible_devs:
        node_visible_devs[ip] = sorted(node_visible_devs[ip])

    for host in hosts:
        dev_id = 0 if host.gpu_id is None else host.gpu_id
        local_rank = node_visible_devs[host.ip].index(dev_id)
        key = f'{host.ip}-{local_rank}'
        coord_map[key] = [host.pc_rank, host.coordinate[1], host.coordinate[2]]
    return coord_map


def get_param_client_cmd(hosts, cfg, ckpt_path, strategy):
    hoststr, nprocs = get_hosts_info(hosts)
    env_str = ''
    if os.environ.get('NCCL_IB_DISABLE', None):
        env_str += ' -x NCCL_IB_DISABLE '
    if os.environ.get('NCCL_IB_HCA', None):
        env_str += ' -x NCCL_IB_HCA '
    if os.environ.get('NCCL_SOCKET_IFNAME', None):
        ifname = os.environ['NCCL_SOCKET_IFNAME']
        env_str += f' -x NCCL_SOCKET_IFNAME={ifname} '
        env_str = f' -mca btl_tcp_if_include {ifname} ' + env_str
    cmd = f'{MPI_EXEC} -np {nprocs} -host {hoststr} {env_str} {PARAM_CLIENT_EXEC} {cfg} {ckpt_path} {MASTER_IP} {MASTER_PORT} {strategy[0]} {strategy[1]} {strategy[2]}'
    print(f'[{datetime.now()}] {cmd}')
    return cmd


def get_ft_cmd(hosts, cfg, replica_id, tp_deg, pp_deg, mbs, M1, M2, server_ip, server_port, query_file, profile_path, need_notify=None, need_block=None):
    coord_map = get_coord_map(hosts)
    hoststr, nprocs = get_hosts_info(hosts)

    coord_env_str = json.dumps(coord_map).replace('"', '\\"')
    coord_env = f'-x COORD_MAP="{coord_env_str}"'

    env_str, prefix = '', ''
    if os.environ.get('NCCL_IB_DISABLE', None):
        env_str += ' -x NCCL_IB_DISABLE '
    if os.environ.get('NCCL_IB_HCA', None):
        env_str += ' -x NCCL_IB_HCA '
    env_str += f' -x FT_MICRO_M={mbs},{M1},{M2} '
    if need_notify:
        env_str += f' -x BASELINE_TRITON=1 '
    if need_block:
        env_str += f' -x NAIVE_BLOCK=1 '
    env_str += f' -x NAIVE_NOTIFY=1 '
    if os.environ.get('ENABLE_PROFILER', None):
        prof_logdir = os.environ['ENABLE_PROFILER']
        os.system(f'mkdir -p {os.path.dirname(prof_logdir)}')
        env_str += ' -x FT_NVTX=ON '
        prefix = f'nsys profile -t cudnn,cuda,cublas,osrt,nvtx --output {prof_logdir}n{nprocs}_%q{"{OMPI_COMM_WORLD_RANK}"}.qdrep --force-overwrite true '
    if os.environ.get('NCCL_SOCKET_IFNAME', None):
        ifname = os.environ['NCCL_SOCKET_IFNAME']
        env_str += f' -x NCCL_SOCKET_IFNAME={ifname} '
        env_str = f' -mca btl_tcp_if_include {ifname} ' + env_str

    cmd = f'{MPI_EXEC} -np {nprocs} -host {hoststr} {coord_env} {env_str} {prefix} {FT_INFER_EXEC} {mbs} {replica_id} {tp_deg} {pp_deg} {server_ip} {server_port} {cfg} {query_file} {profile_path}'
    print(f'[{datetime.now()}] {cmd}')
    return cmd


def get_master_node():
    return MASTER_IP, MASTER_PORT


def get_api_server_node():
    return MASTER_IP, API_SERVER_PORT


def clean_logfiles():
    for fn in os.listdir(LOG_PATH):
        if 'ft' in fn or 'inference' in fn or 'param_client' in fn or 'switcher' in fn:
            os.remove(os.path.join(LOG_PATH, fn))


def get_logfile(logfn):
    if '.json' in logfn:
        return os.path.join(LOG_PATH, f'{logfn}')
    return os.path.join(LOG_PATH, f'{logfn}.log')
