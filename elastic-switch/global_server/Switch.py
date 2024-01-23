import math
import numpy as np
import json
import sys
import random

from global_server.KMmatcher import KMMatcher
from scheduler.trace_replayer import VNode
from util.util import timestamp

_ENCODE_BASE_ = 128
# _REPLICA_ID_MASK = _ENCODE_BASE_ ** 3

def _layer_encode(layer_id, tp_stage, tp_grain):
    tp_grain = max(0, tp_grain - 1)
    return tp_grain * (_ENCODE_BASE_**2) + tp_stage * _ENCODE_BASE_ + layer_id

def _layer_decode(layer_encode_id):
    '''
    return: layer_id, tp_stage, tp_grain
    '''
    layer_id = layer_encode_id % _ENCODE_BASE_
    tmp = layer_encode_id // _ENCODE_BASE_
    tp_stage = tmp % _ENCODE_BASE_
    tp_grain = tmp // _ENCODE_BASE_ + 1
    return layer_id, tp_stage, tp_grain

def _insert_entry(dct, rank, mode, layer_id, to=None, prior=False):
    rank = str(rank)
    if rank not in dct:
        dct[rank] = {"pstay":[], "psend":[], "precv":[]}
    if mode not in dct[rank]:
        dct[rank][mode] = []
    if layer_id is None:
        return
    if to is None:
        dct[rank][mode].append(layer_id)
    else:
        dct[rank][mode].append({"layer":layer_id, "peer":to, "prior":prior})

class Switch:
    def __init__(self, layer_num=32, vocab_size=50304, hidden_size=1024, max_seq_len=128) -> None:
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.dont_stop_ft = False

        self.ip2comm = {}
        self.MAGIC_LAYER_NUM = -layer_num - 1
        self.MAGIC_POST_DECODER_LAYER_ID = -layer_num - 2
        random.seed(42)

    '''
    def large_tp_group_assign(self, old_s, old_nodes, new_s, new_nodes, replicas_stay, clustered_intersect_nodes, cluster_size, buffer_param=None, verbose=False):
        assert new_s[1] % cluster_size == 0
        switch_buffer = buffer_param is not None

        intersect_cluster_num = len(clustered_intersect_nodes) // cluster_size
        dpxpp = new_s[0] * new_s[2] * (new_s[1] // cluster_size)

        map_cluster = []
        graph_weights = np.zeros((intersect_cluster_num, dpxpp))
        for cluster_id in range(intersect_cluster_num):
            for coord_dpxpp_id in range(dpxpp):
                dp, pp = coord_dpxpp_id // (new_s[2] * new_s[1] // cluster_size) , coord_dpxpp_id % new_s[2]
                tp_base = (coord_dpxpp_id // new_s[2]) % (new_s[1] // cluster_size)

                # step 1: build weight matrix by TP
                weights = np.zeros((cluster_size, cluster_size))
                for cluster_index in range(cluster_size):
                    vnode = clustered_intersect_nodes[cluster_id*cluster_size+cluster_index]
                    for tp_offset in range(cluster_size):
                        tp = tp_base * new_s[1] + tp_offset
                        new_stages = replicas_stay[dp], tp, pp
                        buff_config = None
                        if switch_buffer and new_stages[0] < old_s[0]:
                            buff_config = buffer_param[new_stages[0]]['batchxbeam'], buffer_param[new_stages[0]]['session_len'], \
                                buffer_param[new_stages[0]]['mem_len'], buffer_param[new_stages[0]]['beam']
                        w = self._get_graph_weights(vnode.prev_coordinate, old_s, new_stages, new_s, buff_config)
                        weights[cluster_index, tp_offset] = w

                # step 2: run KM to match intra-tpgroup
                matcher = KMMatcher(weights)
                graph_weights[cluster_id, coord_dpxpp_id] = matcher.solve()
                map_cluster.append(matcher)

        # step 3: run KM to build the map of cluster <--> coord_dpxpp
        if verbose:
            print(graph_weights)

        is_transposed = False # Transpose weight matrix if dim[0] > dim[1]
        if graph_weights.shape[0] > graph_weights.shape[1]:
            is_transposed = True
            graph_weights = graph_weights.T

        matcher = KMMatcher(graph_weights)
        matcher.solve()

        assigned_coordinate = [False] * dpxpp
        for x in range(matcher.n):
            y = int(matcher.xy[x])
            w = matcher.weights[x, y]
            if is_transposed:
                cls, cod = y, x
            else:
                cls, cod = x, y
            # cls is intersect_cluster, cod is dpxpp

            dp_stage, pp_stage = cod // (new_s[2] * new_s[1] // cluster_size), cod % new_s[2]
            tp_base = (cod // new_s[2]) % (new_s[1] // cluster_size)
            tp_matcher = map_cluster[cls * dpxpp + cod]

            timestamp('do_switch(DPxPP)', f"matched cluster {cls} as ({dp_stage},?,{pp_stage}), weight={w}")
            for i in range(cluster_size):
                vnode = clustered_intersect_nodes[cls * cluster_size + i]
                tp_stage = int(tp_matcher.xy[i]) + tp_base * new_s[1]
                vnode.strategy = new_s
                vnode.coordinate = dp_stage, tp_stage, pp_stage
                timestamp('do_switch(TP)', f"matched {vnode}(rank {vnode.pc_rank}) as {vnode.coordinate}, weight={tp_matcher.weights[i, tp_stage]}")

            assigned_coordinate[cod] = True

        # step 4: assign remaining cluster
        j = 0
        for i in range(dpxpp):
            if assigned_coordinate[i] == False:
                while clustered_new_nodes[j].coordinate is not None:
                    j += cluster_size
                # j == world_size_new # j is impossibly geq than world_size_new
                for k in range(cluster_size):
                    vnode = clustered_new_nodes[j+k]
                    assert vnode.coordinate is None
                    vnode.strategy = new_s
                    vnode.coordinate = i // new_s[2], k, i % new_s[2]
                    timestamp('do_switch', f"linked {vnode}(rank {vnode.pc_rank}) to {vnode.coordinate}")
                j += cluster_size
                assigned_coordinate[i] = True
    '''

    def node_load_from_disk(self, new_s, tp_coord, pp_coord):
        # return a dict, then you should invert it to json, and send it to its rank
        # the node received this json must be an empty rank, or PC will occur error
        # for baseline only
        load_dict = {}
        load_dict['load_disk'] = True
        load_dict['tp'] = new_s[1]
        load_dict['pp'] = new_s[2]
        load_dict['tp_stage'] = tp_coord
        load_dict['pp_stage'] = pp_coord

        return load_dict

    def node_clean_weight(self):
        return {"clean_rank": True}

    def tp_group_assign(self, old_s, old_nodes, new_s, new_nodes, replicas_stay, intersect_nodes, buffer_param=None, verbose=False):
        clustered_intersect_nodes = sorted(intersect_nodes, key=lambda x: x.ip)
        ip = clustered_intersect_nodes[0].ip.split(':')[0]
        gpu_per_node = 0
        while gpu_per_node < len(clustered_intersect_nodes) and clustered_intersect_nodes[gpu_per_node].ip.split(':')[0] == ip:
            gpu_per_node += 1
        timestamp('do_switch', f'find gpu_per_node is {gpu_per_node}')

        # if gpu_per_node < new_s[1]:
        #     self.large_tp_group_assign( old_s, old_nodes, new_s, new_nodes, replicas_stay, clustered_intersect_nodes, gpu_per_node, buffer_param=buffer_param, verbose=verbose)
        #     return

        switch_buffer = buffer_param is not None
        cluster_size = min(new_s[1], gpu_per_node)
        assert len(intersect_nodes) % cluster_size == 0
        if new_s[1] > gpu_per_node:
            assert new_s[1] % cluster_size == 0


        clustered_new_nodes = sorted(new_nodes, key=lambda x: x.ip)
        intersect_cluster_num = len(clustered_intersect_nodes) // cluster_size
        dpxpp = new_s[0] * new_s[2]* (new_s[1] // cluster_size)

        map_cluster = []
        graph_weights = np.zeros((intersect_cluster_num, dpxpp))
        for cluster_id in range(intersect_cluster_num):
            for coord_dpxpp_id in range(dpxpp):
                dp, pp = coord_dpxpp_id // (new_s[2] * new_s[1] // cluster_size), coord_dpxpp_id % new_s[2]
                tp_base = (coord_dpxpp_id // new_s[2]) % (new_s[1] // cluster_size)

                # step 1: build weight matrix by TP
                weights = np.zeros((cluster_size, cluster_size))
                for cluster_index in range(cluster_size):
                    vnode = clustered_intersect_nodes[cluster_id*cluster_size+cluster_index]
                    for tp_offset in range(cluster_size):
                        tp = tp_base * cluster_size + tp_offset
                        new_stages = replicas_stay[dp], tp, pp
                        buff_config = None
                        if switch_buffer and new_stages[0] < old_s[0] and buffer_param[new_stages[0]] is not None:
                            # if batchxbeam is unequal, it must be in diffrent replicas, and return 0 immediately
                            # So here, buffer config is simply set to `new` version, since old & new are the same
                            buff_config = buffer_param[new_stages[0]]['batchxbeam'], buffer_param[new_stages[0]]['session_len'], \
                                buffer_param[new_stages[0]]['mem_len'], buffer_param[new_stages[0]]['beam']
                        w = self._get_graph_weights(vnode.prev_coordinate, old_s, new_stages, new_s, buff_config)
                        weights[cluster_index, tp_offset] = w

                # step 2: run KM to match intra-tpgroup
                matcher = KMMatcher(weights)
                graph_weights[cluster_id, coord_dpxpp_id] = matcher.solve()
                map_cluster.append(matcher)

        # step 3: run KM to build the map of cluster <--> coord_dpxpp
        if verbose:
            print(graph_weights)

        is_transposed = False # Transpose weight matrix if dim[0] > dim[1]
        if graph_weights.shape[0] > graph_weights.shape[1]:
            is_transposed = True
            graph_weights = graph_weights.T

        matcher = KMMatcher(graph_weights)
        matcher.solve()

        assigned_coordinate = [False] * dpxpp
        for x in range(matcher.n):
            y = int(matcher.xy[x])
            w = matcher.weights[x, y]
            if is_transposed:
                cls, cod = y, x
            else:
                cls, cod = x, y
            # cls is intersect_cluster, cod is dpxpp

            dp_stage, pp_stage = cod // (new_s[2] * new_s[1] // cluster_size), cod % new_s[2]
            tp_base = (cod // new_s[2]) % (new_s[1] // cluster_size)
            tp_matcher = map_cluster[cls * dpxpp + cod]

            timestamp('do_switch(DPxPP)', f"matched cluster {cls} as ({dp_stage},{tp_base*cluster_size}-{(tp_base+1)*cluster_size-1},{pp_stage}), weight={w}")
            for i in range(cluster_size):
                vnode = clustered_intersect_nodes[cls * cluster_size + i]
                tp_offset = int(tp_matcher.xy[i])
                tp_stage = tp_offset + tp_base * cluster_size
                vnode.strategy = new_s
                vnode.coordinate = dp_stage, tp_stage, pp_stage
                timestamp('do_switch(TP)', f"matched {vnode}(rank {vnode.pc_rank}) as {vnode.coordinate}, weight={tp_matcher.weights[i, tp_offset]}")

            assigned_coordinate[cod] = True

        # step 4: assign remaining cluster
        j = 0
        for i in range(dpxpp):
            if assigned_coordinate[i] == False:
                while clustered_new_nodes[j].coordinate is not None:
                    j += cluster_size
                # j == world_size_new # j is impossibly geq than world_size_new
                k_base = (i // new_s[2]) % (new_s[1] // cluster_size)
                for k in range(cluster_size):
                    vnode = clustered_new_nodes[j+k]
                    assert vnode.coordinate is None
                    vnode.strategy = new_s
                    vnode.coordinate = i // (new_s[2] * new_s[1] // cluster_size), k + k_base * cluster_size, i % new_s[2]
                    timestamp('do_switch', f"linked {vnode}(rank {vnode.pc_rank}) to {vnode.coordinate}")
                j += cluster_size
                assigned_coordinate[i] = True

    def do_switch(self, old_s, old_nodes, new_s, new_nodes, replicas_stay
                    , buffer_param=None, km_match=True, group_tp=True, memory_tolarance=-1.
                    , enable_no_lock=True, try_dont_stop_ft=True, slpt=-1.
                    , verbose=False, json_out_file=sys.stdout):
        # replicas_stay given IN INCREASING ORDER
        # e.g.  dp 4 -> 3 : replicas_stay = [0, 2, 3] if eliminate replica 1
        #       dp 2 -> 4 : replicas_stay = [0, 1, 2, 3], in which 2,3 are new replicas
        # buffer_param    : old_replica_id -> buffer_dict

        timestamp('do_switch', f'switching from {old_s} to {new_s}: computing json start')

        tp_grain = max(old_s[1], new_s[1])
        world_size_old = old_s[0] * old_s[1] * old_s[2]
        world_size_new = new_s[0] * new_s[1] * new_s[2]
        switch_buffer = buffer_param is not None

        self.ip2comm = {}

        assert buffer_param is None or len(buffer_param) == old_s[0]
        assert len(replicas_stay) == new_s[0]
        assert len(old_nodes) == world_size_old
        assert len(new_nodes) >= world_size_new

        intersect_nodes = [vn for vn in new_nodes if vn in old_nodes]
        dying_ranks = set([vn.pc_rank for vn in old_nodes if not vn in new_nodes])

        for vnode in [vn for vn in new_nodes if vn not in old_nodes]:
            vnode.prev_coordinate = None

        rank2vnode = dict()
        for vnode in old_nodes:
            if not vnode.pc_rank in rank2vnode:
                rank2vnode[vnode.pc_rank] = vnode
            if not vnode.ip in self.ip2comm:
                self.ip2comm[vnode.ip] = [0, 0]
        for vnode in new_nodes:
            if not vnode.pc_rank in rank2vnode:
                rank2vnode[vnode.pc_rank] = vnode
            if not vnode.ip in self.ip2comm:
                self.ip2comm[vnode.ip] = [0, 0]

        cond1 = old_s[0] <= new_s[0] and old_s[1] == new_s[1] and old_s[2] == new_s[2]
        cond2 = len(dying_ranks) == 0 # no rank dying
        self.dont_stop_ft = cond1 and cond2 and try_dont_stop_ft

        random_assign_baseline = False

        if self.dont_stop_ft:
            switch_buffer = False
            assigned_coordinate = [False] * world_size_new

            for vnode in old_nodes:
                vnode.strategy = new_s
                vnode.coordinate = vnode.prev_coordinate
                coord_id = vnode.coordinate[1] + vnode.coordinate[2] * new_s[1] + vnode.coordinate[0] * new_s[1] * new_s[2]
                assigned_coordinate[coord_id] = True
                timestamp('do_switch', f"DONT STOP FT: linked {vnode}(rank {vnode.pc_rank}) to {vnode.coordinate}")

            x, y = 0, 0
            clustered_new_nodes = sorted(new_nodes, key=lambda x: x.ip)
            clustered_new_nodes = clustered_new_nodes[:world_size_new]
            while x < world_size_new:
                if clustered_new_nodes[x].coordinate is None:
                    while y < world_size_new and assigned_coordinate[y]:
                        y += 1
                    if y == world_size_new:
                        break
                    clustered_new_nodes[x].strategy = new_s
                    clustered_new_nodes[x].coordinate = y // (new_s[1] * new_s[2]) , y % new_s[1] , (y // new_s[1]) % new_s[2]
                    y += 1
                    # if verbose:
                    timestamp('do_switch', f"linked {clustered_new_nodes[x]}(rank {clustered_new_nodes[x].pc_rank}) to {clustered_new_nodes[x].coordinate}")
                x += 1
        elif km_match and group_tp and new_s[1] > 0 and len(intersect_nodes) > 0:
            self.tp_group_assign(old_s, old_nodes, new_s, new_nodes, replicas_stay, intersect_nodes, buffer_param=buffer_param, verbose=verbose)
            new_nodes = list(filter(lambda x: x.coordinate is not None, new_nodes))
        elif km_match and len(intersect_nodes) > 0:
            assigned_coordinate = [False] * world_size_new
            # build graph weights matrix
            graph_weights = np.zeros((len(intersect_nodes), world_size_new), dtype=np.float64)
            for i in range(graph_weights.shape[0]):
                for j in range(graph_weights.shape[1]):
                    # map new replica id to old replica id
                    new_stages = replicas_stay[j // (new_s[1] * new_s[2])] , (j // new_s[2]) % new_s[1] , j % new_s[2]
                    rank = intersect_nodes[i].pc_rank

                    buff_config = None
                    if switch_buffer and new_stages[0] < old_s[0] and buffer_param[new_stages[0]] is not None:
                        # if batchxbeam is unequal, it must be in diffrent replicas, and return 0 immediately
                        # So here, buffer config is simply set to `new` version, since old & new are the same
                        buff_config = buffer_param[new_stages[0]]['batchxbeam'], buffer_param[new_stages[0]]['session_len'], \
                            buffer_param[new_stages[0]]['mem_len'], buffer_param[new_stages[0]]['beam']

                    graph_weights[i, j] = self._get_graph_weights(intersect_nodes[i].prev_coordinate, intersect_nodes[i].prev_strategy, \
                            new_stages, new_s, buff_config)
            if verbose:
                print(graph_weights)

            # Transpose weight matrix if dim[0] > dim[1]
            is_transposed = False
            if graph_weights.shape[0] > graph_weights.shape[1]:
                is_transposed = True
                graph_weights = graph_weights.T

            # Run KM algorirhm to determine placement
            matcher = KMMatcher(graph_weights)
            matcher.solve()

            if is_transposed:
                # x is world_size_new, y is intersect_nodes, when avalible #vnodes is greater than new world size
                for x in range(matcher.n):
                    y = int(matcher.xy[x])
                    intersect_nodes[y].strategy = new_s
                    intersect_nodes[y].coordinate = x // (new_s[1] * new_s[2]) , (x // new_s[2]) % new_s[1] , x % new_s[2]
                    assigned_coordinate[x] = True
                    timestamp('do_switch', f"matched {intersect_nodes[y]}(rank {intersect_nodes[y].pc_rank}) as {intersect_nodes[y].coordinate}, weight={matcher.weights[x, y]}")
            else:
                # x is intersect_nodes, y is world_size_new
                for x in range(matcher.n):
                    y = int(matcher.xy[x])
                    intersect_nodes[x].strategy = new_s
                    intersect_nodes[x].coordinate = y // (new_s[1] * new_s[2]) , (y // new_s[2]) % new_s[1] , y % new_s[2]
                    assigned_coordinate[y] = True
                    # if verbose:
                    timestamp('do_switch', f"matched {intersect_nodes[x]}(rank {intersect_nodes[x].pc_rank}) as {intersect_nodes[x].coordinate}, weight={matcher.weights[x, y]}")

            # casually assign remaining coord, this will occur when #intersect_nodes < #new_nodes
            x, y = 0, 0
            while x < len(new_nodes):
                if new_nodes[x].coordinate is None:
                    while y < world_size_new and assigned_coordinate[y]:
                        y += 1
                    if y == world_size_new: # all coordinate assigned
                        new_nodes = list(filter(lambda x: x.coordinate is not None, new_nodes))
                        break
                    new_nodes[x].strategy = new_s
                    new_nodes[x].coordinate = y // (new_s[1] * new_s[2]) , (y // new_s[2]) % new_s[1] , y % new_s[2]
                    y += 1
                    # if verbose:
                    timestamp('do_switch', f"linked {new_nodes[x]}(rank {new_nodes[x].pc_rank}) to {new_nodes[x].coordinate}")
                x += 1
        elif group_tp:
            # plain match
            assigned_coordinate = [False] * world_size_new
            x, y = 0, 0
            clustered_new_nodes = sorted(new_nodes, key=lambda x: x.ip)
            clustered_new_nodes = clustered_new_nodes[:world_size_new]
            while x < world_size_new:
                if clustered_new_nodes[x].coordinate is None:
                    while assigned_coordinate[y]:
                        y += 1
                    clustered_new_nodes[x].strategy = new_s
                    clustered_new_nodes[x].coordinate = y // (new_s[1] * new_s[2]) , y % new_s[1] , (y // new_s[1]) % new_s[2]
                    y += 1
                    # if verbose:
                    timestamp('do_switch', f"linked {clustered_new_nodes[x]}(rank {clustered_new_nodes[x].pc_rank}) to {clustered_new_nodes[x].coordinate}")
                x += 1
        else:
            # baseline: random assign
            random_assign_baseline = True
            # shuffled_new_nodes = list(new_nodes)
            # random.shuffle(shuffled_new_nodes)
            shuffled_new_nodes = sorted(new_nodes, key=lambda x: x.pc_rank)
            for i in range(world_size_new):
                shuffled_new_nodes[i].strategy = new_s
                shuffled_new_nodes[i].coordinate =  i // (new_s[1] * new_s[2]) , i % new_s[1] , (i // new_s[1]) % new_s[2]
                timestamp('do_switch', f"Orderd by rank: linked {shuffled_new_nodes[i]}(rank {shuffled_new_nodes[i].pc_rank}) to {shuffled_new_nodes[i].coordinate}")


        new_nodes = list(filter(lambda x: x.coordinate is not None, new_nodes))

        # Analyse where are the layers on old ranks
        layer_where_old = [dict() for _ in range(old_s[0])]
        rank_to_replica_old = dict()
        for vnode in old_nodes:
            pc_rank = vnode.pc_rank
            dp_rank = vnode.prev_coordinate[0] # if switch_buffer else 0
            rank_to_replica_old[pc_rank] = dp_rank
            tp_rank = vnode.prev_coordinate[1]
            pp_rank = vnode.prev_coordinate[2]

            layer_where_old_replica_p = layer_where_old[dp_rank]

            for layer_encoded_id in self._layers_on_node(tp_rank, pp_rank, old_s, tp_grain, False):
                if layer_encoded_id in layer_where_old_replica_p:
                    layer_where_old_replica_p[layer_encoded_id].append(pc_rank)
                else:
                    layer_where_old_replica_p[layer_encoded_id] = [pc_rank]

        # match new vnodes to old replica id
        # priority: dp < tp < pp
        # undying_replicas = list(range(old_s[0]))
        # for rank in dying_ranks:
        #     prev_replica = rank2vnode[rank].prev_coordinate[0]
        #     if prev_replica in undying_replicas:
        #         undying_replicas.remove(prev_replica)

        sorted_new_nodes = sorted(new_nodes, key=lambda x: x.coordinate[2]*(world_size_new**2)+x.coordinate[1]*world_size_new+x.coordinate[0])
        cnt = 0
        for vnode in sorted_new_nodes:
            if not vnode.prev_coordinate is None:
                continue
            vnode.prev_coordinate = (cnt % old_s[0], 0, 0)
            # i = len(undying_replicas)
            # if i == 0:
            #     vnode.prev_coordinate = (cnt % old_s[0], 0, 0)
            # else:
            #     vnode.prev_coordinate = (undying_replicas[cnt % i], 0, 0)
            cnt += 1

        # Analyse where are the layers on new ranks
        layer_where_new_b = [dict() for _ in range(old_s[0])]
        layer_where_new_p = [dict() for _ in range(old_s[0])]
        rank_to_replica_new = dict()
        for vnode in new_nodes:
            pc_rank = vnode.pc_rank
            dp_rank = vnode.coordinate[0] if switch_buffer else 0
            if dp_rank < old_s[0]:
                rank_to_replica_new[pc_rank] = replicas_stay[dp_rank] # map to oldl
            layer_where_new_replica_p = layer_where_new_p[vnode.prev_coordinate[0] if not random_assign_baseline else 0]

            tp_rank = vnode.coordinate[1]
            pp_rank = vnode.coordinate[2]

            if switch_buffer and dp_rank < old_s[0] and buffer_param[replicas_stay[dp_rank]] is not None:
                layer_where_new_replica_b = layer_where_new_b[replicas_stay[dp_rank]]
                for layer_encoded_id in self._layers_on_node(tp_rank, pp_rank, new_s, tp_grain, True):
                    if layer_encoded_id in layer_where_new_replica_b:
                        layer_where_new_replica_b[layer_encoded_id].append(pc_rank)
                    else:
                        layer_where_new_replica_b[layer_encoded_id] = [pc_rank]

            for layer_encoded_id in self._layers_on_node(tp_rank, pp_rank, new_s, tp_grain, False):
                if layer_encoded_id in layer_where_new_replica_p:
                    layer_where_new_replica_p[layer_encoded_id].append(pc_rank)
                else:
                    layer_where_new_replica_p[layer_encoded_id] = [pc_rank]

        switch_dict = {}

        # calc memory efficient order
        trans_order = None
        if memory_tolarance >= 0:
            trans_order, has_violate = self._get_memefi_layer_order(intersect_nodes, old_s, new_s, memory_tolarance)
            if has_violate == False:
                timestamp('do_switch', f'original order has no violation within memory limitation, do as usual')

        # Do polling in every replica id independently
        for replica_id in range(2 * old_s[0] if switch_buffer else old_s[0]):
            # 0 ~ old_s[0]-1        : param
            # old_s[0] ~ 2*old_s[0] : buffer

            if replica_id < old_s[0]:
                # params
                prefix = 'p'
                layer_where_old_dict = layer_where_old[replica_id]
                layer_where_new_dict = layer_where_new_p[replica_id]
                if verbose:
                    print(layer_where_old_dict)
                    print(layer_where_new_dict)
            else:
                # buffers
                prefix = 'b'
                layer_where_old_dict = layer_where_old[replica_id - old_s[0]]
                layer_where_new_dict = layer_where_new_b[replica_id - old_s[0]]
                if verbose:
                    print(layer_where_old_dict)
                    print(layer_where_new_dict)
                if len(layer_where_new_dict) == 0:
                    continue

            for layer_id in layer_where_old_dict:
                old_ranks = layer_where_old_dict[layer_id]
                if not layer_id in layer_where_new_dict:
                    new_ranks = []
                else:
                    new_ranks = layer_where_new_dict[layer_id]

                send_ranks = [i for i in new_ranks if i not in old_ranks]
                stay_ranks = [i for i in old_ranks if i in new_ranks]
                del_ranks = [i for i in old_ranks if i not in new_ranks]

                for i, dst_rank in enumerate(send_ranks):
                    src = old_ranks[i % len(old_ranks) if not random_assign_baseline else 0] # Polling
                    prior = src in dying_ranks or dst_rank in dying_ranks
                    _insert_entry(switch_dict, src, prefix+'send', layer_id, dst_rank, prior)
                    _insert_entry(switch_dict, dst_rank, prefix+'recv', layer_id, src, prior)
                    if trans_order is not None or prefix == 'b' or prior:
                        src_node, dst_node = rank2vnode[src], rank2vnode[dst_rank]
                        buff_config = buffer_param[src_node.prev_coordinate[0]] if prefix == 'b' else None
                        comm = self._get_paramsize_by_layerid(layer_id, buff_config) # 1e-6*param_num
                        comm *= 32 #Mb
                        if src_node.ip != dst_node.ip:
                            self.ip2comm[src_node.ip][0] += comm
                            self.ip2comm[dst_node.ip][0] += comm
                        else:
                            self.ip2comm[src_node.ip][1] += comm

                for del_rank in del_ranks:
                    _insert_entry(switch_dict, del_rank, prefix+'stay', None)
                for stay_rank in stay_ranks:
                    _insert_entry(switch_dict, stay_rank, prefix+'stay', layer_id)

        if verbose:
            info = "buffer + dying_ranks" if trans_order is None else "overall"
            print(f"Communication({info}) on each node (Mb)", self.ip2comm)

                # trans_order = None

        for rank in switch_dict:
            n_rank = int(rank)
            switch_dict[rank]['tp_grain'] = tp_grain
            switch_dict[rank]['tp'] = new_s[1]
            # switch_dict[rank]['first_undying'] = first_undying_layer[n_rank] if n_rank in first_undying_layer else self.layer_num + 2
            if trans_order is not None:
                switch_dict[rank]['order'] = trans_order
            if switch_buffer:
                switch_dict[rank]['buffer'] = {}
                if n_rank in rank_to_replica_old and buffer_param[rank_to_replica_old[n_rank]] is not None:
                    switch_dict[rank]['buffer']['old'] = buffer_param[rank_to_replica_old[n_rank]]
                if n_rank in rank_to_replica_new and buffer_param[rank_to_replica_new[n_rank]] is not None:
                    switch_dict[rank]['buffer']['new'] = buffer_param[rank_to_replica_new[n_rank]]
            if enable_no_lock:
                cond1 = 'precv' not in switch_dict[rank] or len(switch_dict[rank]['precv'])==0
                # cond2 = 'brecv' not in switch_dict[rank] or len(switch_dict[rank]['brecv'])==0
                cond3 = old_s[1] == new_s[1] and old_s[2] == new_s[2]
                if cond1 and cond3:
                    switch_dict[rank]['nolock'] = True
            if self.dont_stop_ft:
                switch_dict[rank]['ignore_buff'] = True
            if slpt >= 0:
                switch_dict[rank]['sleep'] = 0.0


        timestamp('do_switch', f'switching from {old_s} to {new_s}: computing json end')
        if verbose:
            print(json.dumps(switch_dict), file=json_out_file)
        return switch_dict

    def estimate_last_switch_transfer_time(self):
        # TODO
        inter_node_bd = 50000 # Mbps
        intra_node_bd = 50000
        factor = 1
        max_inter_node = max(map(lambda x: x[0], self.ip2comm.values()))
        max_intra_node = max(map(lambda x: x[1], self.ip2comm.values()))
        t1 = max_inter_node / inter_node_bd
        t2 = max_intra_node / intra_node_bd
        return (t1 + t2) * factor

    def _get_memefi_layer_order(self, intersect_nodes, old_s, new_s, tolerance=0.):
        layers_per_pp_stage_new = self.layer_num // new_s[2]
        layers_per_pp_stage_old = self.layer_num // old_s[2]

        layer_tol_n = max(tolerance * layers_per_pp_stage_old / old_s[1], 0)

        placed = [False] * self.layer_num
        current_storage = [0] * len(intersect_nodes)

        res = []

        layer_in_old = lambda l, vnode: 1/old_s[1] if vnode.prev_coordinate is not None and l // layers_per_pp_stage_old == vnode.prev_coordinate[2] else 0
        layer_in_new = lambda l, vnode: 1/new_s[1] if vnode.coordinate is not None and l // layers_per_pp_stage_new == vnode.coordinate[2] else 0
        get_bias = lambda l, vnode: layer_in_new(l, vnode) - layer_in_old(l, vnode)

        has_violate = False

        for _ in range(self.layer_num):
            for ii in range(self.layer_num):
                if placed[ii]:
                    continue
                for j, vnode in enumerate(intersect_nodes):
                    bias = get_bias(ii, vnode)
                    if current_storage[j] + bias > layer_tol_n:
                        has_violate = True
                        break
                else:
                    # for loop normally returned, ii is feasible
                    res.append(ii)
                    placed[ii] = True
                    for j, vnode in enumerate(intersect_nodes):
                        bias = get_bias(ii, vnode)
                        current_storage[j] += bias
                    break
            else:
                # for loop normally returned, no ii is feasible
                # find minmax
                minmax_stoage = self.layer_num + 1
                arg_minmax = -1

                max_st = max(current_storage)
                is_max_nodes = [False] * len(current_storage)
                for n_idx in filter(lambda x: math.isclose(current_storage[x], max_st), range(len(current_storage))):
                    is_max_nodes[n_idx] = True

                prior_layers = [0] * self.layer_num
                # print(is_max_nodes)
                for ii in range(self.layer_num):
                    for j, vnode in enumerate(intersect_nodes):
                        bias = get_bias(ii, vnode)
                        if is_max_nodes[j] and bias < 0:
                            prior_layers[ii] = 1
                            break
                # print(prior_layers)

                loop_layers_order = sorted(range(self.layer_num), key=lambda x: -prior_layers[x])
                # print(loop_layers_order)

                # sorted is stable
                # for ii in range(self.layer_num):
                for ii in loop_layers_order:
                    if placed[ii]:
                        continue
                    max_storage = 0
                    for j, vnode in enumerate(intersect_nodes):
                        bias = get_bias(ii, vnode)
                        if current_storage[j] + bias > max_storage:
                            max_storage = current_storage[j] + bias

                    if minmax_stoage > max_storage:
                        arg_minmax = ii
                        minmax_stoage = max_storage

                timestamp('memefi_order', f'cannot within tolerance={layer_tol_n}, get minmax={minmax_stoage}')
                res.append(arg_minmax)
                placed[arg_minmax] = True
                for j, vnode in enumerate(intersect_nodes):
                    bias = get_bias(arg_minmax, vnode)
                    current_storage[j] += bias

        res = [-self.MAGIC_LAYER_NUM-1] + res + [-self.MAGIC_POST_DECODER_LAYER_ID-1]
        timestamp('memefi_order', f'final order: {res}')
        return res, has_violate


    def _get_graph_weights(self, old_stages, old_s, new_stages, new_s, buff_config=None):
        # model_param_config: list [layer_num, vocab_size, hidden_size, max_seq_len]
        # buff_config: list[batchxbeam, sesstion_len, mem_len, beam]
        # (dp, tp, pp), nagetive dp stage means ignoring replica id
        # replica id = dp_stage

        # check dp stage, return 0 if not equal
        # if (not buff_config is None) and old_stages[0] > 0 and old_stages[0] != new_stages[0]:
        #     return 0
        if buff_config is None or (old_stages[0] != new_stages[0]):
            # replica id not match, set buffer to 0
            buff_config = 0, 0, 0, 0
        batchxbeam, sesstion_len, mem_len, beam = buff_config
        cache_indirection = 2 * mem_len if beam > 1 else 0

        # all weights are divided by hidden_size to avoid giant number
        global_dp_params = (self.vocab_size+self.max_seq_len) + (sesstion_len+3*self.hidden_size+4+cache_indirection)*batchxbeam/self.hidden_size
        if old_stages[2] == old_s[2] - 1 and new_stages[2] == new_s[2] - 1:
            global_dp_params += self.vocab_size+2  # final layernorm+embedding
        layer_dp_params = 6
        layer_tp_params = 12*self.hidden_size + 7 + batchxbeam*mem_len

        # check pp stage, get intersection layer num
        layer_num_per_stage = self.layer_num // old_s[2], self.layer_num // new_s[2]
        l = layer_num_per_stage[0] * old_stages[2], layer_num_per_stage[1] * new_stages[2]
        r = l[0] + layer_num_per_stage[0], l[1] + layer_num_per_stage[1]
        arg_max = 0 if l[0] > l[1] else 1
        layer_intersection = max(0, min(r[0], r[1]) - l[arg_max])

        # check tp stage, get intersection fraction
        layer_num_per_stage = 1 / old_s[1], 1 / new_s[1]
        l = layer_num_per_stage[0] * old_stages[1], layer_num_per_stage[1] * new_stages[1]
        r = l[0] + layer_num_per_stage[0], l[1] + layer_num_per_stage[1]
        arg_max = 0 if l[0] > l[1] else 1
        hidden_intersection = max(0, min(r[0], r[1]) - l[arg_max])

        # print(layer_intersection, hidden_intersection)

        return layer_intersection * (layer_dp_params + layer_tp_params * hidden_intersection) + global_dp_params

    def _get_paramsize_by_layerid(self, layer_id, buff_config):
        if buff_config is None:
            # param
            if layer_id == self.MAGIC_LAYER_NUM: #pre_decoder
                return 1e-6 * (self.vocab_size+self.max_seq_len) * self.hidden_size
            elif layer_id == self.MAGIC_POST_DECODER_LAYER_ID:
                return 1e-6 * (self.vocab_size+2) * self.hidden_size
            elif layer_id < 0: # layer dp param
                return 1e-6 * 6 * self.hidden_size
            else: # layer tp param
                _, _, tp_grain = _layer_decode(layer_id)
                return 1e-6*(12*self.hidden_size + 7)*self.hidden_size / tp_grain
        else:
            cache_indirection = 2 * buff_config['mem_len'] if buff_config['beam'] > 1 else 0
            if layer_id == self.MAGIC_LAYER_NUM: # dp buffer
                return 1e-6 * (buff_config['session_len']+3*self.hidden_size+4+cache_indirection)*buff_config['batchxbeam']
            elif layer_id >= 0: # layer tp buffer
                return 1e-6 * buff_config['batchxbeam'] * buff_config['mem_len'] * self.hidden_size
        return 0



    def _layers_on_node(self, tp_stage, pp_stage, s, tp_grain, is_buffer):
        if tp_grain < s[1]:
            tp_grain = s[1]

        pp_layers_per_stage = self.layer_num // s[2]
        layer_list = [pp_stage * pp_layers_per_stage + i for i in range(pp_layers_per_stage)]

        return_list = []
        factor = tp_grain // s[1]
        for layer_id in layer_list:
            return_list += [_layer_encode(layer_id, tp_stage * factor + i, tp_grain) for i in range(factor)]

        if is_buffer == False: # buffer no dp param
            return_list += [-i-1 for i in layer_list] # dp params, no replica id

        # no buffer on post_decoder_layer
        tail_list = [] if (pp_stage != s[2] - 1 or is_buffer) else [self.MAGIC_POST_DECODER_LAYER_ID]

        return [self.MAGIC_LAYER_NUM] + return_list + tail_list

if __name__ == '__main__':
    swicher = Switch()
    world_size = 16
    gpu_per_node = 2
    ip = '127.0.0.%d'

    vnodes = [VNode(ip%((i//gpu_per_node)), i%gpu_per_node) for i in range(world_size)]
    for i, vnode in enumerate(vnodes):
        vnode.pc_rank = i

    old_s = (2, 1, 2)
    # old_s = (2, 1, 1)
    old_nodes = []

    ws_old = old_s[0] * old_s[1] * old_s[2]
    old_node_map = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
    for i in range(ws_old):
        idx = old_node_map[i]
        vnodes[idx].prev_strategy = old_s
        vnodes[idx].prev_coordinate = i // (old_s[1] * old_s[2]), (i // old_s[2]) % old_s[1], i % old_s[2]
        old_nodes.append(vnodes[idx])

    new_s = (4, 1, 2)
    # new_s = (1, 2, 4)
    new_nodes = [vnodes[0], vnodes[1], vnodes[2], vnodes[3], vnodes[8], vnodes[9], vnodes[10], vnodes[11]]
    # new_nodes = vnodes[:(new_s[0]*new_s[1]*new_s[2])]

    buffer_dict1 = {
        'batchxbeam': 8,
        'beam' : 1,         # batch = batchxbeam // beam
        'mem_len': 40,      # = input_len + output_len
        'session_len': 40,  # = mem_len
        'new_layers_per_device': 32 // new_s[2]
    }

    buffer_dict2 = {
        'batchxbeam': 4,
        'beam' : 1,         # batch = batchxbeam // beam
        'mem_len': 40,      # = input_len + output_len
        'session_len': 40,  # = mem_len
        'new_layers_per_device': 32 // new_s[2]
    }

    buffer_param = [buffer_dict1, buffer_dict2, buffer_dict2, buffer_dict1]

    # replica_stay = [1]
    replica_stay = list(range(new_s[0]))
    swicher.do_switch(old_s, old_nodes, new_s, new_nodes, replica_stay, buffer_param[:old_s[0]], km_match=False, group_tp=False, verbose=True, memory_tolarance=0., try_dont_stop_ft=False)
    print("Estimation time:", swicher.estimate_last_switch_transfer_time())
