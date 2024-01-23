MAX_LAYER = None

MAGIC_LAYER_NUM = None
MAGIC_POST_DECODER_LAYER_ID = None

def set_num_layer(num):
    global MAX_LAYER, MAGIC_LAYER_NUM, MAGIC_POST_DECODER_LAYER_ID
    MAX_LAYER = num
    MAGIC_LAYER_NUM = -MAX_LAYER - 1
    MAGIC_POST_DECODER_LAYER_ID = -MAX_LAYER - 2


def ceil_div(a, b):
    return (a + b - 1) // b

def insert_entry(dct, rank, mode, layer_id,to=None, tag=None):
    rank = str(rank)
    if rank not in dct:
        dct[rank] = {"stay":[], "send":[], "recv":[]}
    if mode not in dct[rank]:
        dct[rank][mode] = []
    if layer_id is None:
        return
    if to is None:
        dct[rank][mode].append(layer_id)
    else:
        dct[rank][mode].append({"layer":layer_id, "pair":to})
        # dct[rank][mode].append({"layer":layer_id, "pair":to, "tag":tag})

_ENCODE_BASE_ = 128
def layer_encode(layer_id, tp_stage, tp_grain):
    tp_grain = max(0, tp_grain - 1)
    return tp_grain * _ENCODE_BASE_ * _ENCODE_BASE_ + tp_stage * _ENCODE_BASE_ + layer_id

def layer_decode(layer_encode_id):
    '''
    return: layer_id, tp_stage, tp_grain
    '''
    layer_id = layer_encode_id % _ENCODE_BASE_
    tmp = layer_encode_id // _ENCODE_BASE_
    tp_stage = tmp % _ENCODE_BASE_
    tp_grain = tmp // _ENCODE_BASE_ + 1
    return layer_id, tp_stage, tp_grain


def tp_stage_of_rank(rank, s):
    world_size = s[0] * s[1] * s[2]
    rank_dp = rank % (world_size // s[0])
    tp_stage = rank_dp % s[1]
    return tp_stage

def layers_on_rank(rank, s, tp_grain):
    # (dp, tp, pp)
    # 12 layers
    N = MAX_LAYER #24

    if tp_grain < s[1]:
        tp_grain = s[1]

    world_size = s[0] * s[1] * s[2]

    rank = rank % (s[1] * s[2]) # map to dp
    pp_stage = rank // s[1]
    tp_stage = rank % s[1]

    # pp_stage = rank // (s[0] * s[1])
    # rank_pp = rank % (s[0] * s[1])
    # tp_stage = rank_pp % s[1] # tp_consec
    '''
    rank_dp = rank % (world_size // s[0])
    pp_stage = rank_dp // s[1]
    tp_stage = rank_dp % s[1]
    '''

    pp_layers_per_stage = N // s[2]
    layer_list = [pp_stage * pp_layers_per_stage + i for i in range(pp_layers_per_stage)]
    # layer_list = [i * s[1] * s[2] + j for i in range(s[0]) for j in layer_list]

    return_list = []
    factor = tp_grain // s[1]
    for layer_id in layer_list:
        return_list += [layer_encode(layer_id, tp_stage * factor + i, tp_grain) for i in range(factor)]
    return_list += [-i-1 for i in layer_list] # layer norm

    return [MAGIC_LAYER_NUM] + return_list + ([] if pp_stage != s[2] - 1 else [MAGIC_POST_DECODER_LAYER_ID])


def layers_on_node(tp_stage, pp_stage, s, tp_grain):
    # number of layers
    N = MAX_LAYER

    if tp_grain < s[1]:
        tp_grain = s[1]

    pp_layers_per_stage = N // s[2]
    layer_list = [pp_stage * pp_layers_per_stage + i for i in range(pp_layers_per_stage)]
    # layer_list = [i * s[1] * s[2] + j for i in range(s[0]) for j in layer_list]

    return_list = []
    factor = tp_grain // s[1]
    for layer_id in layer_list:
        return_list += [layer_encode(layer_id, tp_stage * factor + i, tp_grain) for i in range(factor)]
    return_list += [-i-1 for i in layer_list] # layer norm

    return [MAGIC_LAYER_NUM] + return_list + ([] if pp_stage != s[2] - 1 else [MAGIC_POST_DECODER_LAYER_ID])


def simple_switch(old_s, old_nodes, new_s, new_nodes, shitty_balance=False, buffer_param=None):
    switch_dict = {}
    tp_grain = max(old_s[1], new_s[1])

    world_size_old = old_s[0] * old_s[1] * old_s[2]
    world_size_new = new_s[0] * new_s[1] * new_s[2]

    layer_where_old = {}
    for vnode in old_nodes:
        pc_rank = vnode.pc_rank
        tp_rank = vnode.prev_coordinate[1]
        pp_rank = vnode.prev_coordinate[2]
        for layer_encoded_id in layers_on_node(tp_rank, pp_rank, old_s, tp_grain):
            if layer_encoded_id in layer_where_old:
                layer_where_old[layer_encoded_id].append(pc_rank)
            else:
                layer_where_old[layer_encoded_id] = [pc_rank]

    layer_where_new = {}
    for vnode in new_nodes:
        pc_rank = vnode.pc_rank
        tp_rank = vnode.coordinate[1]
        pp_rank = vnode.coordinate[2]
        for layer_encoded_id in layers_on_node(tp_rank, pp_rank, new_s, tp_grain):
            if layer_encoded_id in layer_where_new:
                layer_where_new[layer_encoded_id].append(pc_rank)
            else:
                layer_where_new[layer_encoded_id] = [pc_rank]

    '''
    # (dp, tp, pp) 12 layer
    def layer_where(id, s):
        fac = s[1] * s[2]
        layer_per_stage = 12 // s[2]
        return [id //layer_per_stage + i * fac for i in range(s[0])]
    '''


    tag = 1010
    for layer_id in layer_where_old:
        old_ranks = layer_where_old[layer_id]
        new_ranks = layer_where_new[layer_id]


        send_ranks = [i for i in new_ranks if i not in old_ranks]
        stay_ranks = [i for i in old_ranks if i in new_ranks]
        del_ranks = [i for i in old_ranks if i not in new_ranks]

        for i, dst_rank in enumerate(send_ranks):
            src = old_ranks[0 if shitty_balance else i % len(old_ranks)]
            insert_entry(switch_dict, src,  'send', layer_id, dst_rank, tag)
            insert_entry(switch_dict, dst_rank, 'recv', layer_id, src, tag)
            has_entry = True
            tag += _ENCODE_BASE_

        for del_rank in del_ranks:
            insert_entry(switch_dict, del_rank, 'stay', None)
        for stay_rank in stay_ranks:
            insert_entry(switch_dict, stay_rank, 'stay', layer_id)

    for rank in switch_dict:
        switch_dict[rank]['tp_grain'] = tp_grain
        switch_dict[rank]['tp'] = new_s[1]
        if len(new_s) > 3:
            switch_dict[rank]['buffer'] = buffer_param

    return switch_dict

if __name__ == '__main__':
    print(layers_on_rank(1, (2, 2, 2), 1))

    s = input('old strategy:')
    old_s = list(map(int, s.split()))
    s = input('new strategy:')
    new_s = list(map(int, s.split()))

    print(simple_switch(old_s, new_s))
