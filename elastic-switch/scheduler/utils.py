import json
from multiprocessing import shared_memory


def encode_tuple(state, int_tup, shm: shared_memory.SharedMemory):
    if state < 0: # init state
        tup_str = b'x'
    else:
        assert state in [0, 1]
        tup_str = str(state).encode()
    tup_str += ','.join(map(str, int_tup)).encode()
    tup_str += b' ' * (shm.size - len(tup_str))
    return tup_str


def decode_tuple(shm: shared_memory.SharedMemory):
    tup_str = bytes(shm.buf[:]).decode()
    state = -1 if tup_str[0] == 'x' else int(tup_str[0])
    strategy = tuple(map(int, tup_str[1:].strip().split(',')))
    return state, strategy


def clean_shared_mem(shm_name):
    try:
        shm = shared_memory.SharedMemory(shm_name)
        shm.close()
        shm.unlink()
    except:
        pass


def dump_shared_mem(data, shm_name):
    data = json.dumps(data).encode('utf-8')
    try:
        shm = shared_memory.SharedMemory(shm_name)
        shm.close()
        shm.unlink()
    except:
        pass
    shm = shared_memory.SharedMemory(shm_name, create=True, size=len(data))
    shm.buf[:] = data
    shm.close()


def load_shared_mem(shm_name):
    try:
        shm = shared_memory.SharedMemory(shm_name)
        data = json.loads(bytes(shm.buf[:]).decode('utf-8'))
        shm.close()
    except:
        data = None
    return data
