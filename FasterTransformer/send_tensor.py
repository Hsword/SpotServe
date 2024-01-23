import struct
import time
import torch
import traceback
import socket

address = 'localhost'
port = 10040

test_tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]]).cuda()
t = test_tensor.storage()._share_cuda_()

print(t[1:4])
print(len(t[1]))

# del test_tensor
# exit(0)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((address, port))
sock.listen(1)
print('listenning')

while True:
    conn, addr = sock.accept()
    print('accepted')

    '''
    (storage_device, storage_handle, storage_size_bytes, storage_offset_bytes,
    ref_counter_handle, ref_counter_offset, event_handle, event_sync_required)
    '''
    metadata = struct.pack(f'64sqq', *t[1:4])
    print(metadata)
    conn.sendall(metadata)
    conn.close()

    