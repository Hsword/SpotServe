import asyncio
import sys
import time
import socket
import struct
from datetime import datetime

def timestamp(name, stage):
    print(f'[{datetime.now()}] {name}, {stage}, {time.time():.3f}', file=sys.stderr)

class TcpServer():
    def __init__(self, address, port, blocking=True):
        self.address = address
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.address, self.port))
        self.sock.listen(1)
        self.sock.setblocking(blocking)

    def __del__(self):
        self.sock.close()

    def accept(self):
        conn, address = self.sock.accept()
        return conn, address

    async def async_accept(self):
        loop = asyncio.get_running_loop()
        conn, address = await loop.sock_accept(self.sock)
        return conn, address


class TcpAgent:
    def __init__(self, conn, address=None, blocking=True):
        self.conn = conn
        self.address = address
        self.conn.setblocking(blocking)

    def __del__(self):
        self.conn.close()

    def send(self, msg):
        self.conn.sendall(msg)

    def recv(self, msg_len):
        return self.conn.recv(msg_len, socket.MSG_WAITALL)

    async def async_send(self, msg):
        while msg:
            try:
                sent = self.conn.send(msg)
            except socket.error as e:
                if e.errno == socket.errno.EWOULDBLOCK:
                    await asyncio.sleep(0)
                    continue
                raise
            msg = msg[sent:]

    async def async_recv(self, msg_len):
        msg = b''
        while len(msg) < msg_len:
            try:
                chunk = self.conn.recv(msg_len - len(msg))
            except socket.error as e:
                if e.errno == socket.errno.EWOULDBLOCK:
                    await asyncio.sleep(0)
                    continue
                raise
            if not chunk:
                raise Exception('Connection closed by remote end')
            msg += chunk
        return msg

    def send_string(self, s):
        data = s.encode('utf-8')
        l = len(data)
        self.conn.sendall(struct.pack('I', l))
        self.conn.sendall(data)

    def recv_string(self):
        l = self.recv(4)
        l, = struct.unpack('I', l)
        data = self.recv(l)
        return data.decode('utf-8')


    def settimeout(self, t):
        self.conn.settimeout(t)


class TcpClient(TcpAgent):
    def __init__(self, address, port):
        super().__init__(None)
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((address, port))
