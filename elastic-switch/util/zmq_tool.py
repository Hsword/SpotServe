# Deprecated file

import zmq

class ZMQAgent:
    def __init__(self):
        self.socket = None
    
    def __del__(self):
        self.socket.close()

    def send_string(self, msg):
        self.socket.send_string(msg)

    def recv_string(self):
        return self.socket.recv_string()

    def send_pyobj(self, msg):
        self.socket.send_pyobj(msg)

    def recv_pyobj(self):
        return self.socket.recv_pyobj()


class ZMQServer(ZMQAgent):
    def __init__(self, address, port, method=zmq.REP):
        super().__init__()
        self.address = address
        self.port = port

        self.context = zmq.Context()
        self.socket = self.context.socket(method)
        self.socket.bind(f"tcp://*:{port}")

class ZMQClient(ZMQAgent):
    def __init__(self, address, port, method=zmq.REQ):
        super().__init__()
        self.address = address
        self.port = port

        self.context = zmq.Context()
        self.socket = self.context.socket(method)
        self.socket.connect(f"tcp://{address}:{port}")