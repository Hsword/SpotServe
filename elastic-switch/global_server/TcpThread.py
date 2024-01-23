import threading
import struct
from util.util import TcpAgent, timestamp

class TcpThread(threading.Thread):
    def __init__(self, tcp_server, agent_dict) -> None:
        super().__init__()
        self.tcp_server = tcp_server
        self.agent_dict = agent_dict

    def run(self):
        timestamp('gs TcpThread', 'listening')
        while True:
            conn, address = self.tcp_server.accept()
            agent = TcpAgent(conn, address)

            data = agent.recv(4)
            rank, = struct.unpack('I', data)
            timestamp('gs', f'accepted from pc rank {rank}')

            self.agent_dict[str(rank)] = agent
