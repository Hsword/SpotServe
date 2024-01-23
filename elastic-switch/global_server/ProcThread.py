import threading
import json
from HotSwitch import simple_switch

class ProcThread(threading.Thread):
    def __init__(self, agent_list, init_strat=(1,1,1)) -> None:
        super().__init__()
        self.agent_list = agent_list
        self.strat = init_strat
    
    def run(self):
        while True:
            s = input()
            if s == 'quit':
                break

            new_s = map(int, s.split())

            switch_dict = simple_switch(self.strat, new_s)

            for k in switch_dict:
                self.agent_list[k].send_string(json.dumps(switch_dict[k]))



