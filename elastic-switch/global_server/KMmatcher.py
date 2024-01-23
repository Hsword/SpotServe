'''
reference: 
- https://www.topcoder.com/community/competitive-programming/tutorials/assignment-problem-and-hungarian-algorithm/
- https://github.com/mayorx/hungarian-algorithm
'''

import numpy as np

#max weight assignment
class KMMatcher:

    ## weights : nxm weight matrix (numpy , float), n <= m
    def __init__(self, weights):
        weights = np.array(weights).astype(np.float64)
        self.weights = weights
        self.n, self.m = weights.shape
        assert self.n <= self.m
        # init label
        self.label_x = np.max(weights, axis=1)
        self.label_y = np.zeros((self.m, ), dtype=np.float64)

        self.max_match = 0
        self.xy = -np.ones((self.n,), dtype=np.int32)
        self.yx = -np.ones((self.m,), dtype=np.int32)

    def do_augment(self, x, y):
        self.max_match += 1
        while x != -2:
            self.yx[y] = x
            ty = self.xy[x]
            self.xy[x] = y
            x, y = self.prev[x], ty

    def find_augment_path(self):
        self.S = np.zeros((self.n,), bool)
        self.T = np.zeros((self.m,), bool)

        self.slack = np.zeros((self.m,), dtype=np.float64)
        self.slackyx = -np.ones((self.m,), dtype=np.int32)  # l[slackyx[y]] + l[y] - w[slackx[y], y] == slack[y]

        self.prev = -np.ones((self.n,), np.int32)

        queue, st = [], 0
        root = -1

        for x in range(self.n):
            if self.xy[x] == -1:
                queue.append(x);
                root = x
                self.prev[x] = -2
                self.S[x] = True
                break

        self.slack = self.label_y + self.label_x[root] - self.weights[root]
        self.slackyx[:] = root

        while True:
            while st < len(queue):
                x = queue[st]; st+= 1

                is_in_graph = np.isclose(self.weights[x], self.label_x[x] + self.label_y)
                nonzero_inds = np.nonzero(np.logical_and(is_in_graph, np.logical_not(self.T)))[0]

                for y in nonzero_inds:
                    if self.yx[y] == -1:
                        return x, y
                    self.T[y] = True
                    queue.append(self.yx[y])
                    self.add_to_tree(self.yx[y], x)

            self.update_labels()
            queue, st = [], 0
            is_in_graph = np.isclose(self.slack, 0)
            nonzero_inds = np.nonzero(np.logical_and(is_in_graph, np.logical_not(self.T)))[0]

            for y in nonzero_inds:
                x = self.slackyx[y]
                if self.yx[y] == -1:
                    return x, y
                self.T[y] = True
                if not self.S[self.yx[y]]:
                    queue.append(x)
                    self.add_to_tree(self.yx[y], x)

    def solve(self, verbose = False):
        while self.max_match < self.n:
            x, y = self.find_augment_path()
            self.do_augment(x, y)

        sum = 0.
        for x in range(self.n):
            if verbose:
                print('match {} to {}, weight {:.4f}'.format(x, self.xy[x], self.weights[x, self.xy[x]]))
            sum += self.weights[x, self.xy[x]]
        self.best = sum
        if verbose:
            print('ans: {:.4f}'.format(sum))
        return sum


    def add_to_tree(self, x, prevx):
        self.S[x] = True
        self.prev[x] = prevx

        better_slack_idx = self.label_x[x] + self.label_y - self.weights[x] < self.slack
        self.slack[better_slack_idx] = self.label_x[x] + self.label_y[better_slack_idx] - self.weights[x, better_slack_idx]
        self.slackyx[better_slack_idx] = x

    def update_labels(self):
        delta = self.slack[np.logical_not(self.T)].min()
        self.label_x[self.S] -= delta
        self.label_y[self.T] += delta
        self.slack[np.logical_not(self.T)] -= delta

def get_graph_weights(old_stages, old_s, new_stages, new_s, model_param_config, buff_config=None):
    # model_param_config: list [layer_num, vocab_size, hidden_size, max_seq_len]
    # buff_config: list[batchxbeam, sesstion_len, mem_len, beam]
    # (dp, tp, pp), nagetive dp stage means ignoring replica id
    # replica id = dp_stage
    
    # check dp stage, return 0 if not equal
    if buff_config is not None and old_stages[0] > 0 and old_stages[0] != new_stages[0]:
        return 0
    
    layer_num, vocab_size, hidden_size, max_seq_len = model_param_config
    if buff_config is None:
        buff_config = 0, 0, 0, 0
    batchxbeam, sesstion_len, mem_len, beam = buff_config
    # cache_indirection = 2 * mem_len if beam > 1 else 0
    
    # all weights are divided by hidden_size to avoid giant number
    # all layers have global_dp_params, ignore
    # global_dp_params = (2*vocab_size+max_seq_len+2) + (sesstion_len+3*hidden_size+4+cache_indirection)*batchxbeam/hidden_size
    layer_dp_params = 6
    layer_tp_params = 12*hidden_size + 7 + batchxbeam*mem_len
    
    # check pp stage, get intersection layer num
    layer_num_per_stage = layer_num // old_s[2], layer_num // new_s[2]
    l = layer_num_per_stage[0] * old_stages[2], layer_num_per_stage[1] * new_stages[2]
    r = l[0] + layer_num_per_stage[0], l[1] + layer_num_per_stage[1]
    arg_max = 0 if l[0] > l[1] else 1
    layer_intersection = max(0, min(r[0], r[1]) - l[arg_max])
    
    # check tp stage, get intersection hidden_size
    layer_num_per_stage = 1 / old_s[1], 1 / new_s[1]
    l = layer_num_per_stage[0] * old_stages[1], layer_num_per_stage[1] * new_stages[1]
    r = l[0] + layer_num_per_stage[0], l[1] + layer_num_per_stage[1]
    arg_max = 0 if l[0] > l[1] else 1
    hidden_intersection = max(0, min(r[0], r[1]) - l[arg_max])
    
    print(layer_intersection, hidden_intersection)
    
    return layer_intersection * (layer_dp_params + layer_tp_params * hidden_intersection) #+ global_dp_params
    

if __name__ == '__main__':
    # print(get_graph_weights((0,1,1),(1,2,2),(0,1,3),(1,2,4),(32,51200,4096,1024)))
    
    matcher = KMMatcher([
        [2., 3., 0., 3.],
        [0., 4., 4., 0.],
        [5., 6., 0., 0.],
        [0., 0., 7., 0.]
    ])
    matcher.solve(verbose=True)