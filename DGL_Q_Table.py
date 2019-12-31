import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import dgl
import networkx as nx
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 16
LR = 0.01                   # learning rate
EPSILON = 0.5               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 100
env = gym.make('FrozenLake-v0')
env = env.unwrapped
SAMPLE_EDGES = 20
N_ACTIONS = env.action_space.n
N_H = 20
BETA = 0.2
N_STATES = 16
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1)

    def forward_from_record(self, g, h):
        return g.ndata['z'];
      
class GAT(nn.Module):
  def __init__(self, in_feats, hidden_size, num_classes):
      super(GAT, self).__init__()
      self.gat1 = GATLayer(in_feats, num_classes)
      self.gat2 = GATLayer(hidden_size, num_classes)
      self.out1 = nn.Linear(in_feats, num_classes)
      self.out2 = nn.Linear(hidden_size, num_classes)

  def record(self, g, nodes_id, records):
        g.ndata['z'][nodes_id,:] = BETA * g.ndata['z'][nodes_id,:] + (1 - BETA) * records

  def forward(self, g, features):
        h = self.gat1.forward_from_record(g, features)
        return h
      
class DQN(object):
    def __init__(self):
        self.bg = dgl.DGLGraph()
        self.eval_net, self.target_net = GAT(N_STATES, N_H, N_ACTIONS), GAT(N_STATES, N_H, N_ACTIONS)
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, 1 * 2 + 3))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def add_nodes(self, features):
        nodes_id = self.bg.number_of_nodes()
        if nodes_id != 0:
            for i in range(len(self.bg.ndata['x'])):
                if self.bg.ndata['x'][i].equal(features[0]):
                    return i;
        self.bg.add_nodes(1, {'x': features, 'z': torch.zeros(1, N_ACTIONS)})
        src = [nodes_id]
        dst = [nodes_id]
        self.bg.add_edges(src, dst) 
        return nodes_id
            
        
        
    def choose_action(self, nodes_id):
        actions_value = self.eval_net(self.bg, self.bg.ndata['x'])[nodes_id]
        if np.random.uniform() < EPSILON:   # greedy
            action = torch.argmax(actions_value).data.item()
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        Q = actions_value[action];

        return action, Q.detach().numpy()

    def learn_one(self, nodes_id, next_nodes_id, r):
        h_target = self.eval_net(self.bg, self.bg.ndata['x'])
        q_target = (r + GAMMA * h_target[next_nodes_id, :].max(0)[0])
        self.eval_net.record(self.bg, nodes_id, q_target)

dqn = DQN()
      
r_sum = 0
rList = []
for i in range(2000):
    s = env.reset()
    x = np.zeros((1, 16))
    x[0,s] = 1
    x = torch.FloatTensor(x)
    nodes_id = dqn.add_nodes(x)
    rAll = 0
    d = False
    j = 0
    while j < 99:
        j += 1
        a, Q = dqn.choose_action(nodes_id)
        s1, r, d, _ = env.step(a)
        x = np.zeros((1, 16))
        x[0,s1] = 1
        x = torch.FloatTensor(x)
        next_nodes_id = dqn.add_nodes(x)
        # Update Q_Table
        dqn.learn_one(nodes_id, next_nodes_id, r);
        rAll += r
        nodes_id = next_nodes_id
        if d == True:
            break
    rList.append(rAll)
 
    print("Score over time："+ str(sum(rList)/2000))
    
