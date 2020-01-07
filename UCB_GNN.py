import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from gym.envs.registration import register

# Hyper Parameters
C = 0.0001
EPSILON = 0.5               # greedy policy
GAMMA = 0.95                 # reward discount
env = gym.make('FrozenLake-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
BETA = 0.2
N_STATES = 16
INNER_STATES = 10
LR = 0.1
HIDDEN_SIZE = 50
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class GNNLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GNNLayer, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def message_func(self, edges):
        Q = GAMMA * torch.max(edges.src['z'], dim = -1, keepdim = True)[0] * edges.data['e'][:,1,:] + edges.data['e'][:,0,:]
        a_count = edges.data['e'][:,1,:]
        return {'q': Q, 'ac' : a_count}

    def reduce_func(self, nodes):
        z = BETA * nodes.data['z'] + (1 - BETA) * torch.sum(nodes.mailbox['q'], dim = 1) / (torch.sum(nodes.mailbox['ac'], dim = 1) + 1e-6)
        a = torch.sum(nodes.mailbox['ac'], dim = 1)
        return {'z': z, 'a': a}

    def bp(self, g):
        g.update_all(self.message_func, self.reduce_func)

    def forward_from_nn(self, feature):
        h = self.fc1(feature)
        out = self.fc2(h)
        return out

    def forward(self, g):
        return g.ndata['z']

    def ac(self, g):
        return g.ndata['a']
      
class GNN(nn.Module):
  def __init__(self, in_dim, hiddien_dim, out_dim):
      super(GNN, self).__init__()
      self.gnn = GNNLayer(in_dim, hiddien_dim, out_dim)
      self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=LR)

  def record(self, g, nodes_id, records):
      g.ndata['z'][nodes_id,:] = BETA * g.ndata['z'][nodes_id,:] + (1 - BETA) * records

  def bp(self, g):
      self.gnn.bp(g)

  def bp_from_nn(self, g):
      nn_predict = self.gnn.forward_from_nn(g.ndata['x'])
      mse = (g.ndata['z'] - nn_predict) * (g.ndata['z'] - nn_predict)
      loss = torch.mean(mse)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()  

  def forward_from_nn(self, feature):
      out = self.gnn.forward_from_nn(feature)
      return out

  def forward(self, g):
      h = self.gnn(g)
      return h
  
  def action_number(self, g):
      h = self.gnn.ac(g)
      return h
      
class DQN(object):
    def __init__(self):
        self.bg = dgl.DGLGraph()
        self.eval_net = GNN(N_STATES, HIDDEN_SIZE, N_ACTIONS)

    def add_edges(self, nodes_id, next_nodes_id, a, r):
        if nodes_id == next_nodes_id:
            return
        src = [nodes_id]
        dst = [next_nodes_id]
        if self.bg.has_edge_between(next_nodes_id, nodes_id):
            edge = torch.zeros([1, 2, N_ACTIONS])
            edge[0, 0, a] = r
            edge[0, 1, a] = 1.0
            self.bg.edges[next_nodes_id, nodes_id].data['e'] += edge
            return
        edge = torch.zeros([1, 2, N_ACTIONS])
        edge[0, 0, a] = r
        edge[0, 1, a] = 1.0
        #print(edge)
        self.bg.add_edges(dst, src, {'e': edge})

    def add_nodes(self, features):
        nodes_id = self.bg.number_of_nodes()
        if nodes_id != 0:
            for i in range(len(self.bg.ndata['x'])):
                if self.bg.ndata['x'][i].equal(features[0]):
                    return i;
        self.bg.add_nodes(1, {'x': features, 'z': self.eval_net.forward_from_nn(features).detach(), 'a': torch.zeros(1, N_ACTIONS)})
        return nodes_id

    def choose_action_ucb(self, nodes_id, time):
        #actions_value = self.eval_net(self.bg)[nodes_id].detach().numpy()
        actions_value = self.eval_net.forward_from_nn(self.bg.ndata['x'][nodes_id]).detach().numpy()
        actions_value += C * np.sqrt( np.log(time+1) / (self.eval_net.action_number(self.bg)[nodes_id].data.numpy() + 1))
        action = np.argmax(actions_value)
        Q = actions_value[action];

        return action, Q

    def choose_action(self, nodes_id):
        actions_value = self.eval_net(self.bg)[nodes_id]
        if np.random.uniform() < EPSILON:   # greedy
            action = torch.argmax(actions_value).data.item()
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        Q = actions_value[action];

        return action, Q.detach().numpy()

    def learn(self):
        self.eval_net.bp(self.bg)
        self.eval_net.bp_from_nn(self.bg)


dqn = DQN()
      
r_sum = 0
rList = []
for i in range(10000):
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
        a, Q = dqn.choose_action_ucb(nodes_id, i)
        s1, r, d, _ = env.step(a)

        x = np.zeros((1, 16))
        x[0,s1] = 1
        x = torch.FloatTensor(x)
        next_nodes_id = dqn.add_nodes(x)
        dqn.add_edges(nodes_id, next_nodes_id, a, r)
        # Update Q_Table
        rAll += r
        nodes_id = next_nodes_id
        dqn.learn()
        if d == True:
            break
    rList.append(rAll)
    if (len(rList) == 1000):
        print("Score over time："+ str(sum(rList)/1000))
        rList = []

