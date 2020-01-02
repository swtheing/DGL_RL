# Reinforcement Learning(RL) with Graph Neutral Network(GNN)

在DGL基础上做的RL with GNN Demo

## 介绍

这个Demo分为两部分

1）使用GNN完全复现Q-Table机制。

2）使用GNN抛弃Q机制，实现新的RL机制。

### 环境介绍

[FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/)：来自OpenAI Gym，特点是状态分布有限（16个基础状态）。

### Q-Table与强化学习的介绍
https://www.cnblogs.com/DjangoBlog/p/9398684.html

### 预先安装
```bash
pip install dgl
pip install pytorch
pip install gym
```

## Demo 1 : Q-Table with GNN
初始化图:
```bash
self.bg = dgl.DGLGraph()
```
利用 add_nodes加入新节点, 加入data['x']表示节点，加入data['z']表示Q结果:
```bash
def add_nodes(self, features):
    nodes_id = self.bg.number_of_nodes()
    if nodes_id != 0:
    for i in range(len(self.bg.ndata['x'])):
        if self.bg.ndata['x'][i].equal(features[0]):
            return i;
    self.bg.add_nodes(1, {'x': features, 'z': torch.zeros(1, N_ACTIONS)})
```
进行Q-table更新
```bash
def learn_one(self, nodes_id, next_nodes_id, r):
    h_target = self.eval_net(self.bg, self.bg.ndata['x'])
    q_target = (r + GAMMA * h_target[next_nodes_id, :].max(0)[0])
    self.eval_net.record(self.bg, nodes_id, q_target)
```
预测结果
```bash
def forward_from_record(self, g, h):
    return g.ndata['z'];
```
## Demo 2 : RL with GNN
在Demo 2中我们将抛弃整个Q机制，而是用图自动更新Q值。我们利用图构建状态游戏中的状态转换机制，并在图更新中自动学习Q值.

除了在DEMO 1中的构图之外，我们需要简历图中节点的边：

- 边方向是状态转换的反方向，是为了reward的回溯，
- 边的第一列记录了状态A到状态B的action & reward，
- 边的第二列记录了状态A到状态B action走过的次数。

```bash
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
     self.bg.add_edges(dst, src, {'e': edge})
```
除此之外，我们加入了图的更新机制：
- 在message_fuc中我们计算了B节点的Q的最大值，并在不同边上乘以了边的action返回给A节点，除此之外我们也把走过action 的次数返回给B节点。
- 在B节点上，我们reduce每个action的Q值，处以整个action整体的次数（相当于加权平均），这样就按照(action, State)的分布更新了Q值，这就是为什么GNN领先与Q-Table的地方。
```bash
 def message_func(self, edges):
     Q = GAMMA * torch.max(edges.src['z'], dim = -1, keepdim = True)[0] \
       * edges.data['e'][:,1,:] + edges.data['e'][:,0,:]
     a_count = edges.data['e'][:,1,:]
     return {'q': Q, 'ac' : a_count}

 def reduce_func(self, nodes):
     z = BETA * nodes.data['z'] + (1 - BETA) * \ 
            torch.sum(nodes.mailbox['q'],  dim = 1)\
            / (torch.sum(nodes.mailbox['ac'], dim = 1) + 1e-6)
     return {'z': z}
```

# 评测和对比
下面对比了一下在游戏上的整体效果：

| Epoch| Q-Table | Q-Table with GNN | RL with GNN|
| ------ | ------ | ------ |------|
| 1 | 0.029 | 0.013|0.054|
| 2 | 0.040 | 0.006 |0.077|
| 3 | 0.032 | 0.009 |0.072|
| 4 | 0.039 | 0.012 |0.085|
| 5 | 0.040 | 0.005|0.075|
| 6 | 0.029 | 0.007 |0.075|
| 7 | 0.030 | 0.015|0.070|
| 8 | 0.034 | 0.006 |0.064|
| 9 | 0.032 | 0.008 |0.060|
| 10 | 0.035 | 0.009 |0.070|


## License
[MIT](https://choosealicense.com/licenses/mit/)
