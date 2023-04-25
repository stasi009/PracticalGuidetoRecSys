
class NeighborSampler(object):
    """邻居采样，生成各层卷积所需要的计算子图"""

    def __init__(self, g, ......):
        self.g = g
        # 每层都有一个采样器，根据随机游走来决定某节点邻居的重要性
        # 可以认为经过多次游走，落脚于某邻居节点的次数越多，则这个邻居越重要，就更应该优先作为邻居
        self.samplers = [dgl.sampling.PinSAGESampler(g, ......) for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        for sampler in self.samplers:
            # 通过随机游走，选择重要邻居，构成子图
            frontier = sampler(seeds)

            if heads is not None:
                # 如果是在训练，需要将heads->tails和head->neg_tails这些待预测的边都去掉
                # 否则head/tails的信息会沿着边相互传统，引发信息泄漏
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)

            # 只保留seeds这些节点，将frontier压缩成block
            block = compact_and_copy(frontier, seeds)

            # 本层的输入节点就是下一层的seeds
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return blocks  # 各层卷积所需要的计算子图


class WeightedSAGEConv(nn.Module):
    """单层图卷积"""

    def forward(self, g, h, weights):
        """
        g : 某一层的计算子图，就是NeighborSampler生成的block
        h : 是一个tuple，包含源节点、目标节点上一层的embedding
        weights : 边上的权重
        """
        h_src, h_dst = h  # 源节点、目标节点上一层的embedding
        with g.local_scope():
            # 将src节点上的原始特征映射成hidden_dims长，存储于各节点的'n'字段
            # Q是线性映射的权重，act是激活函数
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            # 边上的权重，存储于各边的'w'字段
            g.edata['w'] = weights.float()

            # DGL采取"消息传递"方式来实现图卷积
            # g.update_all是更新全部节点，更新方式是：
            # fn.u_mul_e: src节点上的特征'n'乘以边权重'w'，构成消息'm'
            # fn.sum:     dst节点将所有接收到的消息'm'，相加起来，更新dst节点的'n'字段
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))

            # 将边上的权重w拷贝成消息'm'
            # dst节点将所有接收到的消息'm'，相加起来，存入dst节点的'ws'字段
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))

            # 某个dst节点的n字段，已经被更新成，它的所有邻居节点的embedding的加权和
            n = g.dstdata['n']
            # 某个dst节点的ws字段，是指向它的所有边上权重之和
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)

            # n / ws：将邻居节点的embedding，做加权平均
            # 再拼接上一轮卷积后，dst节点自身的embedding
            # 再经过线性映射（W）与非线性激活（act），得到这一轮卷积后各dst节点的embedding
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))

            # 本轮卷积后，各dst节点的embedding除以模长，进行归一化
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z


def load_variables():
    pass


def load_bias():
    pass


def ReLU():
    pass


def emit():
    pass


def zero_vector():
    pass


def matmul():
    pass


def concat():
    pass


class Mapper:
    def __init__(self, k) -> None:
        # 装载第k层卷积的权重
        self._Q = load_variables(k, "Q")
        self._q = load_variables(k, "q")

    def map(self, node, embedding, neighNodes, weightsToNeigh):
        """
        node: 某个节点
        embedding: node上一轮卷积后得到的向量
        neighNodes: node的邻居节点
        weightsToNeigh: node对其每个邻居的重要程度
        """
        # 线性映射+非线性激活，获得要发向邻居的新向量
        emb4neigh = ReLU(self._Q * embedding + self._q)

        for (destNode, weight2neigh) in zip(neighNodes, weightsToNeigh):
            message = (node, emb4neigh, weight2neigh)
            # node作为消息来源，将其新向量发往，目标destNode
            # MapReduce框架会以destNode为key，将所有message聚合起来
            emit(destNode, message)

        # node自己作为目标节点，也要参与reduce，所以也要发出
        emit(node, (node, embedding, 1))


class Reducer:
    def __init__(self, k) -> None:
        # 装载第k层卷积的权重
        self._W = load_variables(k, "W")
        self._w = load_variables(k, "w")

    def reduce(self, node, messages):
        """
        node: 当前节点
        messages: node的所有邻居发向node的消息集合
        """
        old_self_embedding = None
        neigh_agg_embedding = zero_vector()  # 初始化空向量
        neigh_sum_weight = 0

        for (nbNode, nbEmbedding, nbWeight) in messages:
            # 每个消息由三部分组成：
            # nbNode: 从哪个邻居节点发来的
            # nbEmbedding: 邻居节点发来的向量
            # nbWeight: 邻居节点对当前节点node的重要程度
            if nbNode == node:
                old_self_embedding = nbEmbedding  # 当前节点上一轮的向量
            else:
                neigh_agg_embedding += nbWeight * nbEmbedding
                neigh_sum_weight += nbWeight

        # 所有邻居向当前节点扩散信息的加权平均
        neigh_agg_embedding = neigh_agg_embedding / neigh_sum_weight

        new_embedding = ReLU(self._W * concat(neigh_agg_embedding, old_self_embedding) + self._w)
        new_embedding = new_embedding / l2_norm(new_embedding)  # L2 normalization

        emit(node, new_embedding)  # MapReduce会把每个节点和它的新向量，保存到HDFS上


def broadcast():
    pass


def join_emb_with_neighbors():
    pass


def get_emb_path():
    pass


def run_distributedly():
    pass


temp_path = None


def inference_embeddings():
    # 把每层卷积所需要的权重，广播到集群中的每台机器上
    for k in range(num_layers):
        broadcast(Q[k], q[k], W[k], w[k])

    for k in range(1, num_layers+1):
        old_emb_path = get_emb_path(k-1)# 上一轮卷积结果的保存路径

        # 上一轮的卷积结果，每个节点只有(node,embedding)信息
        # 还要再拼接上，每个节点的邻居列表，和当前节点对每个邻居的重要性
        # 拼接后，每个节点的数据包括：(node, embedding, neighNodes, weightsToNeigh)，才符合mapper的需要
        # 保存到HDFS的input_path路径下
        input_path = join_emb_with_neighbors(old_emb_path)

        run_distributedly(input=input_path,
                          job=Mapper(k),
                          output=temp_path)

        output_path = get_emb_path(k)
        run_distributedly(input=temp_path,
                          job=Reducer(k),
                          output=output_path)

    # 各节点最终embedding的保存路径
    final_path = get_emb_path(num_layers)
