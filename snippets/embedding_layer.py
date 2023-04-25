

class EmbeddingLayer:
    """ 每个Field有自己独立的Embedding Layer，底层有自己独立的Embedding矩阵
    """

    def __init__(self, W, vocab_name, field_name):
        self.vocab_name = vocab_name
        self.field_name = field_name  # 这个Embedding Layer对应的Field
        self._W = W  # 底层的Embedding矩阵，形状是[vocab_size,embed_size]
        self._last_input = None

    def forward(self, X):
        """
        :param X: 属于某个Field的一系列稀疏类别Feature的集合
        :return: [batch_size, embed_size]
        """
        self._last_input = X  # 保存本次前代时的输入，回代时要用到

        # output：该Field的embedding，形状是[batch_size, embed_size]
        output = np.zeros((X.n_total_examples, self._W.shape[1]))

        # 稀疏输入是一系列三元组的集合，每个三元组由以下三个元素组成
        # example_idx：可以认为是sample的id
        # feat_id：每个类别特征（不是Field）的id
        # feat_val：每个类别特征对应的特征值（一般情况下都是1）
        for example_idx, feat_id, feat_val in X.iterate_non_zeros():
            # 根据feature id从Embedding矩阵中取出embedding
            embedding = self._W[feat_id, :]
            # 某个Field的embedding是，属于这个Field各个feature embedding的加权和，权重就是各feature value
            output[example_idx, :] += embedding * feat_val

        return output  # 这个Field的embedding

    def backward(self, prev_grads):
        """
        :param prev_grads: loss对这个Field output的梯度，[batch_size, embed_size]
        :return: dw，对Embedding Matrix部分行的梯度
        """
        # 只有本次前代中出现的feature id，才有必要计算梯度
        # 其结果肯定是非常稀疏的，用dict来保存
        dW = {}

        # _last_input是前代时的输入，只有其中出现的feature_id才有必要计算梯度
        for example_idx, feat_id, feat_val in self._last_input.iterate_non_zeros():
            # 由对field output的梯度，根据链式法则，计算出对feature embedding的梯度
            # 形状是[1,embed_size]
            grad_from_one_example = prev_grads[example_idx, :] * feat_val

            if feat_id in dW:
                # 一个batch中的多个样本，可能引用了相同的feature
                # 因此对某个feature embedding的梯度，应该是来自多个样本的累加
                dW[feat_id] += grad_from_one_example
            else:
                dW[feat_id] = grad_from_one_example

        return dW


class EmbeddingCombineLayer:
    """ 多个EmbeddingLayer的集合，每个EmbeddingLayer对应一个Field
    允许多个Field共享同一套Emebedding Matrix（用vocab_name标识）
    """
    ......

    def forward(self, sparse_inputs):
        """ 所有Field，先经过Embedding，再拼接
        :param sparse_inputs: dict {field_name: SparseInput}
        :return:    每个SparseInput贡献一个embedding vector，返回结果是这些embedding vector的拼接
        """
        embedded_outputs = []
        for embed_layer in self._embed_layers:
            # 获得属于这个Field的稀疏特征输入，sp_input是一组<example_idx, feat_id, feat_val>
            sp_input = sparse_inputs[embed_layer.field_name]
            # 得到属于当前Field的embedding
            embedded_outputs.append(embed_layer.forward(sp_input))

        # 最终结果是所有Field Embedding的拼接
        # [batch_size, sum of all embed-layer's embed_size]
        return np.hstack(embedded_outputs)

    def backward(self, prev_grads):
        """
        :param prev_grads:  [batch_size, sum of all embed-layer's embed_size]
                            上一层传入的, Loss对本层输出（i.e., 所有field embedding拼接）的梯度
        """

        # prev_grads是loss对“所有field embedding拼接”的导数
        # prev_grads_splits把prev_grads拆解成数组，
        # 数组内每个元素对应loss对某个field embedding的导数
        col_sizes = [layer.output_dim for layer in self._embed_layers]
        prev_grads_splits = utils.split_column(prev_grads, col_sizes)

        # _grads_to_embed也只存储"本次前代中出现的各field的各feature"的embedding
        # 其结果是超级稀疏的，因此_grads_to_embed是一个dict
        self._grads_to_embed.clear()  # reset
        for layer, layer_prev_grads in zip(self._embed_layers, prev_grads_splits):
            # layer_prev_grads: 上一层传入的，Loss对某个field embedding的梯度
            # layer_grads_to_feat_embed: dict, feat_id==>grads，
            # 某个field的embedding layer造成对某vocab的embedding矩阵的某feat_id对应行的梯度
            layer_grads_to_embed = layer.backward(layer_prev_grads)

            for feat_id, g in layer_grads_to_embed.items():
                # 表示"对某个vocab的embedding weight中的第feat_id行的总导数"
                key = "{}@{}".format(layer.vocab_name, feat_id)

                if key in self._grads_to_embed:
                    # 由于允许多个field共享embedding matrix，
                    # 因此对某个embedding矩阵的某一行的梯度应该是多个field贡献梯度的叠加
                    self._grads_to_embed[key] += g
                else:
                    self._grads_to_embed[key] = g
