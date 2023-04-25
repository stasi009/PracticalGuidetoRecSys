
import tensorflow as tf


def create_padding_mask(seq):
    """
    seq: [batch_size, seq_len]的整数矩阵。如果某个元素==0，代表那个位置是padding
    """
    # (batch_size, seq_len)
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 返回结果：(batch_size, 1, 1, seq_len)
    # 加入中间两个长度=1的维度，是为了能够broadcast成希望的形状
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """
    输入：
        q: (batch_size, num_heads, seq_len_q, dim_key)
        k: (batch_size, num_heads, seq_len_k, dim_key)
        v: (batch_size, num_heads, seq_len_k, dim_val)
        mask: 必须能够broadcastableto (..., seq_len_q, seq_len_k)的形状
    输出：
        output: q对k/v做attention的结果, (batch_size, num_heads, seq_len_q, dim_val)
        attention_weights: q对k的注意力权重, (batch_size, num_heads, seq_len_q, seq_len_k)
    """

    # q: (batch_size, num_heads, seq_len_q, dim_key)
    # k: (batch_size, num_heads, seq_len_k, dim_key)
    # matmul_qk: 每个head下，每个q对每个k的注意力权重（尚未归一化）
    # (batch_size, num_heads, seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 为了使训练更稳定，除以sqrt(dim_key)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 在mask的地方，加上一个极负的数，-1e9，保证在softmax后，mask位置上的权重都是0
    if mask is not None:
        # mask的形状一般是(batch_size, 1, 1, seq_len_k)
        # 但是能够broadcast成与scaled_attention_logits相同的形状
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention_logits += (mask * -1e9)

    # 沿着最后一维(i.e., seq_len_k)用softmax归一化
    # 保证一个query对所有key的注意力权重之和==1
    # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    # v: (batch_size, num_heads, seq_len_k, dim_val)
    # output: (batch_size, num_heads, seq_len_q, dim_val)
    output = tf.matmul(attention_weights, v)

    # output: (batch_size, num_heads, seq_len_q, dim_val)
    # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, dim_key, dim_val, dim_out):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_key = dim_key  # 每个query和key都要映射成相同的长度
        # 每个value要映射成的长度
        self.dim_val = dim_val if dim_val is not None else dim_key

        # 定义映射矩阵
        self.wq = tf.keras.layers.Dense(num_heads * dim_key)
        self.wk = tf.keras.layers.Dense(num_heads * dim_key)
        self.wv = tf.keras.layers.Dense(num_heads * dim_val)
        self.wo = tf.keras.layers.Dense(dim_out)  # dim_out：希望输出的维度长

    def split_heads(self, x, batch_size, dim):
        # 输入x: (batch_size, seq_len, num_heads * dim)
        # 输出x: (batch_size, seq_len, num_heads, dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, dim))

        # 最终输出：(batch_size, num_heads, seq_len, dim)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        """
        输入：
            q: (batch_size, seq_len_q, old_dq)
            k: (batch_size, seq_len_k, old_dk)
            v: (batch_size, seq_len_k, old_dv),与k序列相同长度
            mask: 可以为空，否则形状为(batch_size, 1, 1, seq_len_k)，表示哪个key不需要做attention
        输出：
            output: Attention结果，(batch_size, seq_len_q, dim_out)
            attention_weights: Attention权重，(batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # **************** 将输入映射成希望的形状
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len_q, num_heads * dim_key)
        k = self.wk(k)  # (batch_size, seq_len_k, num_heads * dim_key)
        v = self.wv(v)  # (batch_size, seq_len_k, num_heads * dim_val)

        # (bs, nh, seq_len_q, dim_key)
        q = self.split_heads(q, batch_size, self.dim_key)
        # (bs, nh, seq_len_k, dim_key)
        k = self.split_heads(k, batch_size, self.dim_key)
        # (bs, nh, seq_len_k, dim_val)
        v = self.split_heads(v, batch_size, self.dim_val)

        # **************** Multi-Head Attention
        # scaled_attention: (batch_size, num_heads, seq_len_q, dim_val)
        # attention_weights:(batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # **************** 将Attention结果映射成希望的形状
        # (batch_size, seq_len_q, num_heads, dim_val)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, num_heads * dim_val)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.num_heads * self.dim_val))

        output = self.wo(concat_attention)  # (batch_size, seq_len_q, dim_out)

        return output, attention_weights


def target_attention():
    target_item_embedding = ...  # 候选item的embedding, [batch_size, dim_target]
    user_behavior_seq = ...  # 某个用户行为序列, [batch_size, seq_len, dim_seq]
    padding_mask = ...  # user_behavior_seq中哪些位置是填充的，不需要Attention

    # 把候选item，变形成一个长度为1的序列
    query = tf.reshape(target_item_embedding, [-1, 1, dim_target])

    # atten_result: (batch_size, 1, dim_out)
    attention_layer = MultiHeadAttention(num_heads, dim_key, dim_val, dim_out)
    atten_result, _ = attention_layer(
        q=query,  # query就是候选物料
        k=user_behavior_seq,
        v=user_behavior_seq,
        mask=padding_mask)

    # reshape去除中间不必要的1维
    # user_interest_emb是提取出来的用户兴趣向量，喂给上层模型，参与CTR建模
    user_interest_emb = tf.reshape(atten_result, [-1, dim_out])


def double_attention():
    target_item_embedding = ...  # 候选item的embedding, [batch_size, dim_target]
    user_behavior_seq = ...  # 某个用户行为序列, [batch_size, seq_len, dim_in_seq]
    padding_mask = ...  # user_behavior_seq中哪些位置是填充的，不需要attention
    dim_in_seq = tf.shape(user_behavior_seq)[-1]  # sequence中每个element的长度

    # *********** 第一层做Self-Attention，建模序列内部的依赖性
    self_atten_layer = MultiHeadAttention(num_heads=n_heads1,
                                            dim_key=dim_in_seq,
                                            dim_val=dim_in_seq,
                                            dim_out=dim_in_seq)
    # 做self-attention，q=k=v=user_behavior_seq
    # 输入q/k/v与输出self_atten_seq，它们的形状都是
    # [batch_size, len(user_behavior_seq), dim_in_seq]
    self_atten_seq, _ = self_atten_layer(q=user_behavior_seq,
                                            k=user_behavior_seq,
                                            v=user_behavior_seq,
                                            mask=padding_mask)

    # *********** 第二层做Target-Attention，建模候选item与行为序列的相关性
    target_atten_layer = MultiHeadAttention(num_heads=n_heads2,
                                            dim_key=dim_key,
                                            dim_val=dim_val,
                                            dim_out=dim_out)
    # 把候选item，变形成一个长度为1的序列
    target_query = tf.reshape(target_item_embedding, [-1, 1, dim_target])
    # atten_result: (batch_size, 1, dim_out)
    atten_result, _ = target_atten_layer(
        q=target_query,  # 代表候选物料
        k=self_atten_seq,  # 以self-attention结果作为target-attention的对象
        v=self_atten_seq,
        mask=padding_mask)

    # reshape去除中间不必要的1维
    # user_interest_emb是提取出来的用户兴趣向量，喂给上层模型，参与CTR建模
    user_interest_emb = tf.reshape(atten_result, [-1, dim_out])


def auto_int():
    # 原始特征的拼接而成的矩阵，[batch_size, num_fields, dim]
    # num_fields：一共有多少个field
    # dim：每个field都被映射成相当长度为dim的embedding
    X = ...

    attention_layer = MultiHeadAttention(num_heads, dim_key, dim_val, dim_out)
