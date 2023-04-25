

class MovielensModel(tfrs.models.Model):
    """电影推荐场景下的双塔召回模型"""

    def __init__(self, layer_sizes):
        super().__init__()
        self.query_model = QueryModel(layer_sizes)  # 用户塔
        self.candidate_model = CandidateModel(layer_sizes)  # 物料塔
        self.task = tfrs.tasks.Retrieval(......)  # 负责计算Loss

    def compute_loss(self, features, training=False):
        # 只把用户特征喂入“用户塔”，得到user embedding "query_embeddings"
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "timestamp": features["timestamp"],
        })
        # 只把物料特征喂入“物料塔”，生成item embedding "movie_embeddings"
        movie_embeddings = self.candidate_model(features["movie_title"])

        # 根据Batch内负采样方式，计算Sampled Softmax Loss
        return self.task(query_embeddings, movie_embeddings, ......)


class Retrieval(tf.keras.layers.Layer, base.Task):

    def call(self, query_embeddings, candidate_embeddings,
             sample_weight, candidate_sampling_probability, ......) -> tf.Tensor:
        """
        query_embeddings: [batch_size, dim]，可以认为是user embedding
        candidate_embeddings: [batch_size, dim]，可以认为是item embedding
        """
        # query_embeddings: [batch_size, dim]
        # candidate_embeddings: [batch_size, dim]
        # scores: [batch_size, batch_size]，batch中的每个user对batch中每个item的匹配度
        scores = tf.linalg.matmul(query_embeddings, candidate_embeddings, transpose_b=True)

        # labels: [batch_size, batch_size]，对角线上全为1，其余位置都是0
        labels = tf.eye(tf.shape(scores)[0], tf.shape(scores)[1])

        if self._temperature is not None:  # 通过温度，调整训练难度
            scores = scores / self._temperature

        if candidate_sampling_probability is not None:
            # SamplingProbablityCorrection的实现就是
            # logits - tf.math.log(candidate_sampling_probability)
            # 因为负样本是抽样的，而非全体item，Sampled Softmax进行了概率修正
            scores = layers.loss.SamplingProbablityCorrection()(scores, candidate_sampling_probability)

        ......

        # labels: [batch_size, batch_size]
        # scores: [batch_size, batch_size]
        # self._loss就是tf.keras.losses.CategoricalCrossentropy
        # 对于第i个样本，只有labels[i,i]等于1，scores[i,i]是正样本得分
        # 其他位置上的labels[i,j]都为0，scores[i,j]都是负样本得分
        # 所以这里实现的是Batch内负采样，第i行样本的用户，把除i之外所有样本中的正例物料，当成负例物料
        loss = self._loss(y_true=labels, y_pred=scores, sample_weight=sample_weight)
        return loss
