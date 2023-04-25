
import tensorflow as tf

# ----------- 准备
unq_categories = ["music", "movie", "finance", "game", "military", "history"]
# 这一层负责将string转化为int型id
id_mapping_layer = tf.keras.layers.StringLookup(vocabulary=unq_categories)

emb_layer = tf.keras.layers.Embedding(
    # 多加一维是为了处理，当输入不包含在unq_categories的情况
    input_dim=len(unq_categories) + 1,
    output_dim=4)  # output_dim指明映射向量的长度

# ----------- Embedding
cate_input = ...  # [batch_size,1]的string型"文章分类"向量
cate_ids = id_mapping_layer(cate_input)  # string型输入的“文章分类”映射成int型id
# 得到形状=[batch_size,4]的float稠密向量，表示每个“文章分类”的语义
cate_embeddings = emb_layer(cate_ids)
