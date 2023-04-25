
import tensorflow as tf

# *********************** 生成最优初始值
# 定义一个简单网络结构，负责生成个性化的id embedding的初值
# 它的参数是整个元学习唯一需要优化的参数
meta_emb_generator = ...

# 输入item_emb_list: a list，每个元素是物料的某个非ID特征的embedding
# item_emb_list可以训练好的主模型中提取，并且stop_gradient将其固定
# 输出item_id_emb0: 每个item的id embedding的最优初值
item_id_emb0 = meta_emb_generator(tf.stop_gradient(item_emb_list))

# *********************** COLD-START阶段
# ph_cold_inputs代表cold-start阶段的输入数据的placeholder
# ph_cold_labels是与ph_cold_inputs对应的标签的placeholder
ph_cold_inputs, ph_cold_labels = ...

# main_model是主模型，比如预测CTR
# 输入当前item id embedding（这里是item_id_emb0，由meta_emb_generator生成的最优初值）和
# cold-start阶段的输入ph_cold_inputs
# 输出cold-start阶段的预测值cold_preds
cold_preds = main_model(item_id_emb0, ph_cold_inputs)
cold_loss = loss_func(ph_cold_labels, cold_preds)  # 计算cold-start阶段的loss

# *********************** 由最优初值向前迭代一步
# cold_emb_grads是冷启loss对item id embedding的梯度
# item_id_emb1由id embedding的最优初值item_id_emb0, 经过一次梯度下降得到
cold_emb_grads = tf.gradients(cold_loss, item_id_emb0)[0]
item_id_emb1 = item_id_emb0 - cold_lr * cold_emb_grads  # cold_lr是步长

# *********************** WARM-UP阶段
# ph_warm_inputs代表warm-up阶段的输入数据的placeholder
# ph_warm_labels是与ph_warm_inputs对应的标签的placeholder
ph_warm_inputs, ph_warm_labels = ...

# 基于当前item id embedding（item_id_emb1）和warm-up阶段的输入ph_warm_inputs
# 输出warm-up阶段的预测值warm_preds
warm_preds = main_model(item_id_emb1, ph_warm_inputs)
warm_loss = loss_func(ph_warm_labels, warm_preds)  # 计算warm-up阶段的loss

# *********************** 定义训练操作
# alpha是控制两阶段loss的权重
meta_loss = alpha * cold_loss + (1-alpha) * warm_loss

optimizer = tf.train.AdamOptimizer(...)
# meta_emb_generator是生成最优item id embedding的网络结构，它的参数是Meta Emnbedding唯一需要优化的参数
meta_train_op = optimizer.minimize(meta_loss, var_list=meta_emb_generator.trainable_variables)

# *********************** 开始训练
sess = tf.Session()
for batch in train_data_stream:
    # 将当前batch拆分为cold-start与warm-up两个阶段
    cold_inputs, cold_labels = get_cold_train_data(batch)
    warm_inputs, warm_labels = get_warm_train_data(batch)

    feed_dict = {
        ph_cold_inputs: cold_inputs,
        ph_cold_labels: cold_labels,
        ph_warm_inputs: warm_inputs,
        ph_warm_labels: warm_labels
    }
    sess.run(meta_train_op, feed_dict=feed_dict)
