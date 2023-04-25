
# group_fea_arr: 每个field的embedding
# all_fea: 将所有field embedding拼接在一起，准备接入上层网络
# 所有底层特征的embedding都是共享的
all_fea = tf.concat(group_fea_arr, axis=1)

# ------------ CVR模型
# cvr_tower_cfg是CVR模型的配置，比如有几层，每层的激活函数是什么
cvr_model = DNN(cvr_tower_cfg, ......)
cvr_logits = cvr_model(all_fea)  # 底层all_fea特征共享
probs_cvr = tf.sigmoid(cvr_logits)

# ------------ CTR模型
# ctr_tower_cfg是CTR模型的配置，比如有几层，每层的激活函数是什么
ctr_model = DNN(ctr_tower_cfg, ......)
ctr_logits = ctr_model(all_fea)  # 底层特征all_fea共享
probs_ctr = tf.sigmoid(ctr_logits)

ctr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_labels, logits=ctr_logits, name='ctr_loss')
ctr_loss = tf.reduce_sum(ctr_loss)

# ------------ CTCVR模型
probs_ctcvr = tf.multiply(probs_ctr, probs_cvr)  # pctcvr = pctr * pcvr
ctcvr_labels = cvr_labels * ctr_labels

ctcvr_loss = tf.keras.binary_crossentropy(ctcvr_labels, probs_ctcvr)
ctcvr_loss = tf.reduce_sum(ctcvr_loss, name='ctcvr_loss')

# ------------ TOTAL LOSS
# 要优化的total loss是两个子loss的加权和
return ctr_loss_weight * ctr_loss + ctcvr_loss_weight * ctcvr_loss
