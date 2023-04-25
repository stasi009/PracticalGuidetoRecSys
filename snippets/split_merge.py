
scene_indicator = ...  # “场景指示”特征
is_domain_a = tf.squeeze(tf.equal(scene_indicator, 1), axis=1)  # 样本是否来自A场景
is_domain_b = tf.squeeze(tf.equal(scene_indicator, 0), axis=1)  # 样本是否来自B场景


def split(inputs):
    # indices是每条样本的序号
    indices = tf.range(tf.shape(inputs)[0], dtype=tf.int32)

    # a_inputs是inputs中属于“场景A”的那些样本
    a_segments = tf.boolean_mask(inputs, is_domain_a, axis=0)
    # a_indices是a_inputs在原始完整的inputs中的序号，未来merge时用得上
    a_indices = tf.boolean_mask(indices, is_domain_a)

    # b_inputs是inputs中属于“场景B”的那些样本
    b_segments = tf.boolean_mask(inputs, is_domain_b, axis=0)
    # b_indices是b_inputs在原始完整的inputs中的序号，未来merge时用得上
    b_indices = tf.boolean_mask(indices, is_domain_b)

    return a_segments, a_indices, b_segments, b_indices


def merge(a_segments, a_indices, b_segments, b_indices):
    merged_inputs = tf.concat([a_segments, b_segments], axis=0)
    merged_indices = tf.concat([a_indices, b_indices], axis=0)

    # positions是将merged_indices升序排序，所需要的位置映射表
    positions = tf.argsort(merged_indices)

    # merged_indices应该永远与merged_inputs保持相同顺序
    # 能够将merged_indices调整成正确顺序的位置映射表是positions
    # 按照positions，也能够merged_inputs调整成与原始输入相同的顺序
    return tf.gather(merged_inputs, positions, axis=0)


# **************** 先经过“共享底层”
all_features = ...  # 原始一个batch内的样本
all_labels = ...  # 原始一个batch内的预测目标
shared_bottom_output = SharedBottom(all_features)

# **************** SPLIT
a_inputs, a_indices, b_inputs, b_indices = split(shared_bottom_output)

# **************** 不同场景独立建模
tower_a = Tower(...)  # tower_a只处理“A场景”的样本
a_outputs = tower_a(a_inputs)  # A场景的输出

tower_b = Tower(...)  # tower_b只处理“B场景”的样本
b_outputs = tower_b(b_inputs)  # B场景的输出

# **************** MERGE
# 所有场景下的样本的预测值，与Batch原来的顺序相同
all_outputs = merge(a_outputs, a_indices, b_outputs, b_indices)
# all_outputs中每条样本与all_labels中的每个label，顺序对得上
total_loss = LOSS(all_outputs, all_labels)
