


def nce_loss(weights,
			biases,
			labels,
			inputs,
			num_sampled,
			num_classes,
			num_true=1,......):
    """
    weights: 待优化的矩阵，形状[num_classes, dim]。可以理解为所有item embedding矩阵，此时num_classes=所有item的个数
    biases: 待优化变量，[num_classes]。每个item还有自己的bias，与user无关，代表自己本身的受欢迎程度。
    labels: 正例的item ids，[batch_size,num_true]的整数矩阵。center item拥有的最多num_true个positive context item id
    inputs: 输入的[batch_size, dim]矩阵，可以认为是center item embedding
    num_sampled：整个batch要采集多少负样本
    num_classes: 在i2i中，可以理解成所有item的个数
    num_true: 一条样本中有几个正例，一般就是1
    """
    # logits: [batch_size, num_true + num_sampled]的float矩阵
    # labels: 与logits相同形状，如果num_true=1的话，每行就是[1,0,0,...,0]的形式
    logits, labels = _compute_sampled_logits(......)

    # sampled_losses：形状与logits相同，也是[batch_size, num_true + num_sampled]
    # 一行样本包含num_true个正例和num_sampled个负例
    # 所以一行样本也有num_true + num_sampled个sigmoid loss
    sampled_losses = sigmoid_cross_entropy_with_logits(
                labels=labels,
                logits=logits,
                name="sampled_losses")
                
    # 把每行样本的num_true + num_sampled个sigmoid loss相加
    return _sum_rows(sampled_losses)


def _compute_sampled_logits(weights,
							biases,
							labels,
							inputs,
							num_sampled,
							num_classes,
							num_true=1,
							......
							subtract_log_q=True,
							remove_accidental_hits=False,......):
    """
    输入：
        weights: 待优化的矩阵，形状[num_classes, dim]。可以理解为所有item embedding矩阵，那时num_classes=所有item的个数
        biases: 待优化变量，[num_classes]。每个item还有自己的bias，与user无关，代表自己的受欢迎程度。
        labels: 正例的item ids，[batch_size,num_true]的整数矩阵。center item拥有的最多num_true个positive context item id
        inputs: 输入的[batch_size, dim]矩阵，可以认为是center item embedding
        num_sampled：整个batch要采集多少负样本
        num_classes: 在i2i中，可以理解成所有item的个数
        num_true: 一条样本中有几个正例，一般就是1
        subtract_log_q：是否要对匹配度，进行修正。如果是NEG Loss，关闭此选项。
        remove_accidental_hits：如果采样到的某个负例，恰好等于正例，是否要补救
    输出：
        out_logits: [batch_size, num_true + num_sampled]
        out_labels: 与out_logits同形状
    """
	# labels原来是[batch_size, num_true]的int矩阵
	# reshape成[batch_size * num_true]的数组
	labels_flat = array_ops.reshape(labels, [-1])

	# ------------ 负采样
	# 如果没有提供负例，根据log-uniform进行负采样
	# 采样公式：P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)
	# 在I2I场景下，class可以理解为item id，排名靠前的item被采样到的概率越大
	# 所以，为了打压高热item，item id编号必须根据item的热度降序编号
	# 越热门的item，排前越靠前，被负采样到的概率越高
	if sampled_values is None:
		sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
		true_classes=labels,# 正例的item ids
		num_true=num_true,
		num_sampled=num_sampled,
		unique=True,
		range_max=num_classes,
		seed=seed)
		
	# sampled: [num_sampled]，一个batch内的所有正样本，共享一批负样本
	# true_expected_count: [batch_size, num_true]，正例在log-uniform采样分布中的概率，接下来修正logit时用得上
	# sampled_expected_count: [num_sampled]，负例在log-uniform采样分布中的概率，接下来修正logit时用得上
	sampled, true_expected_count, sampled_expected_count = (
		array_ops.stop_gradient(s) for s in sampled_values)

	# ------------ Embedding
	# labels_flat is a [batch_size * num_true] tensor
	# sampled is a [num_sampled] int tensor
	# all_ids: [batch_size * num_true + num_sampled]的整数数组，集中了所有正负item ids
	all_ids = array_ops.concat([labels_flat, sampled], 0)	
	# 给batch中出现的所有item，无论正负，进行embedding
	all_w = embedding_ops.embedding_lookup(weights, all_ids, ...)
	
	# true_w: [batch_size * num_true, dim]
	# 从all_w中抽取出对应正例的item embedding
	true_w = array_ops.slice(all_w, [0, 0],
		array_ops.stack([array_ops.shape(labels_flat)[0], -1]))

	# sampled_w: [num_sampled, dim]
	# 从all_w中抽取出对应负例的item embedding
	sampled_w = array_ops.slice(all_w,
		array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])

	# ------------ 计算center item与每个negative context item的匹配度
	# inputs: 可以理解成center item embedding，[batch_size, dim]
	# sampled_w: 负例item的embedding，[num_sampled, dim]
	# sampled_logits: [batch_size, num_sampled]
	sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)
	
	# ------------ 计算center item与每个positive context item的匹配度
	# inputs: 可以理解成center item embedding，[batch_size, dim]
	# true_w：正例item embedding，[batch_size * num_true, dim]
	# row_wise_dots：是element-wise相乘的结果，[batch_size, num_true, dim]	
	......
	row_wise_dots = math_ops.multiply(
		array_ops.expand_dims(inputs, 1),
		array_ops.reshape(true_w, new_true_w_shape))
	......
	# _sum_rows是把所有dim上的乘积相加，得到dot-product的结果
	# true_logits: [batch_size,num_true]
	true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
	......

	# ------------ 修正结果
	# 如果采样到的负例，恰好也是正例，就要补救
	if remove_accidental_hits:
		......
		# 补救方法是在冲突的位置(sparse_indices)的负例logits(sampled_logits)
		# 加上一个非常大的负数acc_weights（值为-FLOAT_MAX）
		# 这样在计算softmax时，相应位置上的负例对应的exp值=0，就不起作用了
		sampled_logits += gen_sparse_ops.sparse_to_dense(
				sparse_indices,
				sampled_logits_shape,
				acc_weights,
				default_value=0.0,
				validate_indices=False)
	
	if subtract_log_q: # 如果是NEG Loss，subtract_log_q=False
		# 对匹配度做修正，对应上边公式中的
		# G(x,y)=F(x,y)-log Q(y|x)
		# item热度越高，被修正得越多
		true_logits -= math_ops.log(true_expected_count)
		sampled_logits -= math_ops.log(sampled_expected_count)

	# ------------ 返回结果
	# true_logits：[batch_size,num_true]
	# sampled_logits: [batch_size, num_sampled]
	# out_logits：[batch_size, num_true + num_sampled]
	out_logits = array_ops.concat([true_logits, sampled_logits], 1)
	
	# We then divide by num_true to ensure the per-example
	# labels sum to 1.0, i.e. form a proper probability distribution.
	# 如果num_true=n，那么每行样本的label就是[1/n,1/n,...,1/n,0,0,...,0]的形式
	# 对于下游的sigmoid loss或softmax loss，属于soft label
	out_labels = array_ops.concat([
		array_ops.ones_like(true_logits) / num_true,
		array_ops.zeros_like(sampled_logits)], 1)

	return out_logits, out_labels