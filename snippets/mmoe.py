

class MMOE:
    def __init__(self, expert_dnn_config, num_task, num_expert=None, ):
        # expert_dnn_config是一个list
        # expert_dnn_config[i]是第i个expert的配置
        self._expert_dnn_configs = expert_dnn_config
        self._num_expert = len(expert_dnn_config)
        self._num_task = num_task

    def gate(self, unit, deep_fea, name):
        # deep_fea还是底层的输入向量，和喂给expert的是同一个向量
        # unit：应该是experts的数量
        fea = tf.layers.dense(inputs=deep_fea, units=unit,)

        # fea: [B,N]，N是experts的个数
        # 代表对于某个task，每个expert的贡献程度
        fea = tf.nn.softmax(fea, axis=1)
        return fea

    def __call__(self, deep_fea):
        """ 
        输入deep_fea: 底层的输入向量
        输出：一个长度为T的数组，T是task的个数，其中第i个元素是Expert层给第i个task的输入
        """
        expert_fea_list = []
        for expert_id in range(self._num_expert):
            # expert_dnn_config是expert_id对应的expert的配置
            # 比如有几层、每一层采用什么样的激活函数、......
            expert_dnn_config = self._expert_dnn_configs[expert_id]
            expert_dnn = DNN(expert_dnn_config, ......)

            expert_fea = expert_dnn(deep_fea)  # 单个expert的输出
            expert_fea_list.append(expert_fea)

        # 假设有N个expert，每个expert的输出是expert_fea，其形状是[B,D]
        # B=batch_size, D=每个expert输出的维度
        # experts_fea是N个expert_fea拼接成的向量，形状是[B,N,D]
        experts_fea = tf.stack(expert_fea_list, axis=1)

        task_input_list = []  # 给每个task tower的输入
        for task_id in range(self._num_task):
            # gate: [B,N]，N是experts的个数, 代表对于某个task，每个expert的贡献程度
            gate = self.gate(self._num_expert, deep_fea, ......)
            # gate: 变形成[B,N,1]
            gate = tf.expand_dims(gate, -1)

            # experts_fea: [B,N,D]
            # gate: [B,N,1]
            # task_input: [B,N,D]，根据gate给每个expert的输出加权后的结果
            task_input = tf.multiply(experts_fea, gate)
            # task_input: [B,D]，每个expert的输出加权相加
            task_input = tf.reduce_sum(task_input, axis=1)

            task_input_list.append(task_input)
        return task_input_list


mmoe_layer = MMOE(......)

# feature_dict是每个field的输入
# 通过input_layer的映射，映射成一个向量features
features = input_layer(feature_dict, 'all')
# task_input_list是一个长度为T的数组，T是task的个数
# task_input_list[i]是Expert层给第i个task的输入
# 形状是[B,D],B=batch_size, D是每个expert的输出维度
task_input_list = mmoe_layer(features)

tower_outputs = {}
for i, task_tower_cfg in enumerate(model_config.task_towers):
    # task_tower_cfg是第i个task tower的配置
    # 比如：当前task的名字、task tower有几层、每层的激活函数等
    tower_name = task_tower_cfg.tower_name

    # 构建针对第i个task的Tower网络结构
    tower_dnn = DNN(task_tower_cfg, ......)

    # task_input_list[i]是Expert层给第i个task的输入
    # tower_output是第i个task的输出
    tower_output = tower_dnn(task_input_list[i])
    tower_outputs[tower_name] = tower_output
