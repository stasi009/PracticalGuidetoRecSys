class WideDeepModel(keras_training.Model):

    def call(self, inputs, training=None):
        linear_inputs, dnn_inputs = inputs

        # Wide部分前代，得到logit
        linear_output = self.linear_model(linear_inputs)
        # Deep部分前代，得到logit
        dnn_output = self.dnn_model(dnn_inputs)

        # Wide logits与Deep logits相加
        output = tf.nest.map_structure(
            lambda x, y: (x + y), linear_output, dnn_output)

        # 一般采用sigmoid激活函数，由logit得到ctr
        return tf.nest.map_structure(self.activation, output)

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # ------------- 前代
        # GradientTape是TF2自带功能，GradientTape内的操作能够自动求导
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # 前代
            # 由外界设置的compiled_loss计算loss
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)

        # ------------- 回代
        linear_vars = self.linear_model.trainable_variables  # Wide部分的待优化参数
        dnn_vars = self.dnn_model.trainable_variables  # Deep部分的待优化参数

        # 分别计算loss对linear_vars的导数linear_grads
        # 和loss对dnn_vars的导数dnn_grads
        linear_grads, dnn_grads = tape.gradient(loss, (linear_vars, dnn_vars))

        # 一般用FTRL优化Wide侧，以得到更稀疏的解
        linear_optimizer = self.optimizer[0]
        linear_optimizer.apply_gradients(zip(linear_grads, linear_vars))

        # 用Adam、Adagrad优化Deep侧
        dnn_optimizer = self.optimizer[1]
        dnn_optimizer.apply_gradients(zip(dnn_grads, dnn_vars))