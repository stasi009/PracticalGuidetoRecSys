
import numpy as np
from scipy.optimize import minimize
import tensorflow as tf

def pareto_efficient_weights(prev_w, c, G):
    """
    G: [K,m]，G[i,:]是第i个task对所有参数的梯度，m是所有待优化参数的个数
    c: [K,1] 每个目标权重的下限约束
    prev_w: [K,1] 上一轮迭代各loss的权重
    """
    # ------------------ 暂时忽略非负约束
    # 对应公式(7-30)
    GGT = np.matmul(G, np.transpose(G))  # [K, K]
    e = np.ones(np.shape(prev_w))  # [K, 1]

    m_up = np.hstack((GGT, e))  # [K, K+1]
    m_down = np.hstack((np.transpose(e), np.zeros((1, 1))))  # [1, K+1]
    M = np.vstack((m_up, m_down))  # [K+1, K+1]

    z = np.vstack((-np.matmul(GGT, c), 1 - np.sum(c)))  # [K+1, 1]

    MTM = np.matmul(np.transpose(M), M)
    w_hat = np.matmul(np.matmul(np.linalg.inv(MTM), M), z)  # [K+1, 1]
    w_hat = w_hat[:-1]  # [K, 1]
    w_hat = np.reshape(w_hat, (w_hat.shape[0],))  # [K,]

    # ------------------ 重新考虑非负约束时的最优解
    return active_set_method(w_hat, prev_w, c)


def active_set_method(w_hat, prev_w, c):
    # ------------------ 对应公式(7-30)
    A = np.eye(len(c))
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 等式约束
    bounds = [[0., None] for _ in range(len(w_hat))]  # 不等式约束，要求所有weight都非负
    result = minimize(lambda x: np.linalg.norm(A.dot(x) - w_hat),
                      x0=prev_w,  # 上次的权重作为本次的初值
                      method='SLSQP',
                      bounds=bounds,
                      constraints=cons)
    # ------------------ 对应公式(7-31)
    return result.x + c


# ------------------ 定义模型
ph_wa = tf.placeholder(tf.float32)  # A loss的权重的占位符
ph_wb = tf.placeholder(tf.float32)  # B loss的权重的占位符
W = ...  # 模型要优化的所有参数

loss_a = loss_fun_a(...)  # A目标的loss
loss_b = loss_fun_b(...)  # B目标的loss
loss = ph_wa * loss_a + ph_wb * loss_b

a_gradients = tf.gradients(loss_a, W)  # A目标loss对所有权重的梯度
b_gradients = tf.gradients(loss_b, W)  # B目标loss对所有权重的梯度

optimizer = tf.train.AdamOptimizer(...)
train_op = optimizer.minimize(loss)  # 优化参数的op

# ------------------ 开始训练
sess = tf.Session()

w_a, w_b = 0.5, 0.5  # 权重初值
c = ...  # 公式(7-26)各权重的下限
for step in range(0, max_train_steps):
    res = sess.run([a_gradients, b_gradients, train_op],
                   feed_dict={ph_wa: w_a, ph_wa: w_b})

    # 当前的梯度矩阵
    G = np.hstack((res[0][0], res[1][0]))
    G = np.transpose(G)

    # 得到新一轮的各目标的最优权重
    w_a, w_b = pareto_efficient_weights(prev_w=np.asarray(w_a, w_b),
                                        c=c,  # 各目标权重的下限约束
                                        G=G)  # 梯度矩阵
