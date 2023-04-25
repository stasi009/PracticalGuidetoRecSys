
import numpy as np
K = ...  # 总目标数
M = ...  # 一次实验中要同时实验几组参数


def draw_weights(mu, sigma):
    weights = np.zeros((M, K))
    for j in range(K):
        # weights的第j列，代表第j个目标的不同实验组的权重
        weights[:, j] = np.random.normal(loc=mu[j], scale=sigma[j]+1e-17, size=(M,))
    return weights


def retain_top_weights(rewards, topN):
    # rewards[i][0]是第i组实验的reward（业务指标）
    # 按各组实验的业务指标从大到小排序
    rewards.sort(key=lambda x: x[0], reverse=True)

    top_weights = []
    for i in range(topN):
        # rewards[i][1]是第i组实验的K个权重
        top_weights.append(rewards[i][1])

    return np.asarray(top_weights)


# 参数初始化, mu和sigma都是K维向量
mu = np.zeros(K)
sigma = np.ones(K) * init_sigma  # init_sigma是sigma的初始值

for t in range(MaxRounds):  # MaxRounds最多实验的轮数

    # 从mu和sigma指定的正态分布中，抽取M组超参
    # weights的形状是[M,K]，每行代表给一组实验的K个权重
    weights = draw_weights(mu, sigma)

    # do_experiments: 开M组小流量进行实验，返回M个实验结果
    # rewards是M长的list，每个元素是一个tuple
    # rewards[i][0]是第i组实验的reward（业务指标）
    # rewards[i][1]是第i组实验的K个权重
    rewards = do_experiments(weights)

    # 提取效果最好的topN组超参数
    # top_weights: [topN,K]
    top_weights = retain_top_weights(rewards, topN)

    # 用topN组超参数，更新mu和sigma
    mu = top_weights.mean(axis=0)
    sigma = top_weights.std(axis=0)
