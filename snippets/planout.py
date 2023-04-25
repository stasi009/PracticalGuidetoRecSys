
from planout.experiment import SimpleExperiment

# ============== 配置实验
class XRecallExperiment(SimpleExperiment):
    def assign(self, params, userid):
        # 是否采用x_recall这种新召回策略，遵循Bernoulli分布
        # 有80%的可能性不采用，有20%的概率采用
        params.use_x_recall = BernoulliTrial(p=0.2, unit=userid)

class YRecallExperiment(SimpleExperiment):
    def assign(self, params, userid):
        # 是否采用y_recall这种新召回策略，遵循Bernoulli分布
        # 有70%的可能性不采用，有30%的概率采用
        params.use_y_recall = BernoulliTrial(p=0.3, unit=userid)

# ============== 在运行X实验的进程或线程中
recall_exp = XRecallExperiment(userid=session['userid'])
# PlanOut调用get时，将experiment_name.parameter_name.unit_id三者一起hash
# 这里是Hash(XRecallExperiment.use_x_recall.userid)
# 保证进入XRecallExperiment这个实验时，流量被重新打散
if recall_exp.get('use_x_recall'):
    recalled_items = x_recall()  # x_recall执行具体的召回逻辑
else:
    recalled_items = []

# ============== 在运行Y实验的进程或线程中
recall_exp = YRecallExperiment(userid=session['userid'])
# PlanOut调用get时，将experiment_name.parameter_name.unit_id三者一起hash
# 这里是Hash(YRecallExperiment.use_y_recall.userid)
# 保证进入YRecallExperiment这个实验时，流量被重新打散
if recall_exp.get('use_y_recall'):
    recalled_items = y_recall()  # y_recall执行具体的召回逻辑
else:
    recalled_items = []
