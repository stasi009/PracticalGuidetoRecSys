
#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include "../Utils/utils.h"
#include "../Mem/mem_pool.h"
#include "../Mem/my_allocator.h"
#include "model_bin_file.h"

using namespace std;

template<typename T>
double ftrl_model<T>::predict(const vector<pair<string, double> >& x, double bias, vector<ftrl_model_unit<T>*>& theta, vector<double>& sum)
{
    /**
     * 输入x：一个样本的所有非零特征，由一系列键值对组成，key=特征名称, value=特征数值 
     * 输入theta：模型参数，包含每个特征的一阶权重w，和Embedding v
     * 输出：该样本的logit
     */
    double result = 0;
    result += bias;

    // *************** 遍历所有非零特征，累加FM的一阶部分
    for(int i = 0; i < x.size(); ++i)
    {
        // theta[i]->wi：第i个特征的一阶权重w
        // x[i].second：第i个特征的特征值
        result += theta[i]->wi * x[i].second;
    }

    // *************** FM中所有非零特征的二阶交叉
    double sum_sqr, d;
    //遍历embedding的每一位, factor_num是embedding的长度
    for(int f = 0; f < factor_num; ++f)
    {
        sum[f] = sum_sqr = 0.0;
        for(int i = 0; i < x.size(); ++i)//遍历所有非零特征
        {
            // theta[i]->vi(f)：第i个特征的embedding的第f位
            d = theta[i]->vi(f) * x[i].second;
            sum[f] += d;
            sum_sqr += d * d;
        }
        // 和的平方 - 平方的和，就等于所有二阶交叉之和
        result += 0.5 * (sum[f] * sum[f] - sum_sqr);
    }
    return result;
}

template<typename T>
void ftrl_trainer<T>::train(int y, const vector<pair<string, double> >& x)
{
    /**
     * 完成一个样本的前代与回代
     * 输入x：当前样本的所有非零特征，由一系列键值对组成，key=特征名称, value=特征数值 
     * 输入y：当前样本的label, 0/1
     */
    // ************* 初始化，类似于从PS中提取要用到的权重参数
    ftrl_model_unit<T>* thetaBias = pModel->get_or_init_model_unit_bias();
    vector<ftrl_model_unit<T>*> theta(x.size(), NULL);
    int xLen = x.size();
    for(int i = 0; i < xLen; ++i)//遍历所有非零特征
    {
        const string& index = x[i].first;// index就是特征名称
        theta[i] = pModel->get_or_init_model_unit(index);//获取该特征最新状态
    }
    
    // ************* 前代
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit<T>& mu = i < xLen ? *(theta[i]) : *thetaBias;
        if((i < xLen && k1) || (i == xLen && k0))
        {
            feaLocks[i]->lock();
            if(fabs(mu.w_zi) <= w_l1)
            {
                mu.wi = 0.0;
            }
            else
            {

                mu.wi = (-1) *
                    (1 / (w_l2 + (w_beta + sqrt(mu.w_ni)) / w_alpha)) *
                    (mu.w_zi - utils::sgn(mu.w_zi) * w_l1);
            }
            feaLocks[i]->unlock();
        }
    }
    //update v via FTRL
    for(int i = 0; i < xLen; ++i)
    {
        ftrl_model_unit<T>& mu = *(theta[i]);
        for(int f = 0; f < pModel->factor_num; ++f)
        {
            feaLocks[i]->lock();
            T& vif = mu.vi(f);
            T& v_nif = mu.v_ni(f);
            T& v_zif = mu.v_zi(f);
            if(v_nif > 0)
            {
                if(force_v_sparse && 0.0 == mu.wi)
                {
                    vif = 0.0;
                }
                else if(fabs(v_zif) <= v_l1)
                {
                    vif = 0.0;
                }
                else
                {
                    vif = (-1) *
                        (1 / (v_l2 + (v_beta + sqrt(v_nif)) / v_alpha)) *
                        (v_zif - utils::sgn(v_zif) * v_l1);
                }
            }
            feaLocks[i]->unlock();
        }
    }
    vector<double> sum(pModel->factor_num);
    double bias = thetaBias->wi;
    double p = pModel->predict(x, bias, theta, sum);
    double mult = y * (1 / (1 + exp(-p * y)) - 1);
    //update w_n, w_z
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit<T>& mu = i < xLen ? *(theta[i]) : *thetaBias;
        double xi = i < xLen ? x[i].second : 1.0;
        if((i < xLen && k1) || (i == xLen && k0))
        {
            feaLocks[i]->lock();
            double w_gi = mult * xi;
            double w_si = 1 / w_alpha * (sqrt(mu.w_ni + w_gi * w_gi) - sqrt(mu.w_ni));
            mu.w_zi += w_gi - w_si * mu.wi;
            mu.w_ni += w_gi * w_gi;
            feaLocks[i]->unlock();
        }
    }
    //update v_n, v_z
    for(int i = 0; i < xLen; ++i)
    {
        ftrl_model_unit<T>& mu = *(theta[i]);
        const double& xi = x[i].second;
        for(int f = 0; f < pModel->factor_num; ++f)
        {
            feaLocks[i]->lock();
            T& vif = mu.vi(f);
            T& v_nif = mu.v_ni(f);
            T& v_zif = mu.v_zi(f);
            double v_gif = mult * (sum[f] * xi - vif * xi * xi);
            double v_sif = 1 / v_alpha * (sqrt(v_nif + v_gif * v_gif) - sqrt(v_nif));
            v_zif += v_gif - v_sif * vif;
            v_nif += v_gif * v_gif;
            //有的特征在整个训练集中只出现一次，这里还需要对vif做一次处理
            if(force_v_sparse && v_nif > 0 && 0.0 == mu.wi)
            {
                vif = 0.0;
            }
            feaLocks[i]->unlock();
        }
    }
}