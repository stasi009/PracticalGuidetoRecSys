
class Server
{
public:
    Server()
    {
        // server_w_是app_id=0的KV数据库，用于读写各特征的一阶权重
        server_w_ = new ps::KVServer<float>(0);
        // 处理一阶权重的具体逻辑，实现在KVServerSGDHandle_w中
        server_w_->set_request_handle(SGD::KVServerSGDHandle_w());

        // server_v_是app_id=1的KV数据库，用于读写各特征的embedding
        server_v_ = new ps::KVServer<float>(1);
        // 处理embedding的具体逻辑，实现在KVServerSGDHandle_v中
        server_v_->set_request_handle(SGD::KVServerSGDHandle_v());
    }
    ~Server() {}
    ps::KVServer<float> *server_w_;
    ps::KVServer<float> *server_v_;
};

typedef struct SGDEntry_v
{
    SGDEntry_v(int k = v_dim)
    {
        w.resize(k, 0.001);
    }
    std::vector<float> w;
} sgdentry_v;

struct KVServerSGDHandle_v
{
    // operator()是Server端处理每个Pull & Push请求的加高函数
    // req_meta包含请求的元信息，req_data包含请求的数据
    // server是调用这个回调函数的KVServer实例，提供一些API可在回调函数中使用
    void operator()(const ps::KVMeta &req_meta, const ps::KVPairs<float> &req_data, ps::KVServer<float> *server)
    {
        size_t keys_size = req_data.keys.size(); // keys_size代表了一共请求了多少个特征的参数
        size_t vals_size = req_data.vals.size();
        ps::KVPairs<float> res; // 用于填充回复结果

        if (req_meta.pull) // 如果是一个Pull请求
        {
            res.keys = req_data.keys; // 请求了哪些特征，就是回复哪些特征
            // v_dim是每个embedding的长度，一共有keys_size个特征
            // 结果的vals一共要开辟长度=keys_size * v_dim的数组
            res.vals.resize(keys_size * v_dim);
        }

        for (size_t i = 0; i < keys_size; ++i) // 遍历请求的每个特征
        {
            ps::Key key = req_data.keys[i]; // key是请求的第i个特征的ID

            // store是KVServerSGDHandle_v的成员变量，是一个unordered_map
            // SGDEntry_v就是只有一个vector<float>成员的struct
            SGDEntry_v &val = store[key]; // 根据请求的特征的ID，找到Server端存储的embedding

            for (int j = 0; j < v_dim; ++j) // 遍历embedding的每一位
            {
                if (req_meta.push) // 如果是push请求，就是要更新本地的embedding
                {
                    // 请求数据req_data的数据域vals，是所有特征embedding的梯度拼接成的大向量
                    // g表示对第i个特征的embedding的第j位的梯度
                    float g = req_data.vals[i * v_dim + j];
                    // val是指向本地存储的引用，这里用SGD算法更新第i个特征的embedding的第j位
                    val.w[j] -= learning_rate * g;
                }
                else // 否则就是pull请求，从Server端提取embedding
                {
                    // val.w就是本地存储的第i个特征的embedding
                    // 这里是将其按位复制到结果res.vals中
                    res.vals[i * v_dim + j] = val.w[j];
                }
            }                            // for遍历每一位
        }                                // for遍历每个特征
        server->Response(req_meta, res); // 回复给worker
    }

private:
    // 用一个map来存储各特征的embedding
    // Key是一个特征的唯一标识ID，sgdentry_v是对一个vector<float>的封装，用于存储embedding向量
    std::unordered_map<ps::Key, sgdentry_v> store;
};

FMWorker(......)
{
    // app_id=0，与Server端server_w_的app_id相同，负责特征一阶权重的通信
    kv_w = new ps::KVWorker<float>(0);
    // app_id=1，与Server端server_v_的app_id相同，负责特征Embedding的通信
    kv_v = new ps::KVWorker<float>(1);
    ......
}

void FMWorker::update(int start, int end) // 运行在独立线程中，训练mini-batch[start:end]这部分数据
{
    size_t idx = 0;
    auto unique_keys = std::vector<ps::Key>();

    // ******************* 遍历分配给自己的数据，统计其中包含了哪些非零特征
    for (int row = start; row < end; ++row)
    {
        // sample_size是第row行样本内部包含的特征个数
        int sample_size = train_data->fea_matrix[row].size();

        for (int j = 0; j < sample_size; ++j) // 遍历当前样本的每个非零特征
        {
            idx = train_data->fea_matrix[row][j].fid;
            ......(unique_keys).push_back(idx); // idx就是feature id
        }
    } // 遍历每条样本

    // 把这部分数据中出现的所有feature_id去重
    // 因为向ps server pull参数的时候，没必要重复pull相同的key
    std::sort((unique_keys).begin(), (unique_keys).end());
    (unique_keys).erase(unique((unique_keys).begin(), (unique_keys).end()), (unique_keys).end());
    int keys_size = (unique_keys).size(); // 去重后的非零feature_id个数

    // ******************* 用去重后的非零feature_ids从ps server拉取最新的参数
    // 拉取这部分样本所包含的非零特征的一阶权重 "w"
    auto w = std::vector<float>();
    kv_w->Wait(kv_w->Pull(unique_keys, &w));

    // 拉取这部分样本所包含的非零特征的的embedding "v"
    auto v = std::vector<float>();
    kv_v->Wait(kv_v->Pull(unique_keys, &v));

    // ******************* 前代
    // loss这个名字取得不好，其实里面存储的是每个样本loss->final logit的导数
    auto loss = std::vector<float>(end - start);
    ...... calculate_loss(w, v, ..., unique_keys, start, end, ..., loss); // loss是计算结果

    // ******************* 回代
    // push_w_gradient: 存放loss对各feature的一阶权重'w'上的导数
    auto push_w_gradient = std::vector<float>(keys_size);

    // push_v_gradient: 存放loss对各feature的embedding的每一位上的导数
    // 由于embedding是个向量，
    // 所以需要开辟的空间是keys_size(去重后有多少feaute)*v_dim_(每个embedding的长度)
    auto push_v_gradient = std::vector<float>(keys_size * v_dim_);

    // 计算结果输出至push_w_gradient和push_v_gradient
    calculate_gradient(..., unique_keys, start, end, v, ..., loss,
                       push_w_gradient, push_v_gradient);

    // ******************* 向Sever push梯度，让server端更新模型参数
    // 注意！！！这里的Wait只是等待异步Push完成
    // 每个worker thread各push各自的，完全没有与其他worker同步，因此这里实现的还是异步模式
    kv_w->Wait(kv_w->Push(unique_keys, push_w_gradient));
    kv_v->Wait(kv_v->Push(unique_keys, push_v_gradient));

    --gradient_thread_finish_num; // 表示有一个线程完成训练
}

void FMWorker::batch_training(ThreadPool *pool)
{
    ...... for (int epoch = 0; epoch < epochs; ++epoch) // 训练上若干epoch
    {
        xflow::LoadData train_data_loader(train_data_path, block_size << 20);
        train_data = &(train_data_loader.m_data); // 用于存储读进来的一个mini-batch的数据

        while (1) // 循环直到将本次的训练数据都读完
        {
            // 读取一个mini-batch的数据存放在train_data->fea_matrix中
            // fea_matrix包含一个batch的样本，是一个vector<vector<kv>>
            // 外层vector代表各个样本，内层的vector代表一个样本中的各个feature
            train_data_loader.load_minibatch_hash_data_fread();

            // 把这个mini-batch的训练数据平分到各个thread上
            // 每个thread分到thread_size条训练数据
            int thread_size = train_data->fea_matrix.size() / core_num;
            gradient_thread_finish_num = core_num;

            for (int i = 0; i < core_num; ++i) // 遍历并启动各训练线程
            {
                int start = i * thread_size;     // 当前线程分到的数据，在mini-batch中的起始位置
                int end = (i + 1) * thread_size; // 当前线程分到的数据，在mini-batch中的截止位置
                // 启动线程运行FMWorker::update，训练当前mini-batch[start:end]这部分局部数据
                pool->enqueue(std::bind(&FMWorker::update, this, start, end));
            }
            while (gradient_thread_finish_num > 0)
            { // 等待所有训练thread结束
                usleep(5);
            }
        }
    }
}