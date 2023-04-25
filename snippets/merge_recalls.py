MAX_NUM_RECALLS = 10000000000


def wrong_merge_recalls():

    merged_results = {}

    # recall_method：某一路的召回算法名
    # recall_results：此路召回的结果集
    for recall_method, recall_results in recall_results_list:
        capacity = MAX_NUM_RECALLS - len(merged_results)  # merged_results还剩余的额度
        if capacity == 0:
            # 已经插满了，直接返回，后面的召回的结果被丢弃
            return merged_results

        # 当前召回能插入最终结果集的额度
        quota = min(len(recall_results), capacity)
        # 把当前结果集中的top quota个物料，插入到最终结果集中
        top_recall_results = recall_results[:quota]
        merged_results.update(top_recall_results)

    return merged_results


def correct_merge_recalls():
    merged_results = {}

    while True:
        # recall_method：某一路的召回算法名
        # recall_results：此路召回的结果集
        for recall_method, recall_results in recall_results_list:
            if len(merged_results) == MAX_NUM_RECALLS:
                return merged_results  # 插满了，返回

            if len(recall_results) > 0:  # 当前召回还有余量
                # 弹出当前召回认为的用户最喜欢的item，插入结果集
                top_item = recall_results.pop()
                merged_results.add(top_item)
