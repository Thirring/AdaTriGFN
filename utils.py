# from http.cookiejar import unmatched

import numpy as np
from sklearn import metrics
from tqdm import tqdm

from common_utils import TokenizedData, stanza_processor, clean_blank
from data import label2id, id2label
import torch.nn as nn
from collections import OrderedDict, defaultdict


def get_aspects(tags, length, token_range, ignore_index=-1):
    spans = []
    start, end = -1, -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l][l] == ignore_index:
            continue
        label = id2label[tags[l][l]]
        if label == 'B-A':
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == 'I-A':
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, length - 1])

    return spans


def get_opinions(tags, length, token_range, ignore_index=-1):
    spans = []
    start, end = -1, -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l][l] == ignore_index:
            continue
        label = id2label[tags[l][l]]
        if label == 'B-O':
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == 'I-O':
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, length - 1])

    return spans


class Metric():
    def __init__(self, args, predictions, goldens, bert_lengths, sen_lengths, tokens_ranges, ignore_index=-1):
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        self.bert_lengths = bert_lengths
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = -1
        self.data_num = len(self.predictions)

    def get_spans(self, tags, length, token_range, type):
        spans = []
        start = -1
        for i in range(length):
            l, r = token_range[i]
            if tags[l][l] == self.ignore_index:
                continue
            elif tags[l][l] == type:
                if start == -1:
                    start = i
            elif tags[l][l] != type:
                if start != -1:
                    spans.append([start, i - 1])
                    start = -1
        if start != -1:
            spans.append([start, length - 1])
        return spans

    def find_pair(self, tags, aspect_spans, opinion_spans, token_ranges):
        pairs = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 4
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                if tag_num[3] == 0: continue
                sentiment = -1
                pairs.append([al, ar, pl, pr, sentiment])
        return pairs

    def find_triplet(self, tags, aspect_spans, opinion_spans, token_ranges):
        # label2id = {'N': 0, 'B-A': 1, 'I-A': 2, 'A': 3, 'B-O': 4, 'I-O': 5, 'O': 6, 'negative': 7, 'neutral': 8, 'positive': 9}
        triplets_utm = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * len(label2id)
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1

                if sum(tag_num[7:]) == 0: continue
                sentiment = -1
                if tag_num[9] >= tag_num[8] and tag_num[9] >= tag_num[7]:
                    sentiment = 9
                elif tag_num[8] >= tag_num[7] and tag_num[8] >= tag_num[9]:
                    sentiment = 8
                elif tag_num[7] >= tag_num[9] and tag_num[7] >= tag_num[8]:
                    sentiment = 7
                if sentiment == -1:
                    print('wrong!!!!!!!!!!!!!!!!!!!!')
                    exit()
                triplets_utm.append([al, ar, pl, pr, sentiment])

        return triplets_utm

    def score_aspect(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = get_aspects(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            for spans in golden_aspect_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_aspect_spans = get_aspects(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            for spans in predicted_aspect_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_opinion(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_opinion_spans = get_opinions(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            for spans in golden_opinion_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_opinion_spans = get_opinions(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            for spans in predicted_opinion_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_uniontags(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = get_aspects(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            golden_opinion_spans = get_opinions(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            if self.args.task == 'pair':
                golden_tuples = self.find_pair(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                golden_tuples = self.find_triplet(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            for pair in golden_tuples:
                golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))

            predicted_aspect_spans = get_aspects(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            predicted_opinion_spans = get_opinions(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            if self.args.task == 'pair':
                predicted_tuples = self.find_pair(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                predicted_tuples = self.find_triplet(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            for pair in predicted_tuples:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_uniontags_print(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        all_golden_triplets = []
        all_predicted_triplets = []
        for i in range(self.data_num):
            golden_aspect_spans = get_aspects(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            golden_opinion_spans = get_opinions(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            if self.args.task == 'pair':
                golden_tuples = self.find_pair(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                golden_tuples = self.find_triplet(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            for pair in golden_tuples:
                golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))

            predicted_aspect_spans = get_aspects(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            predicted_opinion_spans = get_opinions(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            if self.args.task == 'pair':
                predicted_tuples = self.find_pair(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                predicted_tuples = self.find_triplet(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            for pair in predicted_tuples:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))

            all_golden_triplets.append(golden_tuples)
            all_predicted_triplets.append(predicted_tuples)

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1, all_golden_triplets, all_predicted_triplets

    def tagReport(self):
        print(len(self.predictions))
        print(len(self.goldens))

        golden_tags = []
        predict_tags = []
        for i in range(self.data_num):
            for r in range(102):
                for c in range(r, 102):
                    if self.goldens[i][r][c] == -1:
                        continue
                    golden_tags.append(self.goldens[i][r][c])
                    predict_tags.append(self.predictions[i][r][c])

        print(len(golden_tags))
        print(len(predict_tags))
        target_names = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
        print(metrics.classification_report(golden_tags, predict_tags, target_names=target_names, digits=4))


from collections import OrderedDict
import torch

from collections import OrderedDict
import torch


def decode_quadruples(pred_tags, tokens, label_map, tokenized_data, threshold=0.5):
    """
    优化后的解码函数：增加了数量限制以防止Epoch 0时的卡死
    """
    L, L, N = pred_tags.shape
    sen_length = len(tokenized_data.inputs.input_ids)

    # 保持原有的mask逻辑
    mask = torch.zeros(L).to(pred_tags.device)
    mask[1:sen_length - 1] = 1
    pred_tags = pred_tags * mask.view(1, L, 1)

    id2label = {v: k for k, v in label_map.items()}

    # 1. 提取对角线标签
    diag_tags = pred_tags.diagonal(dim1=0, dim2=1).transpose(0, 1)

    # 2. 提取 span
    target_spans = extract_spans(diag_tags, label_map['B-T'], label_map['I-T'], threshold)
    argument_spans = extract_spans(diag_tags, label_map['B-A'], label_map['I-A'], threshold)

    # --- [关键优化] 熔断机制 ---
    # 如果预测出的 span 太多（说明模型在乱猜），直接截断，取前30个即可
    # 这样避免了后续循环次数爆炸
    if len(target_spans) > 30:
        target_spans = target_spans[:30]
    if len(argument_spans) > 30:
        argument_spans = argument_spans[:30]
    # -------------------------

    group_labels = ['LGBTQ', 'Racism', 'Region', 'Sexism', 'others', 'non-hate']
    group_indices = [label_map[g] for g in group_labels]
    hate_indices = [label_map['hate'], label_map['non-hate']]

    quadruples = []

    # 5. 遍历组合
    for t_start, t_end in target_spans:
        for a_start, a_end in argument_spans:
            region = pred_tags[t_start:t_end + 1, a_start:a_end + 1, :]
            region_scores = region.mean(dim=(0, 1))

            hate_score = region_scores[hate_indices[0]]
            non_hate_score = region_scores[hate_indices[1]]

            # 简单的逻辑判断
            if hate_score >= non_hate_score and hate_score >= threshold:
                hateful = 'hate'
            elif non_hate_score >= threshold:
                hateful = 'non-hate'
            else:
                continue

            groups = []
            for idx in group_indices:
                if region_scores[idx] >= threshold:
                    groups.append(id2label[idx])

            # 修正 group 逻辑
            if not groups:
                if hateful == 'hate':
                    group_scores = region_scores[group_indices]
                    max_score, max_idx = torch.max(group_scores, dim=0)
                    if max_score >= threshold * 0.7:
                        groups.append(id2label[group_indices[max_idx]])
                    else:
                        continue
                else:
                    groups.append('non-hate')

            if hateful == 'non-hate' and any(g in groups for g in ['LGBTQ', 'Racism', 'Region', 'Sexism', 'others']):
                groups = ['non-hate']
            if hateful == 'hate' and "non-hate" in groups:
                groups.remove("non-hate")

            target_str = tokenized_data.token_range2str(t_start, t_end + 1)
            argument_str = tokenized_data.token_range2str(a_start, a_end + 1)

            if len(target_str) == 0 or len(argument_str) == 0 or len(groups) == 0:
                continue

            quadruples.append((target_str, argument_str, tuple(sorted(groups)), hateful))

    # --- [关键优化] 数量检查 ---
    # 如果生成的四元组依然过多（例如 > 60），说明是噪声数据
    # 此时进行 O(N^2) 的 merge_duplicates 会极慢，直接去重返回即可，跳过复杂合并
    if len(quadruples) > 60:
        # 仅做简单的去重，跳过耗时的子串合并逻辑
        return list(set(quadruples))

    # 只有数量可控时，才执行复杂的合并逻辑
    return merge_duplicates(quadruples, tokenized_data)


def extract_spans(diag_tags, b_label_idx, i_label_idx, threshold):
    """
    从对角线标签提取连续span

    参数:
        diag_tags: 对角线标签 [seq_len, num_labels]
        b_label_idx: B标签索引
        i_label_idx: I标签索引
        threshold: 激活阈值

    返回:
        spans: span列表 [(start, end)]
    """
    spans = []
    current_span = None
    seq_len = diag_tags.size(0)

    for i in range(1, seq_len - 1):  # 跳过[CLS]和[SEP]
        # 检测B标签
        if diag_tags[i, b_label_idx] >= threshold:
            if current_span is not None:
                spans.append(current_span)
            current_span = [i, i]

        # 检测I标签
        elif diag_tags[i, i_label_idx] >= threshold:
            if current_span is not None and i == current_span[1] + 1:
                current_span[1] = i

        # 结束当前span
        else:
            if current_span is not None:
                spans.append(tuple(current_span))
                current_span = None

    # 处理最后一个span
    if current_span is not None:
        spans.append(tuple(current_span))

    return spans


def span_to_string(start, end, tokens):
    """将span索引转换为字符串"""
    # 注意: 序列位置i对应token[i-1]
    start_idx = start - 1
    end_idx = end  # 切片需要+1
    return ''.join(tokens[start_idx:end_idx])


def merge_duplicates(quadruples, tokenized_data):
    """
    优化后的合并重复项逻辑
    """
    # 先进行简单的去重
    unique_quads = list(set(quadruples))

    # 如果数据量依然很大，直接返回，不再尝试合并（为了性能）
    if len(unique_quads) > 50:
        return unique_quads

    # 待移除的集合
    to_remove = set()
    # 待添加的列表
    to_add = []

    n = len(unique_quads)
    # 使用索引遍历，避免修改列表导致的错误
    for i in range(n):
        if i in to_remove: continue

        t1, a1, g1, h1 = unique_quads[i]

        for j in range(n):
            if i == j: continue
            if j in to_remove: continue

            t2, a2, g2, h2 = unique_quads[j]

            # 仅当 group 和 hateful 一致时才尝试合并
            if g1 == g2 and h1 == h2:
                # 尝试合并 Target
                if a1 == a2:
                    # 检查 t1 + t2 是否在原句中（模拟相邻）
                    if (t1 + t2) in tokenized_data.sentence:
                        to_add.append((t1 + t2, a1, g1, h1))
                        to_remove.add(i)
                        to_remove.add(j)
                        break  # 一次只合并一对，避免混乱
                    elif (t2 + t1) in tokenized_data.sentence:
                        to_add.append((t2 + t1, a1, g1, h1))
                        to_remove.add(i)
                        to_remove.add(j)
                        break

                # 尝试合并 Argument
                elif t1 == t2:
                    if (a1 + a2) in tokenized_data.sentence:
                        to_add.append((t1, a1 + a2, g1, h1))
                        to_remove.add(i)
                        to_remove.add(j)
                        break
                    elif (a2 + a1) in tokenized_data.sentence:
                        to_add.append((t1, a2 + a1, g1, h1))
                        to_remove.add(i)
                        to_remove.add(j)
                        break

    # 构建最终结果
    final_quads = []
    # 添加未被合并的
    for i in range(n):
        if i not in to_remove:
            final_quads.append(unique_quads[i])
    # 添加合并后的
    final_quads.extend(to_add)

    # 最后整理格式（Group聚合）
    merged = OrderedDict()
    for t, a, g, h in final_quads:
        key = (t, a, h)
        if key not in merged:
            merged[key] = set(g)
        else:
            merged[key].update(g)

    return [(t, a, tuple(sorted(tuple(g))), h) for (t, a, h), g in merged.items()]


# 评价指标代码
from collections import defaultdict, OrderedDict
import numpy as np

from collections import defaultdict
from difflib import SequenceMatcher


def calculate_metrics(gold_data, pred_data_dict):
    counts_strict = {
        'target': [0, 0, 0],  # tp, g_n, p_n
        'argument': [0, 0, 0],
        'target_argument': [0, 0, 0],
        'target_argument_hateful': [0, 0, 0],
        'target_argument_group_hateful': [0, 0, 0],
    }

    counts_soft = {
        'target': [0, 0, 0],
        'argument': [0, 0, 0],
        'target_argument': [0, 0, 0],
        'target_argument_hateful': [0, 0, 0],
        'target_argument_group_hateful': [0, 0, 0],
    }

    def soft_tp_compute(gold_set, pred_set, mode):
        tp = 0
        if mode == 'target' or mode == 'argument':
            for g in gold_set:
                for p in pred_set:
                    if SequenceMatcher(None, g, p).ratio() >= 0.5:
                        tp += 1
                        break
        if mode == 'target_argument':
            for g_t, g_a in gold_set:
                for p_t, p_a in pred_set:
                    if SequenceMatcher(None, g_t, p_t).ratio() >= 0.5 and SequenceMatcher(None, g_a, p_a).ratio() >= 0.5:
                        tp += 1
                        break
        if mode == 'target_argument_hateful':
            for g_t, g_a, g_h in gold_set:
                for p_t, p_a, p_h in pred_set:
                    if (SequenceMatcher(None, g_t, p_t).ratio() >= 0.5 and SequenceMatcher(None, g_a, p_a).ratio() >= 0.5
                            and g_h == p_h):
                        tp += 1
                        break
        if mode == 'target_argument_group_hateful':
            for g_t, g_a, g_g, g_h in gold_set:
                for p_t, p_a, p_g, p_h in pred_set:
                    if (SequenceMatcher(None, g_t, p_t).ratio() >= 0.5 and SequenceMatcher(None, g_a, p_a).ratio() >= 0.5
                            and g_h == p_h and g_g == p_g):
                        tp += 1
                        break
        return tp

    unmatched_dict = {}

    for data_item in gold_data:
        id = data_item['id']
        gold_quad_list = extract_gold_quads(data_item)
        if len(gold_quad_list) == 0 or id not in pred_data_dict:
            continue
        pred_quad_list = pred_data_dict[id]

        # T
        gold_target_set = {quad[0] for quad in gold_quad_list}
        pred_target_set = {quad[0] for quad in pred_quad_list}

        counts_strict['target'][0] += len(gold_target_set & pred_target_set)
        counts_strict['target'][1] += len(gold_target_set)
        counts_strict['target'][2] += len(pred_target_set)

        counts_soft['target'][0] += soft_tp_compute(gold_target_set, pred_target_set, mode='target')
        counts_soft['target'][1] += len(gold_target_set)
        counts_soft['target'][2] += len(pred_target_set)

        # A
        gold_argument_set = {quad[1] for quad in gold_quad_list}
        pred_argument_set = {quad[1] for quad in pred_quad_list}

        counts_strict['argument'][0] += len(gold_argument_set & pred_argument_set)
        counts_strict['argument'][1] += len(gold_argument_set)
        counts_strict['argument'][2] += len(pred_argument_set)

        counts_soft['argument'][0] += soft_tp_compute(gold_argument_set, pred_argument_set, mode='argument')
        counts_soft['argument'][1] += len(gold_argument_set)
        counts_soft['argument'][2] += len(pred_argument_set)

        # T-A
        gold_target_argument_set = {quad[:2] for quad in gold_quad_list}
        pred_target_argument_set = {quad[:2] for quad in pred_quad_list}

        counts_strict['target_argument'][0] += len(gold_target_argument_set & pred_target_argument_set)
        counts_strict['target_argument'][1] += len(gold_target_argument_set)
        counts_strict['target_argument'][2] += len(pred_target_argument_set)

        counts_soft['target_argument'][0] += soft_tp_compute(gold_target_argument_set, pred_target_argument_set, mode='target_argument')
        counts_soft['target_argument'][1] += len(gold_target_argument_set)
        counts_soft['target_argument'][2] += len(pred_target_argument_set)

        # T-A-H
        gold_target_argument_hate_set = {(*quad[:2], quad[-1]) for quad in gold_quad_list}
        pred_target_argument_hate_set = {(*quad[:2], quad[-1]) for quad in pred_quad_list}

        counts_strict['target_argument_hateful'][0] += len(gold_target_argument_hate_set & pred_target_argument_hate_set)
        counts_strict['target_argument_hateful'][1] += len(gold_target_argument_hate_set)
        counts_strict['target_argument_hateful'][2] += len(pred_target_argument_hate_set)

        counts_soft['target_argument_hateful'][0] += soft_tp_compute(gold_target_argument_hate_set, pred_target_argument_hate_set, mode='target_argument_hateful')
        counts_soft['target_argument_hateful'][1] += len(gold_target_argument_hate_set)
        counts_soft['target_argument_hateful'][2] += len(pred_target_argument_hate_set)

        # quad
        gold_quad_set = {quad for quad in gold_quad_list}
        pred_quad_set = {quad for quad in pred_quad_list}

        unmatched_dict[id] = {
            'unmatched_gold': list(gold_quad_set - pred_quad_set),
            'unmatched_pred': list(pred_quad_set - gold_quad_set),
        }

        counts_strict['target_argument_group_hateful'][0] += len(gold_quad_set & pred_quad_set)
        counts_strict['target_argument_group_hateful'][1] += len(gold_quad_set)
        counts_strict['target_argument_group_hateful'][2] += len(pred_quad_set)

        counts_soft['target_argument_group_hateful'][0] += soft_tp_compute(gold_quad_set, pred_quad_set, mode='target_argument_group_hateful')
        counts_soft['target_argument_group_hateful'][1] += len(gold_quad_set)
        counts_soft['target_argument_group_hateful'][2] += len(pred_quad_set)

    def compute_metrics(counts_dict):
        metrics = {}
        for key in counts_dict:
            total_tp, total_gold, total_pred = counts_dict[key]

            precision = total_tp / total_pred if total_pred > 0 else 0
            recall = total_tp / total_gold if total_gold > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics[key] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        return metrics

    return {
        'strict': compute_metrics(counts_strict),
        'soft': compute_metrics(counts_soft)
    }, unmatched_dict


def _calculate_metrics(gold_data, pred_data_dict):
    # 初始化计数器 - 分别用于严格匹配和相似度匹配
    counts_strict = {
        'target': defaultdict(lambda: [0, 0, 0]),
        'argument': defaultdict(lambda: [0, 0, 0]),
        'group': defaultdict(lambda: [0, 0, 0]),
        'hateful': defaultdict(lambda: [0, 0, 0]),
        'target_argument': defaultdict(lambda: [0, 0, 0]),
        'target_argument_group': defaultdict(lambda: [0, 0, 0]),
        'target_argument_hateful': defaultdict(lambda: [0, 0, 0]),
        'target_argument_group_hateful': defaultdict(lambda: [0, 0, 0]),
    }

    counts_soft = {
        'target': defaultdict(lambda: [0, 0, 0]),
        'argument': defaultdict(lambda: [0, 0, 0]),
        'group': defaultdict(lambda: [0, 0, 0]),
        'hateful': defaultdict(lambda: [0, 0, 0]),
        'target_argument': defaultdict(lambda: [0, 0, 0]),
        'target_argument_group': defaultdict(lambda: [0, 0, 0]),
        'target_argument_hateful': defaultdict(lambda: [0, 0, 0]),
        'target_argument_group_hateful': defaultdict(lambda: [0, 0, 0]),
    }

    # 辅助函数：更新计数器
    def update_counts(counts_dict, level, pred_val, gold_val, is_group=False, matched=False):
        # 更新预测计数
        if is_group:
            counts_dict[level][pred_val][1] += 1
            if matched:
                counts_dict[level][pred_val][0] += 1  # TP
        else:
            counts_dict[level][pred_val][1] += 1  # 预测计数
            if matched:
                counts_dict[level][gold_val][0] += 1  # TP (使用gold_val作为key)

    # 统计所有真实四元组的计数（同时初始化strict和soft）
    for item in gold_data:
        if item['id'] not in pred_data_dict:
            continue
        gold_quads = extract_gold_quads(item)
        if len(gold_quads) == 0:
            continue
        for target, argument, groups, hateful in gold_quads:
            # 更新strict和soft的真实计数（相同）
            for counts_dict in [counts_strict, counts_soft]:
                counts_dict['target'][target][2] += 1
                counts_dict['argument'][argument][2] += 1
                counts_dict['hateful'][hateful][2] += 1
                for g in groups:
                    counts_dict['group'][g][2] += 1
                counts_dict['target_argument'][(target, argument)][2] += 1
                counts_dict['target_argument_group'][(target, argument, tuple(sorted(groups)))][2] += 1
                counts_dict['target_argument_hateful'][(target, argument, hateful)][2] += 1
                counts_dict['target_argument_group_hateful'][(target, argument, tuple(sorted(groups)), hateful)][2] += 1

    unmatched_gold_dict, unmatched_pred_dict = defaultdict(list), defaultdict(list)

    # 遍历每个数据项
    for item in gold_data:
        if item['id'] not in pred_data_dict:
            continue
        gold_quads = extract_gold_quads(item)
        if len(gold_quads) == 0:
            continue
        pred_quads = pred_data_dict.get(item['id'], [])

        matched_gold = [False] * len(gold_quads)
        matched_pred = [False] * len(pred_quads)

        # 遍历预测四元组
        for j, pred_quad in enumerate(pred_quads):
            if matched_pred[j]:
                continue

            pred_target, pred_argument, pred_groups, pred_hateful = pred_quad
            best_match_score = -1
            best_match_idx = -1

            # 寻找最佳匹配的真实四元组（基于相似度匹配）
            for i, gold_quad in enumerate(gold_quads):
                if matched_gold[i]:
                    continue

                gold_target, gold_argument, gold_groups, gold_hateful = gold_quad
                match_score = 0

                # 计算相似度匹配分数
                if SequenceMatcher(None, pred_target, gold_target).ratio() >= 0.5:
                    match_score += 1
                if SequenceMatcher(None, pred_argument, gold_argument).ratio() >= 0.5:
                    match_score += 1
                if pred_hateful == gold_hateful:
                    match_score += 1
                for g in pred_groups:
                    if g in gold_groups:
                        match_score += 1

                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_idx = i

            # 如果找到匹配，更新计数器
            if best_match_idx >= 0:
                matched_gold[best_match_idx] = True
                matched_pred[j] = True
                gold_target, gold_argument, gold_groups, gold_hateful = gold_quads[best_match_idx]

                # 计算匹配状态（用于soft匹配）
                target_matched = SequenceMatcher(None, pred_target, gold_target).ratio() >= 0.5
                argument_matched = SequenceMatcher(None, pred_argument, gold_argument).ratio() >= 0.5
                hateful_matched = pred_hateful == gold_hateful
                group_tuple_matched = tuple(sorted(pred_groups)) == tuple(sorted(gold_groups))

                # 更新严格匹配计数器
                update_counts(counts_strict, 'target', pred_target, gold_target, matched=pred_target == gold_target)
                update_counts(counts_strict, 'argument', pred_argument, gold_argument,
                              matched=pred_argument == gold_argument)
                update_counts(counts_strict, 'hateful', pred_hateful, gold_hateful, matched=hateful_matched)
                for g in pred_groups:
                    update_counts(counts_strict, 'group', g, gold_groups, is_group=True, matched=g in gold_groups)
                update_counts(counts_strict, 'target_argument',
                              (pred_target, pred_argument), (gold_target, gold_argument),
                              matched=(pred_target == gold_target and pred_argument == gold_argument))
                update_counts(counts_strict, 'target_argument_group',
                              (pred_target, pred_argument, tuple(sorted(pred_groups))),
                              (gold_target, gold_argument, tuple(sorted(gold_groups))),
                              matched=(pred_target == gold_target and pred_argument == gold_argument and
                                       tuple(sorted(pred_groups)) == tuple(sorted(gold_groups))))
                update_counts(counts_strict, 'target_argument_hateful',
                              (pred_target, pred_argument, pred_hateful),
                              (gold_target, gold_argument, gold_hateful),
                              matched=(
                                      pred_target == gold_target and pred_argument == gold_argument and hateful_matched))
                update_counts(counts_strict, 'target_argument_group_hateful',
                              (pred_target, pred_argument, tuple(sorted(pred_groups)), pred_hateful),
                              (gold_target, gold_argument, tuple(sorted(gold_groups)), gold_hateful),
                              matched=(pred_target == gold_target and pred_argument == gold_argument and
                                       tuple(sorted(pred_groups)) == tuple(sorted(gold_groups)) and hateful_matched))

                # 更新相似度匹配计数器
                update_counts(counts_soft, 'target', pred_target, gold_target, matched=target_matched)
                update_counts(counts_soft, 'argument', pred_argument, gold_argument, matched=argument_matched)
                update_counts(counts_soft, 'hateful', pred_hateful, gold_hateful, matched=hateful_matched)
                for g in pred_groups:
                    update_counts(counts_soft, 'group', g, gold_groups, is_group=True, matched=g in gold_groups)
                update_counts(counts_soft, 'target_argument',
                              (pred_target, pred_argument), (gold_target, gold_argument),
                              matched=(target_matched and argument_matched))
                update_counts(counts_soft, 'target_argument_group',
                              (pred_target, pred_argument, tuple(sorted(pred_groups))),
                              (gold_target, gold_argument, tuple(sorted(gold_groups))),
                              matched=(target_matched and argument_matched and group_tuple_matched))
                update_counts(counts_soft, 'target_argument_hateful',
                              (pred_target, pred_argument, pred_hateful),
                              (gold_target, gold_argument, gold_hateful),
                              matched=(target_matched and argument_matched and hateful_matched))
                update_counts(counts_soft, 'target_argument_group_hateful',
                              (pred_target, pred_argument, tuple(sorted(pred_groups)), pred_hateful),
                              (gold_target, gold_argument, tuple(sorted(gold_groups)), gold_hateful),
                              matched=(target_matched and argument_matched and group_tuple_matched and hateful_matched))

        for idx, _flag in enumerate(matched_gold):
            if not _flag:
                unmatched_gold_dict[item['id']].append(gold_quads[idx])

        for idx, _flag in enumerate(matched_pred):
            if not _flag:
                unmatched_pred_dict[item['id']].append(pred_quads[idx])

    # 计算指标函数
    def compute_metrics(counts_dict):
        metrics = {}
        for key in counts_dict:
            total_tp = 0
            total_pred = 0
            total_gold = 0

            for item, (tp, pred, gold) in counts_dict[key].items():
                total_tp += tp
                total_pred += pred
                total_gold += gold

            precision = total_tp / total_pred if total_pred > 0 else 0
            recall = total_tp / total_gold if total_gold > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics[key] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': total_gold
            }
        return metrics

    # 返回两套指标
    return {
        'strict': compute_metrics(counts_strict),
        'soft': compute_metrics(counts_soft)
    }, unmatched_gold_dict, unmatched_pred_dict


def extract_gold_quads(item):
    """
    从数据项中提取真实四元组

    参数:
        item: 数据项字典

    返回:
        quads: 四元组列表 [(target, argument, groups, hateful)]
    """

    quads = []
    keys = list(item.keys())

    # 遍历所有键，查找四元组
    i = 0
    while i < len(keys):
        if 'Target' in keys[i]:
            # 提取四元组元素
            target = item[keys[i]]
            argument = item[keys[i + 1]]

            if target == 'NULL' or len(target) == 0 or len(argument) == 0 or argument == 'NULL':
                i += 4  # 跳过已处理的键
                continue

            # 处理group字段（可能是字符串或列表）
            group_str = item[keys[i + 2]]
            if isinstance(group_str, list):
                groups = group_str
            else:
                # 处理多标签情况，如"Region, Racism"
                groups = [g.strip() for g in group_str.split(',')]

            hateful = item[keys[i + 3]]

            if groups == ['non-hate'] and hateful == 'hate':
                raise Exception('hateful error')

            if hateful == 'non-hate' and groups != ['non-hate']:
                groups = ['non-hate']

            if hateful not in ['non-hate', 'hate']:
                raise Exception('hate text error')

            if any([group not in 'LGBTQ Racism Region Sexism others non-hate'.split() for group in
                    groups]):
                raise Exception('group text error')

            if target != 'NULL' and target not in item['sentence']:
                raise Exception('target span is error')

            if argument != 'NULL' and argument not in item['sentence']:
                raise Exception('argument span is error')

            quads.append((target, argument, tuple(sorted(groups)), hateful))
            i += 4  # 跳过已处理的键
        else:
            i += 1

    return quads


def update_counts(counts_dict, metric_type, pred_value, gold_value, is_group=False):
    """
    更新指标计数器

    参数:
        counts_dict: 计数器字典
        metric_type: 指标类型
        pred_value: 预测值
        gold_value: 真实值
        is_group: 是否为group指标（多标签处理）
    """
    # 更新预测计数
    counts_dict[metric_type][pred_value][1] += 1

    # 检查匹配并更新TP
    if is_group:
        # group指标：多标签匹配
        if pred_value in gold_value:
            counts_dict[metric_type][pred_value][0] += 1
    else:
        # 其他指标：精确匹配
        if pred_value == gold_value:
            counts_dict[metric_type][pred_value][0] += 1


def format_metrics(metrics):
    """
    将评价指标格式化为字符串，支持新的strict/soft结构

    参数:
        metrics: 评价指标字典，包含'strict'和'soft'两个子字典

    返回:
        output: 格式化后的字符串
    """
    # 创建输出缓冲区
    output_lines = []

    # 添加表头
    header = "{:<30} {:<10} {:<10} {:<10}"
    row_format = "{:<30} {:<10.2f} {:<10.2f} {:<10.2f}"
    separator = "-" * 60

    # 处理严格匹配指标
    output_lines.append("Strict Matching Metrics:")
    output_lines.append(header.format("Metric", "Precision", "Recall", "F1"))
    output_lines.append(separator)

    for metric, values in metrics['strict'].items():
        output_lines.append(row_format.format(
            metric.capitalize().replace('_', ' '),
            values['precision'] * 100,  # 转换为百分比
            values['recall'] * 100,
            values['f1'] * 100
        ))

    # 添加空行分隔符
    output_lines.append("")

    # 处理相似度匹配指标
    output_lines.append("Soft Matching Metrics (similarity>=0.5):")
    output_lines.append(header.format("Metric", "Precision", "Recall", "F1"))
    output_lines.append(separator)

    for metric, values in metrics['soft'].items():
        output_lines.append(row_format.format(
            metric.capitalize().replace('_', ' '),
            values['precision'] * 100,
            values['recall'] * 100,
            values['f1'] * 100
        ))

    # 将结果连接为字符串
    return "\n".join(output_lines)


import torch


def determine_labels_tensor(pred_history, thresholds, last_decisions_threshold=0.3):
    """
    真正的三支决策推理逻辑
    pred_history: [Layers, Batch, Seq, Seq, Class] -> 每一层的 Logits
    thresholds:   [Layers, Class, 2] -> 每一层学习到的 (a, b)
    """
    S, B, L, _, N = pred_history.shape

    # 初始化
    decided_mask = torch.zeros((B, L, L, N), dtype=torch.bool, device=pred_history.device)
    final_labels = torch.zeros((B, L, L, N), dtype=torch.float32, device=pred_history.device)

    # 1. 遍历前 S-1 层 (Iterative Refinement)
    for s in range(S - 1):
        # 当前层概率
        prob = torch.sigmoid(pred_history[s])  # [B, L, L, N]

        # 当前层阈值 [N, 2] -> [1, 1, 1, N]
        current_thresh = thresholds[s]
        a = current_thresh[:, 0].view(1, 1, 1, N)
        b = current_thresh[:, 1].view(1, 1, 1, N)

        # 找出尚未决策的样本
        active_mask = ~decided_mask

        # 三支决策逻辑：
        # Reject: prob < a
        # Accept: prob > b
        # Wait:   a <= prob <= b (不做处理，留给下一层)

        reject_mask = active_mask & (prob < a)
        accept_mask = active_mask & (prob > b)

        # 更新已决策状态
        new_decisions = reject_mask | accept_mask
        decided_mask = decided_mask | new_decisions

        # 记录结果 (Reject默认为0, Accept设为1)
        final_labels[accept_mask] = 1.0

    # 2. 最后一层 (强制决策 / Final Decision)
    # 对于在深层依然 undecided 的样本（即难样本），使用更敏感的阈值
    last_prob = torch.sigmoid(pred_history[-1])
    undecided_mask = ~decided_mask

    # 这里的 last_decisions_threshold 设低一点 (如0.3)，专门为了召回难样本
    last_decisions = torch.where(
        last_prob[undecided_mask] > last_decisions_threshold,
        torch.tensor(1.0, device=pred_history.device),
        torch.tensor(0.0, device=pred_history.device)
    )

    final_labels[undecided_mask] = last_decisions

    return final_labels


class DiceLoss(nn.Module):
    """
    Dice Loss 专门用于解决极度不平衡的分割/提取任务。
    它直接优化 F1 Score 的近似值。
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [..., C]
        # targets: [..., C]
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (probs * targets).sum()

        # Dice coef = (2 * TP) / (2 * TP + FP + FN)
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice


def compute_enhanced_three_way_loss(logit_history, p_continue_history, tags, alpha=0.75, gamma=2):
    """
    混合 Loss：Focal + Dice + 动态加权
    """
    S, B, L, _, N = logit_history.shape
    targets = tags.unsqueeze(0).expand(S, B, L, L, N)

    # -------------------------------------------------------------------------
    # 策略 1: 动态正样本加权 (Dynamic Pos_Weight)
    # -------------------------------------------------------------------------
    # 计算当前 batch 的正负样本比例
    num_pos = targets.sum()
    num_neg = targets.numel() - num_pos

    # 自动计算平衡权重
    if num_pos > 0:
        # 限制最大权重为 20，防止梯度爆炸
        dynamic_weight = torch.clamp(num_neg / num_pos, min=1.0, max=20.0)
    else:
        dynamic_weight = torch.tensor(1.0).to(logit_history.device)

    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=dynamic_weight, reduction='none')
    per_layer_bce_loss = bce_criterion(logit_history, targets)

    # -------------------------------------------------------------------------
    # 策略 2: Dice Loss (保持不变)
    # -------------------------------------------------------------------------
    dice_criterion = DiceLoss()
    last_layer_logits = logit_history[-1].reshape(-1, N)
    last_layer_targets = targets[-1].reshape(-1, N)
    dice_loss = dice_criterion(last_layer_logits, last_layer_targets)

    # -------------------------------------------------------------------------
    # 策略 3: 三支决策加权 (保持不变)
    # -------------------------------------------------------------------------
    weights = (1 - p_continue_history) + 0.1
    # 确保维度匹配，直接相乘
    weighted_cls_loss = (per_layer_bce_loss * weights).mean()

    # 约束项
    last_layer_p = p_continue_history[-1]
    termination_loss = torch.mean(last_layer_p ** 2)
    sparsity_loss = torch.mean(p_continue_history)

    # -------------------------------------------------------------------------
    # 总 Loss
    # -------------------------------------------------------------------------
    total_loss = weighted_cls_loss + 1.0 * dice_loss + 0.1 * termination_loss + 0.01 * sparsity_loss

    return total_loss


def data_processor(args, tokenizer, stanza_model, data_list):
    # 清理数据

    for data_item in tqdm(data_list):
        del data_item['deprel']
        del data_item['head']
        del data_item['postag']
        data_item['sentence'] = clean_blank(data_item['sentence'])

        tokenized_data = TokenizedData(args.model_path.name, tokenizer, data_item['sentence'])
        data_item['tokenized_data'] = tokenized_data

        data_item_key_list = list(data_item.keys())
        for key_idx, key in enumerate(data_item.keys()):
            if 'Target' in key:

                target_text = clean_blank(data_item[key])
                if target_text != 'NULL' and len(target_text) != 0:
                    if target_text not in data_item['sentence']:
                        raise Exception('target子串匹配错误')
                    target_text, _ = tokenized_data.sub_str2token_range(target_text)
                data_item[key] = target_text

                argument_text = clean_blank(data_item[data_item_key_list[key_idx + 1]])
                if argument_text != 'NULL' and len(argument_text) != 0:
                    if argument_text not in data_item['sentence']:
                        raise Exception('argument子串匹配错误')
                    argument_text, _ = tokenized_data.sub_str2token_range(argument_text)
                data_item[data_item_key_list[key_idx + 1]] = argument_text

    data_length = len(data_list)

    i, batch = 0, 200
    # 创建进度条
    with tqdm(total=data_length, desc="Processing data") as pbar:

        while i < data_length:
            j = min(i + batch, data_length)

            tokens_list = [data_item['tokenized_data'].token_without_special_list for data_item in data_list[i:j]]

            doc = stanza_model(tokens_list)
            for sentence_idx, sentence in enumerate(doc.sentences):
                data_list[i + sentence_idx]['postag'] = [word.pos for word in sentence.words]
                data_list[i + sentence_idx]['head'] = [word.head for word in sentence.words]
                data_list[i + sentence_idx]['deprel'] = [word.deprel for word in sentence.words]

            i += batch
            pbar.update(batch)  # 更新进度条

    return data_list


# 测试

# 使用示例
if __name__ == "__main__":
    # 示例真实数据
    gold_data = [
        {
            "id": 6343,
            "sentence": "说河南人偷井盖的明明是北京人，我一个南方人都知道，东北人会不知道。东北人就会舔北京，然后拉着整个北方对抗南方，搞得像分裂国家一样。真的没治了。",
            "platform": "zhihu",
            "topic": "region",
            "sen_hate": "1",
            "Q1 Target": "北京人",
            "Q1 Argument": "说河南人偷井盖的",
            "Q1 Group": "Region",
            "Q1 hateful": "hate",
            "Q2 Target": "东北人",
            "Q2 Argument": "就会舔北京",
            "Q2 Group": "Region",
            "Q2 hateful": "hate"
        },
        {
            "id": 6578,
            "sentence": "这暴躁老哥说的好呀要不是疫情我也去东南亚快乐一下，抖音上那群老板也去霍霍非洲世界小姐们了",
            "platform": "zhihu",
            "topic": "race",
            "sen_hate": "1",
            "Q1 Target": "东南亚",
            "Q1 Argument": "快乐一下",
            "Q1 Group": "Region",
            "Q1 hateful": "hate",
            "Q2 Target": "非洲世界小姐们",
            "Q2 Argument": "霍霍",
            "Q2 Group": "Region, Racism",
            "Q2 hateful": "hate"
        }
    ]

    # 示例预测数据（字典格式）
    pred_data_dict = {
        6343: [
            ("北京人", "说河南人偷井盖的", ["Region"], "hate"),
            ("东北人", "就会舔北京", ["Region"], "hate")
        ],
        6578: [
            ("东南亚", "快乐一下", ["Region"], "hate"),
            ("非洲世界小姐们", "霍霍", ["Region"], "hate")
        ]
    }

    # 计算指标
    metrics = calculate_metrics(gold_data, pred_data_dict)

    # 格式化结果
    formatted_output = format_metrics(metrics)

    # 打印结果
    print(formatted_output)