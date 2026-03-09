import gc
import math
import stanza
import torch
import numpy as np
from collections import OrderedDict, defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from common_utils import *

sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}

# label = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
# label2id = {'N': 0, 'B-A': 1, 'I-A': 2, 'A': 3, 'B-O': 4, 'I-O': 5, 'O': 6, 'negative': 7, 'neutral': 8, 'positive': 9}
# {'LGBTQ': '性少数群体', 'Racism': '种族主义', 'Region': '区域', 'Sexism': '性别歧视', 'non-hate': '无类别','others': '其他'}
label = "B-T I-T B-A I-A LGBTQ Racism Region Sexism others hate non-hate".split()


label2id, id2label = OrderedDict(), OrderedDict()
for i, v in enumerate(label):
    label2id[v] = i
    id2label[i] = v


def get_spans(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_evaluate_spans(tags, length, token_range):
    '''for BIO tag'''
    spans = []
    start = -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l] == -1:
            continue
        elif tags[l] == 1:
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[l] == 0:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_best_span_pair(span_list1, span_list2):
    _span_idx1, _span_idx2 = [s for s, e in span_list1], [s for s, e in span_list2]

    min_abs, i, j = 100, 0, 0
    for _i, s_1 in enumerate(_span_idx1):
        for _j, s_2 in enumerate(_span_idx2):
            _abs = abs(s_1 - s_2)
            if _abs < min_abs:
                min_abs = _abs
                i, j = _i, _j

    return i, j


class Instance(object):
    def __init__(self, data_item, post_vocab, deprel_vocab, postag_vocab, synpost_vocab):

        self.tokenized_data = data_item['tokenized_data']

        self.id = data_item['id']
        self.tokens = self.tokenized_data.token_without_special_list

        postag = data_item['postag']
        head = data_item['head']
        deprel = data_item['deprel']
        self.sen_length = len(self.tokens)

        self.bert_tokens = self.tokenized_data.inputs.input_ids


        self.length = len(self.bert_tokens)
        # self.bert_tokens_padding = torch.zeros(args.max_sequence_len)
        self.tags = torch.zeros(self.length, self.length, len(label))
        # self.mask = torch.zeros(args.max_sequence_len)


        # for i in range(self.length):
        #     self.bert_tokens_padding[i] = self.bert_tokens[i]
        # self.mask[:self.length] = 1




        key_list = list(data_item.keys())
        for key_idx, key in enumerate(key_list):
            # 依次处理四元组
            if 'Target' not in key:
                continue

            target_txt = data_item[key]
            argument_txt = data_item[key_list[key_idx + 1]]
            group_txt = data_item[key_list[key_idx + 2]].strip()
            hateful_txt = data_item[key_list[key_idx + 3]].strip()

            if group_txt == 'non-hate' and hateful_txt == 'hate':
                raise Exception('hateful error')

            if hateful_txt == 'non-hate' and group_txt != 'non-hate':
                group_txt = 'non-hate'

            if hateful_txt not in ['non-hate', 'hate']:
                raise Exception('hate text error')

            if any([group not in 'LGBTQ Racism Region Sexism others non-hate'.split() for group in group_txt.split(', ')]):
                raise Exception('group text error')

            group_id_list = []
            for group in group_txt.split(', '):
                group_id_list.append(label2id[group])

            hateful_id = label2id[hateful_txt]

            target_span_list, argument_span_list = [], []

            if target_txt != 'NULL' and len(target_txt) != 0:
                _, target_span_list = self.tokenized_data.sub_str2token_range(target_txt)
            if argument_txt != 'NULL' and len(argument_txt) != 0:
              _, argument_span_list = self.tokenized_data.sub_str2token_range(argument_txt)

            if target_txt != 'NULL' and len(target_span_list) == 0 and len(target_txt) != 0:
                raise Exception('target span is error')

            if argument_txt != 'NULL' and len(argument_span_list) == 0 and len(argument_txt) != 0:
                raise Exception('argument span is error')

            if target_span_list and argument_span_list:
                i, j = get_best_span_pair(target_span_list, argument_span_list)
                target_span_list = [target_span_list[i]]
                argument_span_list = [argument_span_list[j]]


        #     填充对角线
        #     label = "B-T I-T B-A I-A LGBTQ Racism Region Sexism others hate non-hate".split()
            B_T_idx, I_T_idx, B_A_idx, I_A_idx = label2id['B-T'], label2id['I-T'], label2id['B-A'], label2id['I-A'],
            # 填充target标记
            for s, e in target_span_list:
                for i in range(s, e):
                    if i == s:
                        self.tags[i, i, B_T_idx] = 1
                    else:
                        self.tags[i, i, I_T_idx] = 1
            # 填充argument标记
            for s, e in argument_span_list:
                for i in range(s, e):
                    if i == s:
                        self.tags[i, i, B_A_idx] = 1
                    else:
                        self.tags[i, i, I_A_idx] = 1

        #     填充group和hate标记
        #     LGBTQ Racism Region Sexism others hate non-hate
            for t_s, t_e in target_span_list:
                for a_s, a_e in argument_span_list:
                    self.tags[t_s: t_e, a_s: a_e, hateful_id] = 1
                    for group_idx in group_id_list:
                        self.tags[t_s: t_e, a_s: a_e, group_idx] = 1

        '''1. generate position index of the word pair'''
        self.word_pair_position = torch.zeros(self.length, self.length).long()
        for i in range(self.sen_length):
            for j in range(self.sen_length):
                self.word_pair_position[i + 1][j + 1] = post_vocab.stoi.get(abs(i - j), post_vocab.unk_index)

        """2. generate deprel index of the word pair"""
        self.word_pair_deprel = torch.zeros(self.length, self.length).long()
        for i in range(self.sen_length):
            j = head[i] - 1
            self.word_pair_deprel[i + 1][j + 1] = deprel_vocab.stoi.get(deprel[i])
            self.word_pair_deprel[j + 1][i + 1] = deprel_vocab.stoi.get(deprel[i])
            self.word_pair_deprel[i + 1][i + 1] = deprel_vocab.stoi.get('self')

        """3. generate POS tag index of the word pair"""
        self.word_pair_pos = torch.zeros(self.length, self.length).long()
        for i in range(self.sen_length):
            for j in range(self.sen_length):
                self.word_pair_pos[i + 1][j + 1] = postag_vocab.stoi.get(tuple([postag[i], postag[j]]))


        """4. generate synpost index of the word pair"""
        self.word_pair_synpost = torch.zeros(self.length, self.length).long()
        tmp = [[0]*len(self.tokens) for _ in range(len(self.tokens))]
        for i in range(len(self.tokens)):
            j = head[i]
            if j == 0:
                continue
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        tmp_dict = defaultdict(list)
        for i in range(len(self.tokens)):
            for j in range(len(self.tokens)):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        word_level_degree = [[4]*len(self.tokens) for _ in range(len(self.tokens))]

        for i in range(len(self.tokens)):
            node_set = set()
            word_level_degree[i][i] = 0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    word_level_degree[i][j] = 1
                    node_set.add(j)
                for k in tmp_dict[j]:
                    if k not in node_set:
                        word_level_degree[i][k] = 2
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                word_level_degree[i][g] = 3
                                node_set.add(g)


        for i in range(self.sen_length):
            for j in range(self.sen_length):
                self.word_pair_synpost[i + 1][j + 1] = synpost_vocab.stoi.get(word_level_degree[i][j],
                                                                          synpost_vocab.unk_index)


def load_data_instances(data_list, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args):
    instances = list()

    for data_item in tqdm(data_list):
        instances.append(Instance(data_item, post_vocab, deprel_vocab, postag_vocab, synpost_vocab))
    gc.collect()
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def get_batch(self, index):
        # 计算本次 batch 的范围
        start_idx = index * self.args.batch_size
        end_idx = min((index + 1) * self.args.batch_size, len(self.instances))
        curr_batch_size = end_idx - start_idx

        # 预先筛选并计算最大长度
        batch_instances = []
        for i in range(start_idx, end_idx):
            inst = self.instances[i]
            if inst.length <= self.args.max_sequence_len:
                batch_instances.append(inst)

        if not batch_instances:  # 防止空 batch
            return None

        length_list = [inst.length for inst in batch_instances]
        max_length = max(length_list)
        real_batch_size = len(batch_instances)

        # --- 预分配 Tensor (Pre-allocation) ---
        # 直接在 CPU 上开辟好所需的整块内存，避免循环中的内存碎片
        bert_tokens = torch.full((real_batch_size, max_length),
                                 self.tokenizer.pad_token_id, dtype=torch.long)
        masks = torch.zeros((real_batch_size, max_length), dtype=torch.float)
        tags = torch.zeros((real_batch_size, max_length, max_length, len(label)), dtype=torch.float)

        word_pair_position = torch.zeros((real_batch_size, max_length, max_length), dtype=torch.long)
        word_pair_deprel = torch.zeros((real_batch_size, max_length, max_length), dtype=torch.long)
        word_pair_pos = torch.zeros((real_batch_size, max_length, max_length), dtype=torch.long)
        word_pair_synpost = torch.zeros((real_batch_size, max_length, max_length), dtype=torch.long)

        lengths = torch.tensor(length_list, dtype=torch.long)

        # 列表用于存储非 Tensor 数据
        id_list = []
        tokens_list = []
        sens_lens = []
        tokenized_data_list = []

        # --- 填充数据 ---
        for i, inst in enumerate(batch_instances):
            l = inst.length
            sl = inst.sen_length

            # 基础信息
            id_list.append(inst.id)
            tokens_list.append(inst.tokens)
            sens_lens.append(sl)
            tokenized_data_list.append(inst.tokenized_data)

            # Tensor 切片赋值 (Vectorized-like filling)
            # 这里的切片赋值比 torch.cat/stack 快得多
            bert_tokens[i, :l] = torch.tensor(inst.bert_tokens, dtype=torch.long)
            masks[i, :l] = 1.0
            tags[i, :l, :l, :] = inst.tags

            word_pair_position[i, :l, :l] = inst.word_pair_position
            word_pair_deprel[i, :l, :l] = inst.word_pair_deprel
            word_pair_pos[i, :l, :l] = inst.word_pair_pos
            word_pair_synpost[i, :l, :l] = inst.word_pair_synpost

        # 统一移动到 GPU
        device = self.args.device
        return (id_list, tokens_list,
                bert_tokens.to(device),
                lengths.to(device),
                masks.to(device),
                sens_lens,
                tags.to(device),
                word_pair_position.to(device),
                word_pair_deprel.to(device),
                word_pair_pos.to(device),
                word_pair_synpost.to(device),
                tokenized_data_list)