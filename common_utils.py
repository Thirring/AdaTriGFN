# 项目通用的工具
import os
import random


import numpy as np
import yaml
from attrdict import AttrDict
from pathlib import Path
import torch

import pandas as pd
from transformers import AutoTokenizer
from collections import OrderedDict, defaultdict
from transformers import set_seed as transformers_set_seed


class ExcelWriter:
    def __init__(self, columns, save_path, auto_id = True):
        """
        初始化Excel写入器
        :param columns: 列名列表，如 ['姓名', '年龄', '城市']
        :param save_path: Excel文件保存路径，如 'output.xlsx'
        """
        self.columns = columns
        if auto_id:
            self.columns = ['id'] + columns
        self.save_path = save_path
        self.data = []  # 用于存储所有行数据的列表
        self.row_id = 0

    def add_row_data(self, row_data):
        """
        添加一行数据
        :param row_data: 与列名对应的数据列表，长度需与列数相同
        """
        self.row_id += 1
        row_data = [self.row_id] + row_data
        if len(row_data) != len(self.columns):
            raise ValueError(f"数据长度({len(row_data)})与列数({len(self.columns)})不匹配")
        self.data.append(row_data)

    def add_row_data_list(self, row_data_list):
        for row_data in row_data_list:
            self.add_row_data(row_data)

    def save(self):
        """将数据保存为Excel文件"""
        df = pd.DataFrame(self.data, columns=self.columns)
        df.to_excel(self.save_path, index=False)
        print(f"Excel文件已保存至: {self.save_path}")


def find_all_substring_ranges(text, substring):
    """
    在文本中查找所有与给定子串匹配的位置范围

    此函数搜索文本中所有与指定子串完全匹配的连续字符序列，
    并返回每个匹配的起始和结束位置索引（包含结束位置）。

    参数:
        text (str): 要搜索的文本字符串
        substring (str): 要查找的子字符串

    返回:
        list of tuple: 包含所有匹配范围的列表，每个范围表示为 (start_index, end_index) 元组。
                       如果没有找到匹配，返回空列表 []。

    特点:
        - 区分大小写（大小写敏感）
        - 支持Unicode字符（包括中文）
        - 处理重叠匹配（如"aaaa"中查找"aa"会返回3个匹配）
        - 空子串处理：如果substring为空字符串，返回所有位置(0, -1)或根据实现处理

    实现说明:
        使用字符串的find()方法进行迭代搜索，每次从上次找到的位置后一位开始搜索，
        直到找不到更多匹配为止。
    """
    # 初始化起始搜索位置和结果列表
    start_index = 0
    ranges = []

    # 获取子串长度（用于计算结束位置）
    sub_len = len(substring)

    # 处理空子串的特殊情况
    if sub_len == 0:
        # 对于空子串，返回所有可能的位置
        # 注意：这可能不是所有场景都适用，根据需求调整
        return [(i, i - 1) for i in range(len(text) + 1)]

    # 循环搜索所有匹配
    while start_index < len(text):
        # 从当前起始位置开始查找子串
        pos = text.find(substring, start_index)

        # 如果找不到匹配，退出循环
        if pos == -1:
            break

        # 计算结束位置（子串最后一个字符的索引）
        end = pos + sub_len - 1

        # 将匹配范围添加到结果列表
        ranges.append((pos, end + 1))

        # 移动到下一个可能的起始位置
        # 注意：这里使用pos+1而不是pos+sub_len以支持重叠匹配
        start_index = pos + 1

    return ranges

def dict_to_str(dictionary, indent=0, sort_keys=False):
    """
    将字典转换为格式化的字符串

    参数:
        dictionary: 要转换的字典
        indent: 缩进级别（用于嵌套字典）
        sort_keys: 是否按键名排序 (默认False)

    返回:
        格式化后的字符串
    """
    if not dictionary:
        return "{}"

    items = []
    space = ' ' * indent
    keys = sorted(dictionary.keys()) if sort_keys else dictionary.keys()

    for key in keys:
        value = dictionary[key]
        if isinstance(value, dict):
            # 处理嵌套字典
            items.append(f"{space}{repr(key)}: {{\n{dict_to_str(value, indent + 4, sort_keys)}\n{space}}}")
        else:
            items.append(f"{space}{repr(key)}: {repr(value)}")

    # 根据缩进级别处理换行和逗号
    if indent == 0:
        return "{\n" + ",\n".join(items) + "\n}"
    else:
        return ",\n".join(items)

def merge_arg(config_path, args):
    """
    融合参与与配置，参数覆盖配置内容
    """
    config = AttrDict(yaml.load(open(Path(config_path), 'r', encoding='utf-8'), Loader=yaml.FullLoader))

    for k, v in vars(args).items():
        setattr(config, k, v)

    return config


def ensure_dir(path):
    """
    确保路径存在
    """
    if not os.path.exists(Path(path)):
        os.makedirs(path)

# def set_seed(seed):
#     if seed is not None:
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#
#     # 测试随机数生成
#     print("Python random:", random.random())
#     print("NumPy random:", np.random.rand())
#     print("PyTorch random:", torch.rand(1).item())
#
#     # 测试CUDA随机性（如果可用）
#     if torch.cuda.is_available():
#         print("CUDA random:", torch.cuda.FloatTensor(1).normal_().item())


def set_seed(seed=42):
    """设置所有随机种子以确保实验可复现性"""
    # 设置Python随机种子
    random.seed(seed)

    # 设置NumPy随机种子
    np.random.seed(seed)

    # 设置PyTorch随机种子
    torch.manual_seed(seed)

    # 设置CUDA随机种子（如果可用）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 禁用非确定性注意力优化
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)  # 强制使用数学实现

    # 强制使用确定性算法
    torch.use_deterministic_algorithms(True, warn_only=False)

    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 针对CUDA确定性配置

    # 设置transformers库的种子
    try:
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
    except ImportError:
        pass

    # 验证随机数生成 - 使用推荐的方式创建张量
    _random = [random.random(), np.random.rand(), torch.rand(1).item()]

    if torch.cuda.is_available():
        # 使用推荐的方式创建CUDA张量
        cuda_tensor = torch.tensor([1.0], dtype=torch.float32, device='cuda')
        _random.append(cuda_tensor.normal_().item())

    return _random

def verify_reproducibility(model, dataloader):
    """验证模型输出的可复现性"""
    set_seed(42)
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        output1 = model(*batch)

    set_seed(42)
    with torch.no_grad():
        batch = next(iter(dataloader))
        output2 = model(*batch)

    # 比较所有输出
    for i, (o1, o2) in enumerate(zip(output1, output2)):
        if isinstance(o1, (list, tuple)):
            for j, (t1, t2) in enumerate(zip(o1, o2)):
                assert torch.allclose(t1, t2, atol=1e-6), f"输出{i}-{j}不一致"
        else:
            assert torch.allclose(o1, o2, atol=1e-6), f"输出{i}不一致"

    print("可复现性验证通过！")

def clean_blank(_str):
    return ''.join(_str.split())

import torch


def determine_labels_tensor(pred, thresholds):
    """
    根据时间步预测结果和动态阈值确定最终标签（PyTorch张量实现）

    参数:
    pred -- 模型预测结果，形状为[S, L, L, N]的torch.Tensor
    thresholds -- 每个时间步和标签的阈值参数，形状为[S, N, 2]的torch.Tensor

    返回:
    final_labels -- 最终确定的标签，形状为[L, L, N]的torch.Tensor
    """
    S, L, _, N = pred.shape

    # 初始化决策状态和最终标签
    decided = torch.zeros((L, L, N), dtype=torch.bool, device=pred.device)
    final_labels = torch.zeros((L, L, N), dtype=torch.float32, device=pred.device)

    # 处理前S-1个时间步
    for s in range(S - 1):
        # 获取当前时间步预测和阈值
        current_pred = pred[s]
        current_thresholds = thresholds[s]

        # 提取当前时间步的阈值
        a_thresholds = current_thresholds[:, 0].view(1, 1, N)
        b_thresholds = current_thresholds[:, 1].view(1, 1, N)

        # 找出未决策的位置
        undecided_mask = ~decided

        # 在当前时间步做出决策的位置
        reject_mask = undecided_mask & (current_pred < a_thresholds)
        accept_mask = undecided_mask & (current_pred > b_thresholds)
        new_decisions = reject_mask | accept_mask

        # 更新决策状态
        decided = decided | new_decisions

        # 更新最终标签
        final_labels[accept_mask] = 1.0
        final_labels[reject_mask] = 0.0

    # 处理最后一个时间步
    last_pred = pred[-1]
    undecided_mask = ~decided

    # 在最后一个时间步，大于0.5则接受，否则拒绝
    last_decisions = torch.where(
        last_pred[undecided_mask] > 0.5,
        torch.tensor(1.0, device=pred.device),
        torch.tensor(0.0, device=pred.device)
    )

    final_labels[undecided_mask] = last_decisions

    return final_labels

def stanza_processor(stanza_model, tokens_list):
    """
    nlp = stanza.Pipeline(lang = "zh-hans",
                          download_method=None,
                          tokenize_pretokenized=True,
                          use_gpu=True
                          )
    tokens_list: [[token for token in data['content']]]

    """

    doc = stanza_model([tokens_list])

    return {
        'deprel': [word.deprel for word in doc.sentences[0].words],
        'head': [word.head for word in doc.sentences[0].words],
        'postag': [word.pos for word in doc.sentences[0].words]
    }





class TokenizedData():

    def __init__(self, model_name, tokenizer, sentence):
        self.model_name = model_name
        self.sentence = sentence
        self.inputs = tokenizer(
            sentence,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        self.token_with_special_list = self.clean_token(tokenizer.convert_ids_to_tokens(self.inputs.input_ids))

        _input = tokenizer(
            sentence,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        self.token_without_special_list = self.clean_token(tokenizer.convert_ids_to_tokens(_input.input_ids))

    def clean_token(self, token_list):
        if self.model_name in  ['xlm-roberta-large', 'multilingual-e5-large', 'xlm-roberta-large-finetuned-conll03-english']:
            special_token = '▁'

            _token_list = []
            for token in token_list:
                if token == special_token:
                    _token_list.append(' ')
                elif special_token in token:
                    if special_token in token[1:]:
                        raise Exception("特殊字符位置错误")
                    _token_list.append(token.replace(special_token, ''))
                else:
                    _token_list.append(token)

            return _token_list

        return token_list


    def sentence_idx2token_idx(self, sentence_idx):
        """
        原句索引映射分词序号，以及分词对应分词在原句的索引范围(包含特殊词)
        """
        for idx, (_s, _e) in enumerate(self.inputs.offset_mapping):
            if _s <= sentence_idx < _e:
                return idx

        raise Exception("原句序号错误")

    def sentence_range2token_range(self, s, e):
        t_s, t_e = self.sentence_idx2token_idx(s), self.sentence_idx2token_idx(e - 1) + 1
        if t_s >= t_e or t_s*t_e < 0:
            raise Exception("转换错误")
        return t_s, t_e

    def token_range2sentence_range(self, s, e):
        """
        token范围映射原句范围
        """
        str_s, str_e = self.inputs.offset_mapping[s][0], self.inputs.offset_mapping[e - 1][1]
        if str_s >= str_e or str_s*str_e < 0:
            raise Exception("转换错误")
        return str_s, str_e

    def token_range2str(self, s, e):
        """
        token范围映射原句子串
        """
        str_s, str_e = self.token_range2sentence_range(s, e)
        return self.sentence[str_s: str_e]

    def sub_str2token_range(self, sub_str):
        """
        子串在原句中的token(包含特殊字符)范围列表，为空则报错
        """

        sentence_range_list = find_all_substring_ranges(self.sentence, sub_str)

        sub_str_dict = defaultdict(list)
        for s_s, s_e in sentence_range_list:
            t_s, t_e = self.sentence_range2token_range(s_s, s_e)
            _sub_str = self.token_range2str(t_s, t_e)
            sub_str_dict[_sub_str].append((t_s, t_e))

        if not sub_str_dict:
            raise Exception("子串错误")

        if sub_str in sub_str_dict.keys():
            return sub_str, sub_str_dict[sub_str]
        else:
            max_key = max(sub_str_dict, key=lambda k: len(sub_str_dict[k]))
            return max_key, sub_str_dict[max_key]

    def get_special_token_list(self):
        if self.model_name == 'xlm-roberta-large':
            return ['<s>'], ['</s>']


if __name__ == "__main__":


    model_path = Path("/data/fuyiheng/model/xlm-roberta-large-finetuned-conll03-english")
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    for sentence in [
        "一是怕有病，万一………………二是你主动靠到高富帅我还明白，主动  靠近默，还倒贴………",
        "这是一句话，这 是另一句话。",
        "this is a sentence, and other one .",
        "this is a sentence, and  other one .",

    ]:
        tokenized_data = TokenizedData(model_path.name, tokenizer, clean_blank(sentence))

        print(tokenized_data.token_with_special_list)
        print(tokenized_data.sub_str2token_range("这 是"))


