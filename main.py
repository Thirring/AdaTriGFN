#coding utf-8

import json, os
import pickle
import random
import argparse
from datetime import datetime

import stanza
import torch
import torch.nn.functional as F
from tqdm import trange

from data import load_data_instances, DataIterator, label2id, id2label
from model import HSD, FGM
import utils
import torch.nn as nn

import numpy as np

from prepare_vocab import VocabHelp, get_vocab
# from transformers import AdamW
from torch.optim import AdamW

from loguru import logger

from common_utils import *
from utils import data_processor

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 0=第一块GPU, 1=第二块GPU


from transformers import get_linear_schedule_with_warmup


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # 注意：这里不需要 self.bce，直接在 forward 用 functional 计算更灵活

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)

        # 修正：区分正负样本权重
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_bert_optimizer(model, args):
    """
    简化版优化器配置，为thresholds参数单独配置学习率
    """
    # 参数分组
    optimizer_grouped_parameters = []

    # 获取所有参数名称
    param_dict = {n: p for n, p in model.named_parameters()}

    bert_params = [p for n, p in param_dict.items() if 'bert' in n]

    optimizer_grouped_parameters.append({
        "params": bert_params,
        "weight_decay": args.weight_decay,
        "lr": args.bert_lr
    })

    non_bert_params = [p for n, p in param_dict.items() if 'bert' not in n]

    optimizer_grouped_parameters.append({
        "params": non_bert_params,
        "weight_decay": 0.0,
        "lr": args.learning_rate
    })

    # 创建优化器
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)

    return optimizer


def train(args):

    # load dataset
    # train_data_list = json.load(open(Path(args.dataset_dir) / 'train.json'), encoding='utf-8')
    with open(Path(args.dataset_dir) / 'train.json', 'r', encoding='utf-8') as f:
        train_data_list = json.load(f)
    random.shuffle(train_data_list)
    # test_data_list = json.load(open(Path(args.dataset_dir) / 'test.json'), encoding='utf-8')
    with open(Path(args.dataset_dir) / 'test.json', 'r', encoding='utf-8') as f:
        test_data_list = json.load(f)


    train_data_pkl_path = Path(args.pkl_dir) / 'train.pkl'
    test_data_pkl_path = Path(args.pkl_dir) / 'test.pkl'

    if not all([train_data_pkl_path.exists(), test_data_pkl_path.exists()]):

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        stanza_model = stanza.Pipeline(lang = "zh-hans",
                              download_method=None,
                              tokenize_pretokenized=True,
                              use_gpu=False,
                              )

        if not train_data_pkl_path.exists():
            train_data_list = data_processor(args, tokenizer, stanza_model, train_data_list)

            # 将对象打包存储到文件
            with open(train_data_pkl_path, "wb") as f:  # 注意使用二进制写入模式
                pickle.dump(train_data_list, f)

        if not test_data_pkl_path.exists():
            test_data_list = data_processor(args, tokenizer, stanza_model, test_data_list)

            # 将对象打包存储到文件
            with open(test_data_pkl_path, "wb") as f:  # 注意使用二进制写入模式
                pickle.dump(test_data_list, f)

        del stanza_model
    else:
        # 从文件加载对象
        with open(train_data_pkl_path, "rb") as f:  # 二进制读取模式
            train_data_list = pickle.load(f)
        # 从文件加载对象
        with open(test_data_pkl_path, "rb") as f:  # 二进制读取模式
            test_data_list = pickle.load(f)

    post_vocab_path = Path(args.pkl_dir) / 'vocab_post.vocab'
    deprel_vocab_path = Path(args.pkl_dir) / 'vocab_deprel.vocab'
    postag_vocab_path = Path(args.pkl_dir) / 'vocab_postag.vocab'
    synpost_vocab_path = Path(args.pkl_dir) / 'vocab_synpost.vocab'

    if not all([post_vocab_path.exists(), deprel_vocab_path.exists(), postag_vocab_path.exists(), synpost_vocab_path.exists()]):
        post_vocab, deprel_vocab, postag_vocab, synpost_vocab = get_vocab(train_data_list, test_data_list, Path(args.pkl_dir))
    else:
        post_vocab = VocabHelp.load_vocab(post_vocab_path)
        deprel_vocab = VocabHelp.load_vocab(deprel_vocab_path)
        postag_vocab = VocabHelp.load_vocab(postag_vocab_path)
        synpost_vocab = VocabHelp.load_vocab(synpost_vocab_path)

    args.post_size = len(post_vocab)
    args.deprel_size = len(deprel_vocab)
    args.postag_size = len(postag_vocab)
    args.synpost_size = len(synpost_vocab)

    instances_train_path = Path(args.pkl_dir) / f'{Path(args.model_path).name}_instances_train.pkl'
    instances_test_path = Path(args.pkl_dir) / f'{Path(args.model_path).name}_instances_test.pkl'

    if not instances_train_path.exists():
        instances_train = load_data_instances(train_data_list, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args)
        # 将对象打包存储到文件
        with open(instances_train_path, "wb") as f:  # 注意使用二进制写入模式
            pickle.dump(instances_train, f)
    else:
        # 从文件加载对象
        with open(instances_train_path, "rb") as f:  # 二进制读取模式
            instances_train = pickle.load(f)


    if not instances_test_path.exists():
        instances_test = load_data_instances(test_data_list, post_vocab, deprel_vocab, postag_vocab, synpost_vocab,
                                            args)
        # 将对象打包存储到文件
        with open(instances_test_path, "wb") as f:  # 注意使用二进制写入模式
            pickle.dump(instances_test, f)
    else:
        # 从文件加载对象
        with open(instances_test_path, "rb") as f:  # 二进制读取模式
            instances_test = pickle.load(f)


    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    testset = DataIterator(instances_test, args)

    # # 在模型初始化前添加
    # torch.backends.cuda.enable_flash_sdp(False)
    # torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cuda.enable_math_sdp(True)  # 强制使用数学实现（确定性）

    # [建议修改] 稍微提高 BERT 的学习率，8e-6 可能太低导致收敛极慢
    # 可以在 config.yaml 改，或者在这里覆盖
    # [优化] 提高学习率。对于 F1=0 的情况，先用大一点的学习率打破局部最优
    # if args.bert_lr < 2e-5:
    #     args.bert_lr = 3e-5

    model = HSD(args).to(args.device)
    optimizer = get_bert_optimizer(model, args)

    # [改进点1] 添加学习率调度器 (Scheduler)
    # 计算总步数 = epoch * steps_per_epoch
    total_steps = trainset.batch_count * args.epochs
    # Warmup 10% 的步数
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )



    # [改进点3] 初始化 FGM 对抗训练
    fgm = FGM(model)

    best_joint_f1 = 0
    best_joint_epoch = 0
    f1_list = []

    # [+] 初始化列表用于存储每一轮的阈值
    # Shape: [Epochs, Layers, Classes, 2]
    threshold_evolution = []

    logger.info("Using Hybrid Loss (BCE + Dice) to fix Zero F1 Problem...")

    # [改进点2] 使用 Focal Loss 替代 BCE
    # alpha=0.75 意味着更关注正样本 (target=1)，gamma=2 关注困难样本
    # criterion = MultiLabelFocalLoss(alpha=0.75, gamma=2).to(args.device)

    for i in range(args.epochs):
        logger.info('Epoch:{}'.format(i))
        model.train()
        set_seed(args.seed + i)  # 稍微改变每轮的随机性

        for j in trange(trainset.batch_count, desc=f"Epoch {i}"):
            # 获取数据
            id_list, tokens_list, tokens, lengths, masks, _, tags, word_pair_position, \
                word_pair_deprel, word_pair_pos, word_pair_synpost, tokenized_data_list = trainset.get_batch(j)

            tags_flatten = tags.reshape([-1, tags.shape[3]])

            # Forward
            weight_prob_list, logit_history, p_continue_history = model(
                tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost
            )

            # 解包：注意这里只有 biaffine_pred 是预测值，其他都是 Embedding 输入
            biaffine_pred, _, _, _, _ = weight_prob_list
            # -----------------------------------------------------------------
            # 1. 主任务 Loss (三支决策 + Dice + Focal)
            # -----------------------------------------------------------------
            l_main = utils.compute_enhanced_three_way_loss(logit_history, p_continue_history, tags)

            # -----------------------------------------------------------------
            # 2. [修复后] 辅助任务 Loss (只监督 Biaffine 层)
            # -----------------------------------------------------------------
            # Biaffine 层直接从 BERT 提取特征预测四元组，对其监督有助于稳定训练
            # 使用与主任务类似的加权 BCE，但不需要太复杂

            # 动态计算正样本权重 (与 utils 中逻辑保持一致)
            num_pos = tags.sum()
            num_neg = tags.numel() - num_pos
            if num_pos > 0:
                pos_weight = torch.clamp(num_neg / num_pos, min=1.0, max=20.0)
            else:
                pos_weight = torch.tensor(1.0).to(args.device)

            aux_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # biaffine_pred: [Batch, Seq, Seq, Class_Num]
            l_biaffine = aux_criterion(biaffine_pred, tags)

            # 总 Loss：主任务 + 0.2 * Biaffine辅助
            loss = l_main + 0.2 * l_biaffine

            # Backward
            optimizer.zero_grad()
            loss.backward()


            # [改进点3] 对抗训练 (FGM)
            # 攻击 Embedding 层，生成对抗样本，再次计算 Loss 并累加梯度
            fgm.attack(epsilon=1.0, emb_name='word_embeddings')

            # 对抗样本的前向传播
            _, logit_history_adv, p_continue_history_adv = model(
                tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost
            )

            # 计算对抗样本的三支决策 Loss (辅助Loss可以省略以加速)
            loss_adv = utils.compute_enhanced_three_way_loss(logit_history_adv, p_continue_history_adv, tags)
            loss_adv.backward()  # 梯度累加

            fgm.restore()  # 恢复原始参数

            # [改进点5] 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # --- Optimizer & Scheduler Step ---
            optimizer.step()
            scheduler.step()  # 更新学习率


        # [+] 记录当前 Epoch 结束后的阈值参数
        with torch.no_grad():
            # get_thresholds 返回 [Layers, Classes, 2]
            curr_thresh = model.get_thresholds().detach().cpu().numpy()
            threshold_evolution.append(curr_thresh)



        # ----------------------
        # 3. 评估与模型保存
        # ----------------------
        metrics, _ = eval(model, testset, test_data_list, args)
        f1 = metrics['strict']['target_argument_group_hateful']['f1']

        logger.info(f'\n{utils.format_metrics(metrics)}')

        # 记录当前学习率
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Current LR: {current_lr:.2e}")

        f1_list.append(f1)

        # 简单的 Early Stopping 检查 (如果连续多轮为0)
        if len(f1_list) > 2 and sum(f1_list[-3:]) < 1e-6:
            logger.info("F1 Score 长期为 0，模型未收敛，提前结束。")
            break

        if f1 > best_joint_f1:
            best_joint_f1 = f1
            best_joint_epoch = i
            # 保存最佳模型
            save_path = args.log_dir / 'model.pt'
            # torch.save(model, save_path)
            logger.info(f"Model saved to {save_path}")


    # [+] 训练结束后，保存阈值演变数据到文件
    # np.save(config.log_dir / 'threshold_evolution.npy', np.array(threshold_evolution))
    logger.info(f"Threshold evolution data saved to {config.log_dir / 'threshold_evolution.npy'}")

    logger.info(f'best epoch: {best_joint_epoch}\t best F1: {best_joint_f1}')
    # excel_writer.save()


def eval(model, dataset, data_list, args, FLAG=False):
    model.eval()
    with torch.no_grad():
        pred_data_dict = {}

        # 获取学习到的阈值 [Layers, Class, 2]
        thresholds = model.get_thresholds()

        for i in trange(dataset.batch_count):
            id_list, tokens_list, tokens, lengths, masks, sens_lens, tags, \
                word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost, tokenized_data_list = dataset.get_batch(
                i)

            weight_prob_list, logit_history, p_continue_history = model(
                tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost
            )

            # [关键修改] 使用真正的三支决策逻辑进行解码
            # 传入完整的 logit_history 和 thresholds
            pred_tensor = utils.determine_labels_tensor(
                logit_history,
                thresholds,
                last_decisions_threshold=args.last_decisions_threshold  # 最后一层兜底阈值，保持较低以提升Recall
            )


            for batch_idx, data_id in enumerate(id_list):
                # 解码
                pred_data_dict[data_id] = utils.decode_quadruples(
                    pred_tensor[batch_idx],  # 已经是 0/1 张量
                    tokens_list[batch_idx],
                    label2id,
                    tokenized_data_list[batch_idx],
                    threshold=0.5  # 因为已经是0/1了，这里无所谓
                )

    return utils.calculate_metrics(data_list, pred_data_dict)


def test(args):
    print("Evaluation on testset:")
    # 在模型初始化前添加
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)  # 强制使用数学实现（确定性）

    model_path = args.log_dir / 'model.pt'
    model = torch.load(model_path, weights_only=False).to(args.device)
    model.eval()

    test_data_pkl_path = Path(args.pkl_dir) / 'test.pkl'

    with open(test_data_pkl_path, "rb") as f:  # 二进制读取模式
        test_data_list = pickle.load(f)

    instances_test_path = Path(args.pkl_dir) / f'{Path(args.model_path).name}_instances_test.pkl'

    with open(instances_test_path, "rb") as f:  # 二进制读取模式
        instances_test = pickle.load(f)

    testset = DataIterator(instances_test, args, 'test')


    metrics, unmatched_dict = eval(model, testset, test_data_list, args)

    logger.info(f'\n{utils.format_metrics(metrics)}')



    # 保存JSON到文件
    with open(Path(args.log_dir) / 'unmatched_dict.json', 'w', encoding='utf-8') as f:
        json.dump(unmatched_dict, f, ensure_ascii=False, indent=4)  # ensure_ascii=False确保中文正常显示


if __name__ == '__main__':
    # torch.set_printoptions(precision=None, threshold=float("inf"), edgeitems=None, linewidth=None, profile=None)

    # for model_path in [
    #     '/data/fuyiheng/model/chinese-roberta-wwm-ext-large', '/data/fuyiheng/model/bert-base-chinese',
    # '/data/fuyiheng/model/chinese-roberta-wwm-ext']:
    #     for b_l_r in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 2e-5, 1e-5]:
    #         for l_r in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:

    parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--batch_size', type=int, default=4,
    #                     help='bathc size')
    # parser.add_argument('--epochs', type=int, default=10,
    #                     help='training epoch number')

    parser.add_argument('--class_num', type=int, default=len(label2id),
                        help='label number')

    # #
    parser.add_argument('--learning_rate',  default=3e-4,type=float)
    parser.add_argument('--bert_lr',  default=2e-05,type=float)


    # parser.add_argument('--threshold_lr', default=5e-6, type=float)

    # parser.add_argument('--last_decisions_threshold', default=0.45, type=float)

    # parser.add_argument('--model_path', default='/data/fuyiheng/model/xlm-roberta-large', type=str)




    # parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    # parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")

    args = parser.parse_args()

    config = merge_arg("config.yaml", args)

    config.model_path = Path(config.model_path)

    config.log_dir = Path(config.log_dir) / datetime.now().strftime("%Y%m%d%H%M%S")
    # config.log_dir = Path(config.log_dir) / '20251101181046'

    # config.device = torch.device(f'cuda:{config.cuda_index}' if torch.cuda.is_available() else 'cpu')
    config.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    config.model_dir = config.log_dir

    config.pkl_dir = Path(config.dataset_dir) / config.model_path.name

    ensure_dir(config.pkl_dir)

    ensure_dir(config.log_dir)


    logger.add(config.log_dir / 'run.log')

    logger.info(dict_to_str(config))

    logger.info(set_seed(config.seed))




    config.class_num = len(label2id)

    train(config)
    # test(config)


    # if config.mode == 'train':
    #     train(config)
    #     test(config)
    # else:
    #     test(config)
