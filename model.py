import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, BertModel


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class RefiningStrategy(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dim_e, dropout_ratio=0.5):
        super(RefiningStrategy, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        # 保持原有的 Linear 层定义，以便加载旧的权重文件 (state_dict key 不变)
        self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 3, self.dim_e)

    def forward(self, edge, node1, node2):
        # node1 和 node2 在传入前已经是广播后的 [B, L, L, Dim]
        # 但我们知道它们原本是由 node_outputs [B, L, Dim] 变换来的
        # 我们可以利用这一点优化，但为了接口兼容且不修改上层 GraphConvLayer，
        # 我们在这里通过数学拆解 Linear 层来优化计算。

        # 权重切分索引
        idx_edge = 0
        idx_edge_i = self.edge_dim
        idx_edge_j = self.edge_dim * 2
        idx_node1 = self.edge_dim * 3
        idx_node2 = self.edge_dim * 3 + self.hidden_dim
        idx_end = self.edge_dim * 3 + self.hidden_dim * 2

        # 提取权重和偏置
        weight = self.W.weight  # [dim_e, input_dim]
        bias = self.W.bias  # [dim_e]

        # 1. 计算 Edge 部分 (这是唯一原本就是 B,L,L,E 的部分)
        # [B, L, L, E] @ [dim_e, E]^T -> [B, L, L, dim_e]
        w_e = weight[:, idx_edge:idx_edge_i]
        out_edge = F.linear(edge, w_e)

        # 2. 计算 Edge_i 和 Edge_j (源自对角线)
        # 原始代码逻辑复现：
        # edge_diag = torch.diagonal(edge, offset=0, dim1=1, dim2=2).permute(0, 2, 1) # [B, L, E]
        # 优化：直接在 [B, L, E] 维度做 Linear，然后广播，避免构建 [B, L, L, E]
        edge_diag = torch.diagonal(edge, offset=0, dim1=1, dim2=2).permute(0, 2, 1).contiguous()

        w_ei = weight[:, idx_edge_i:idx_edge_j]
        w_ej = weight[:, idx_edge_j:idx_node1]

        # [B, L, dim_e]
        diag_feat_i = F.linear(edge_diag, w_ei)
        diag_feat_j = F.linear(edge_diag, w_ej)

        # 3. 计算 Node 部分
        # 注意：GraphConvLayer 中：
        # node_outputs1 = node.unsqueeze(1).expand... (变化在 dim 2, 即列) -> 实际上是 Col 特征
        # node_outputs2 = node1.permute...            (变化在 dim 1, 即行) -> 实际上是 Row 特征
        # 但是这里为了绝对保证和原代码 "cat([node1, node2])" 结果一致，我们不需要回溯到 GraphConvLayer 修改
        # 我们只需要知道 node1 和 node2 本质上是重复的，低秩的。
        #
        # 为了“完全不改变运算结果”的稳妥起见，如果不想修改上层输入，
        # 我们依然可以在这里针对 node1, node2 做分解会比较麻烦（因为传入的是展开后的）。
        #
        # **最高效且稳妥的方法**：修改 GraphConvLayer 的传参，但你要求改动小。
        # 因此，这里我们只对 edge_i 和 edge_j 做分解优化（节省了 2/5 的巨大 Tensor 显存），
        # 对 node1, node2 维持原样（或者利用 node1[..., 0, :] 还原回 [B, L, H]）。

        # 还原 node 特征以加速运算:
        # node1 的第0行就是原始特征 (因为它是沿着行复制的，列在变？不对)
        # node_outputs1: unsqueeze(1) -> [B, 1, L, D] -> expand -> [B, L, L, D]
        # 它的第 [b, 0, i, :] 是第 i 个 token 的特征。说明它沿 dim 1 (行) 是重复的。
        # 所以取 node1[:, 0, :, :] 即可得到 [B, L, D]。
        node_raw_col = node1[:, 0, :, :]  # [B, L, H]

        # node2 是 node1 permute 得到的。
        # node2[:, :, 0, :] 即可得到 [B, L, H]
        node_raw_row = node2[:, :, 0, :]  # [B, L, H]

        w_n1 = weight[:, idx_node1:idx_node2]
        w_n2 = weight[:, idx_node2:idx_end]

        # 在小维度上计算 Linear
        out_n1 = F.linear(node_raw_col, w_n1)  # [B, L, dim_e]
        out_n2 = F.linear(node_raw_row, w_n2)  # [B, L, dim_e]

        # 4. 汇总与 Broadcasting
        # out_edge: [B, L, L, dim_e]
        # diag_feat_i: [B, L, dim_e] -> unsqueeze(1) -> [B, 1, L, dim_e] (Broadcasting add)
        # diag_feat_j: [B, L, dim_e] -> unsqueeze(2) -> [B, L, 1, dim_e] (Broadcasting add)
        # out_n1 (来自 node1, 沿行重复): [B, L, dim_e] -> unsqueeze(1) -> [B, 1, L, dim_e]
        # out_n2 (来自 node2, 沿列重复): [B, L, dim_e] -> unsqueeze(2) -> [B, L, 1, dim_e]

        # 注意 PyTorch Broadcasting 规则：
        # node1 (source code): unsqueeze(1).expand(B, L, L, D).
        # 意味着 dimension 1 (size L) 是重复的。数据随 dimension 2 (size L) 变化。
        # 所以 node1 对应的是 "Column" 特征，需要 unsqueeze(1) 来匹配 [B, L, L]。

        pred_logit = (
                out_edge
                + diag_feat_i.unsqueeze(1)
                + diag_feat_j.unsqueeze(2)
                + out_n1.unsqueeze(1)
                + out_n2.unsqueeze(2)
                + bias
        )

        return pred_logit

class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, device, gcn_dim, edge_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()
        self.gcn_dim = gcn_dim
        self.edge_dim = edge_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.pooling = pooling
        self.layernorm = LayerNorm(self.gcn_dim)
        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.highway = RefiningStrategy(gcn_dim, self.edge_dim, self.dep_embed_dim, dropout_ratio=0.5)

    # weight_prob_softmax, weight_prob, gcn_outputs, self_loop
    def forward(self, weight_prob_softmax, weight_adj, gcn_inputs, self_loop):
        batch, seq, dim = gcn_inputs.shape
        weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2)

        # 优化点：不需要显式 expand 成 [B, Edge, Seq, Dim]，这会占用大量显存
        # 只需要 unsqueeze 对应维度，matmul 会自动广播
        # 原代码: gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.edge_dim, seq, dim)
        gcn_inputs_expanded = gcn_inputs.unsqueeze(1)  # [B, 1, Seq, Dim]

        weight_prob_softmax = weight_prob_softmax + self_loop

        # Matmul 支持广播: [B, Edge, Seq, Seq] @ [B, 1, Seq, Dim] -> [B, Edge, Seq, Dim]
        Ax = torch.matmul(weight_prob_softmax, gcn_inputs_expanded)

        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)

        gcn_outputs = self.W(Ax)
        gcn_outputs = self.layernorm(gcn_outputs)
        weights_gcn_outputs = F.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs
        weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous()

        # 这里的 node_outputs1/2 生成逻辑保持，
        # 因为我们在 RefiningStrategy 里做了还原，所以这里不需要变
        # 只要保证 RefiningStrategy 接收的是这个形状即可
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()

        pred_logit = self.highway(weight_adj, node_outputs1, node_outputs2)

        return node_outputs, pred_logit

class Biaffine(nn.Module):
    def __init__(self, args, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.args = args
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = torch.nn.Linear(in_features=self.linear_input_size,
                                    out_features=self.linear_output_size,
                                    bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.args.device)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.args.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine


class DeepBiaffine(nn.Module):
    def __init__(self, args, input_dim, mlp_dim, out_features, dropout=0.33):
        super(DeepBiaffine, self).__init__()
        self.args = args
        self.mlp_start = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mlp_end = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.biaffine = Biaffine(args, mlp_dim, mlp_dim, out_features, bias=(True, True))

    def forward(self, h):
        # h: [batch, seq, bert_dim]
        start_feat = self.mlp_start(h)
        end_feat = self.mlp_end(h)
        return self.biaffine(start_feat, end_feat)


# 在 EMCGCN __init__ 中替换：
# self.triplet_biaffine = DeepBiaffine(args, bert_feature_dim, args.gcn_dim, args.class_num)

class ThreeWayDecisionParams(nn.Module):
    """优化后的三支决策参数管理模块"""

    def __init__(self, num_labels, init_a=0.3, init_b=0.7, eps=1e-8):
        super().__init__()
        self.num_labels = num_labels
        self.eps = eps

        # -----------------------------------------------------------------------
        # [创新点优化] 类别感知初始化 (Class-Aware Initialization)
        # -----------------------------------------------------------------------
        # 我们不希望所有类别的初始阈值都一样。
        # 对于难识别的 Argument (通常对应 label index 2, 3)，我们降低拒绝阈值 a。
        # 这样模型在早期更容易 Recall 难样本，而不是直接 Reject。

        # 构造初始 a 向量
        init_a_vec = torch.full((num_labels,), init_a)

        # 假设 Label 顺序是: B-T, I-T, B-A, I-A ... (根据 data.py)
        # B-A(idx 2) 和 I-A(idx 3) 是难点，降低初始门槛到 0.15
        if num_labels > 3:
            init_a_vec[2] = 0.15
            init_a_vec[3] = 0.15

        self.alpha = nn.Parameter(self._inverse_sigmoid_tensor(init_a_vec))

        # beta 控制带宽 (b - a)，保持默认即可
        self.beta = nn.Parameter(torch.full((num_labels,), self._inverse_sigmoid_scalar(init_b - init_a)))

        self.register_buffer('exp1', torch.tensor(1.4))
        self.register_buffer('exp2', torch.tensor(1.6))

    def _inverse_sigmoid_scalar(self, x):
        return torch.logit(torch.tensor(x, dtype=torch.float32)).item()

    def _inverse_sigmoid_tensor(self, x):
        # 确保在安全范围内
        return torch.logit(torch.clamp(x, min=1e-4, max=1 - 1e-4))

    def get_parameters(self):
        """获取满足约束的a, b参数"""
        a = torch.sigmoid(self.alpha)
        b = a + (1 - a) * torch.sigmoid(self.beta)
        return a, b

    def forward(self, x):
        a, b = self.get_parameters()

        # 扩展维度以适配输入 [Batch, Seq, Seq, N]
        # a, b: [N] -> [1, 1, 1, N]
        if x.dim() == 4:
            a = a.view(1, 1, 1, -1)
            b = b.view(1, 1, 1, -1)

        diff = b - a
        one_minus_b = 1 - b

        denom1 = (a.pow(self.exp1) * diff.pow(self.exp2)) + self.eps
        denom2 = (one_minus_b.pow(self.exp1) * diff.pow(self.exp2)) + self.eps

        term1 = (x - a) / denom1
        term2 = (b - x) / denom2

        sig1 = torch.sigmoid(term1)
        sig2 = torch.sigmoid(term2)

        p_continue = sig1 * sig2
        return p_continue


class HSD(torch.nn.Module):
    def __init__(self, args):
        super(HSD, self).__init__()
        self.args = args
        if args.model_path.name in ['xlm-roberta-large']:
            self.bert = AutoModel.from_pretrained(args.model_path)
        if args.model_path.name in ['chinese-roberta-wwm-ext-large', 'chinese-macbert-large']:
            self.bert = BertModel.from_pretrained(args.model_path)
        else:
            self.bert = AutoModel.from_pretrained(args.model_path)

        bert_feature_dim = self.bert.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.dropout_output = torch.nn.Dropout(args.emb_dropout)

        self.post_emb = torch.nn.Embedding(args.post_size, args.class_num, padding_idx=0)
        self.deprel_emb = torch.nn.Embedding(args.deprel_size, args.class_num, padding_idx=0)
        self.postag_emb = torch.nn.Embedding(args.postag_size, args.class_num, padding_idx=0)
        self.synpost_emb = torch.nn.Embedding(args.synpost_size, args.class_num, padding_idx=0)

        # 使用 DeepBiaffine
        self.triplet_biaffine = DeepBiaffine(args, bert_feature_dim, args.gcn_dim, args.class_num)

        # 注意：ap_fc 和 op_fc 不再需要，因为 DeepBiaffine 内部包含了 MLP 层
        # 为了保持代码整洁，你可以注释掉或删除它们，或者保留但不使用
        self.ap_fc = nn.Linear(bert_feature_dim, args.gcn_dim)
        self.op_fc = nn.Linear(bert_feature_dim, args.gcn_dim)

        self.dense = nn.Linear(bert_feature_dim, args.gcn_dim)
        self.num_layers = args.num_layers
        self.gcn_layers = nn.ModuleList()

        self.layernorm = LayerNorm(bert_feature_dim)

        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(args.device, args.gcn_dim, 5 * args.class_num, args.class_num, args.pooling))

        self.three_way_layers = nn.ModuleList([
            ThreeWayDecisionParams(args.class_num) for _ in range(self.num_layers)
        ])

    def get_thresholds(self):
        """
        获取所有时间步的阈值参数
        返回: thresholds张量，形状为[S, N, 2]
        """
        thresholds = []
        for layer in self.three_way_layers:
            a, b = layer.get_parameters()
            layer_thresholds = torch.stack([a, b], dim=1)
            thresholds.append(layer_thresholds)
        return torch.stack(thresholds, dim=0)

    def forward(self, tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost):
        bert_feature = self.bert(input_ids=tokens, attention_mask=masks).last_hidden_state
        bert_feature = self.dropout_output(bert_feature)

        batch, seq = masks.shape
        tensor_masks = masks.unsqueeze(1).expand(batch, seq, seq).unsqueeze(-1)

        # * multi-feature
        word_pair_post_emb = self.post_emb(word_pair_position)
        word_pair_deprel_emb = self.deprel_emb(word_pair_deprel)
        word_pair_postag_emb = self.postag_emb(word_pair_pos)
        word_pair_synpost_emb = self.synpost_emb(word_pair_synpost)

        # BiAffine - 修复部分
        # DeepBiaffine 内部包含 mlp_start 和 mlp_end，直接传入 bert_feature 即可
        # 移除之前的 ap_node 和 op_node 计算
        biaffine_edge = self.triplet_biaffine(bert_feature)  #

        gcn_input = F.relu(self.dense(bert_feature))
        gcn_outputs = gcn_input

        weight_prob_list = [biaffine_edge, word_pair_post_emb, word_pair_deprel_emb, word_pair_postag_emb,
                            word_pair_synpost_emb]

        # 注意：F.sigmoid 已被弃用，建议使用 torch.sigmoid，但这不影响运行
        biaffine_edge_sigmoid = torch.sigmoid(biaffine_edge) * tensor_masks
        word_pair_post_emb_sigmoid = torch.sigmoid(word_pair_post_emb) * tensor_masks
        word_pair_deprel_emb_sigmoid = torch.sigmoid(word_pair_deprel_emb) * tensor_masks
        word_pair_postag_emb_sigmoid = torch.sigmoid(word_pair_postag_emb) * tensor_masks
        word_pair_synpost_emb_sigmoid = torch.sigmoid(word_pair_synpost_emb) * tensor_masks

        self_loop = []
        for _ in range(batch):
            self_loop.append(torch.eye(seq))
        self_loop = torch.stack(self_loop).to(self.args.device).unsqueeze(1).expand(batch, 5 * self.args.class_num, seq,
                                                                                    seq) * tensor_masks.permute(0, 3, 1,
                                                                                                                2).contiguous()

        weight_prob = torch.cat([biaffine_edge, word_pair_post_emb, word_pair_deprel_emb, \
                                 word_pair_postag_emb, word_pair_synpost_emb], dim=-1)
        weight_prob_sigmoid = torch.cat([biaffine_edge_sigmoid, word_pair_post_emb_sigmoid, \
                                         word_pair_deprel_emb_sigmoid, word_pair_postag_emb_sigmoid,
                                         word_pair_synpost_emb_sigmoid], dim=-1)

        logit_history = []
        p_continue_history = []
        for _layer in range(self.num_layers):
            gcn_outputs, pred_logit = self.gcn_layers[_layer](weight_prob_sigmoid, weight_prob, gcn_outputs,
                                                              self_loop)  # [batch, seq, dim]

            pred_logit_sigmoid = torch.sigmoid(pred_logit)

            p_continue_history.append(self.three_way_layers[_layer](pred_logit_sigmoid))

            logit_history.append(pred_logit)  # pred_logit: [B, L, L, N]

        return weight_prob_list, torch.stack(logit_history), torch.stack(p_continue_history)


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.target_params = {}  # 缓存

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        # 缓存逻辑：避免每次都遍历几百层网络寻找 embedding
        if emb_name not in self.target_params:
            self.target_params[emb_name] = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and emb_name in name:
                    self.target_params[emb_name].append((name, param))

        # 直接使用缓存的参数列表
        for name, param in self.target_params[emb_name]:
            self.backup[name] = param.data.clone()
            norm = torch.norm(param.grad)
            if norm != 0 and not torch.isnan(norm):
                r_at = epsilon * param.grad / norm
                param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # 同样使用缓存
        if emb_name in self.target_params:
            for name, param in self.target_params[emb_name]:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}