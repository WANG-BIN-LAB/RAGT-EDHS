import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpretableTransformerEncoder(nn.Module):
    """Custom Transformer Encoder with batch_first support"""
    def __init__(self, d_model, nhead, dim_feedforward, batch_first=True, dropout=0):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.batch_first = batch_first

    def forward(self, src, src_key_padding_mask=None):
        return self.transformer(src, src_key_padding_mask=src_key_padding_mask)


class SparseAttention(nn.Module):
    """Sparse Attention module with Global Top-K and Local Neighbor masking"""
    def __init__(self, cfg, embed_dim=200, dropout=0.1):
        super().__init__()
        assert embed_dim % cfg.model.nhead == 0
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.num_heads = cfg.model.nhead
        self.head_dim = embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [y.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2) for y in qkv]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Global Top-K Mask
        attn_global = attn_scores.mean(dim=(0, 1), keepdim=True)
        global_topk = max(1, int(seq_len * self.cfg.model.global_topk_ratio))
        _, global_topk_indices = torch.topk(attn_global, k=global_topk, dim=-1)
        attention_mask = torch.zeros(batch, self.num_heads, seq_len, seq_len, dtype=torch.bool, device=x.device)
        attention_mask.scatter_(dim=-1, index=global_topk_indices.expand(batch, self.num_heads, seq_len, global_topk),
                                value=True)

        # Local Neighbor Mask
        local_neighbor_num = min(self.cfg.model.local_neighbor_num, seq_len)
        attn_norm_sq = torch.sum(attn_scores ** 2, dim=-1, keepdim=True)
        dist_sq = attn_norm_sq + attn_norm_sq.transpose(-2, -1) - 2 * attn_scores
        dist_sq = torch.clamp(dist_sq, min=0.0)
        dist_matrix = torch.sqrt(dist_sq)
        _, local_topk_indices = torch.topk(-dist_matrix, k=local_neighbor_num, dim=-1)
        attention_mask.scatter_(dim=-1, index=local_topk_indices, value=True)

        # Apply combined attention mask
        attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        output = self.out_proj(output)
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Encoder layer with sparse self-attention and feed-forward network"""
    def __init__(self, cfg, d_model, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = SparseAttention(cfg, embed_dim=d_model, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src, attn_weights


class SparseTransformer(nn.Module):
    """Stacked sparse transformer encoder with configurable layers"""
    def __init__(self, cfg, input_dim, d_model, dim_feedforward, dropout):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(cfg, d_model, dim_feedforward, dropout)
            for _ in range(cfg.model.num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        x = self.input_proj(src)
        attn_weights = []
        for layer in self.layers:
            x, w = layer(x, src_key_padding_mask)
            attn_weights.append(w)
        x = self.norm(x)
        return x


class TransPoolingEncoder(nn.Module):
    """Transformer encoder with pooling for brain network feature extraction"""
    def __init__(self, cfg, input_feature_size, input_node_num, hidden_size, output_node_num,
                 orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer0 = SparseTransformer(
            cfg=cfg,
            input_dim=input_feature_size,
            d_model=input_feature_size,
            dim_feedforward=hidden_size,
            dropout=0
        )

    def forward(self, x):
        x = self.transformer0(x)
        return x, None


class BrainNetworkTransformer(nn.Module):
    """Main model: Brain Network Transformer with age-sex feature modulation"""
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.attention_list = nn.ModuleList()
        forward_dim = 200
        sizes = config.model.sizes
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(
                    cfg=config,
                    input_feature_size=forward_dim,
                    input_node_num=200,
                    hidden_size=1024,
                    output_node_num=size,
                    orthogonal=config.model.orthogonal,
                    freeze_center=config.model.freeze_center,
                    project_assignment=config.model.project_assignment
                )
            )
        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 8),
            nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(200 * 8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
        )
        self.W = nn.Parameter(torch.FloatTensor(forward_dim, forward_dim))
        nn.init.xavier_uniform_(self.W)

        # MLP for age and sex feature modulation
        self.age_sex_mlp = nn.Sequential(
            nn.Linear(2, 1),
        )

    def forward(self, time_series, node_feature, site, age, sex):
        bz, num_nodes, _ = node_feature.shape

        # Concatenate age and sex features
        age_sex = torch.stack([age, sex], dim=1)

        # Compute modulation vector
        modulation_vec = self.age_sex_mlp(age_sex)
        modulation = modulation_vec.unsqueeze(1)

        # Apply feature modulation
        node_feature = node_feature * (modulation * self.W) + node_feature

        assignments = []
        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)
        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))
        return self.fc(node_feature), node_feature

    def get_attention_weights(self):
        return []

    def get_cluster_centers(self):
        return torch.tensor([])