import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_softmax(logits, mask, dim=-1):
    if mask is not None:
        logits = logits.masked_fill(~mask.bool(), float('-inf'))
    return torch.softmax(logits, dim=dim)


class FineGrainedGate(nn.Module):
    """token/region 级门控层
    参数:
        d_model: 特征维度
        n_heads: 多头数量
        dropout: dropout 概率
        use_sparse: 是否启用 top-k 稀疏对齐
        topk: 稀疏对齐的 k 值
        mix_beta: 注意力与门控外积的混合系数
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1,
                 use_sparse=False, topk=8, mix_beta=0.3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        self.gate_t = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.gate_v = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        self.mix_beta = mix_beta
        self.use_sparse = use_sparse
        self.topk = topk
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _shape_heads(self, x):
        B, L, D = x.shape
        x = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        return x

    def forward(self, H_t, H_v, attn_mask_v=None, return_weights=False):
        B, T, D = H_t.shape
        R = H_v.size(1)

        r_t = torch.sigmoid(self.gate_t(H_t))    # [B,T,1]
        r_v = torch.sigmoid(self.gate_v(H_v))    # [B,R,1]

        Q = self._shape_heads(self.Wq(H_t))      # [B,H,T,hd]
        K = self._shape_heads(self.Wk(H_v))      # [B,H,R,hd]
        V = self._shape_heads(self.Wv(H_v))      # [B,H,R,hd]

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask_v is not None:
            attn_mask = attn_mask_v[:, None, None, :].expand(-1, self.n_heads, T, -1)
        else:
            attn_mask = None
        A = masked_softmax(attn_logits, attn_mask, dim=-1)

        Rt = r_t.squeeze(-1)[:, None, :, None]
        Rv = r_v.squeeze(-1)[:, None, None, :]
        outer = Rt * Rv
        outer = outer.expand(B, self.n_heads, T, R)

        if self.use_sparse:
            k = min(self.topk, R)
            topk_vals, topk_idx = torch.topk(attn_logits, k, dim=-1)
            sparse_mask = torch.zeros_like(attn_logits, dtype=torch.bool)
            sparse_mask.scatter_(-1, topk_idx, True)
            G = torch.where(sparse_mask, attn_logits, torch.full_like(attn_logits, float('-inf')))
            G = torch.softmax(G, dim=-1)
            mix = 0.5 * A + 0.5 * G
            selected_idx = topk_idx
        else:
            mix = A
            selected_idx = None

        W = (1.0 - self.mix_beta) * mix + self.mix_beta * outer
        W = self.dropout(W)

        ctx = torch.matmul(W, V)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)
        ctx = self.out(ctx)
        H_t_out = H_t + torch.tanh(self.gamma) * (r_t * ctx)

        gates = {"r_t": r_t, "r_v": r_v,
                 "W": (W if return_weights else None),
                 "topk_idx": selected_idx}
        return H_t_out, gates
