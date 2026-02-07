import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # （maxlen,1）
        
        # 对数变换：a^b = e^(b*ln(a))
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, d_model)
        # 步长为2，2i！

        pe[:, 0::2] = torch.sin(position*div_term)
        # 索引从0开始，步长为2
        pe[:, 1::2] = torch.cos(position*div_term)

        # 将 pe 注册为 buffer，这样它就不会被视为模型参数，但会随模型移动（例如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0)) # (1,max_len,d_model)
        # unsqueeze：在特定位置挤出一个维度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)] # 根据实际长度切pe
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0  # 必须能整除

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 我觉得原文这里有点问题，应该是每个头的维度

        self.W_q = nn.Linear(d_model, d_model)  # 创建特征矩阵Q
        self.W_k = nn.Linear(d_model, d_model)  # K
        self.W_v = nn.Linear(d_model, d_model)  # V
        self.W_o = nn.Linear(d_model, d_model)  # Output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # QK^T
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1))/math.sqrt(self.d_k)
        # transpose to multiple: Q(m,n) * K^T(n,m) -> (m,m)
        # -2: the last second dim: seq_len, -1: the last dim: d_k
        # broadcasting in matmul: keep the other dims: batches, heads...

        if mask is not None:
            # 将掩码中为 0 的位置设置为一个非常小的负数，这样 softmax 后会接近 0
            # 不需要的信息会被过滤
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            # mask: a corresponding tensor to attn_scores only including 0 and 1
            # boolean tensor -> true(0) -> -1e-9

        attn_probs = torch.softmax(attn_scores, dim=-1)
        # 对K进行softmax，即词得分

        # *V
        output = torch.matmul(attn_probs, V)  # (m*m)*(m*n)->(m,n)
        return output

    def split_heads(self, x):
        # 将输入 x 的形状从 (batch_size, seq_length, d_model)
        # 变换为 (batch_size, num_heads, seq_length, d_k)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        # view 只能作用于连续（contiguous）的内存。因为输入 x 是刚经过 nn.Linear 出来的，内存是连续的

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x形状 (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 掩码多头自注意力：模型在预测当前这个词时，只能看前面已经生成出来的词
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # ！交叉注意力：去原文里寻找最相关的线索
        cross_attn_output = self.cross_attn(
            x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x+self.dropout(cross_attn_output))

        # feedforward
        ff_output = self.feed_forward(x)
        x = self.norm3(x+self.dropout(ff_output))

        return x
