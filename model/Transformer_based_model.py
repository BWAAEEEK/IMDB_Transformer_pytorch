import torch
import torch.nn as nn
import math

class Transformer(nn.Module):

    def __init__(self, vocab_size, args):
        super().__init__()

        self.hidden_size = args.hidden_size

        # initial embedding
        self.tok_emb = nn.Embedding(vocab_size, self.hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(args.max_len, self.hidden_size)
        # self.pos_emb = PositionalEncoding(vocab_size, self.hidden_size)

        # transformer_encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=args.attn_heads,
                dim_feedforward=32
            ),
            num_layers=args.transformer_layers
        )

        # linear layer for prediction
        self.linear_1 = nn.Linear(self.hidden_size, 2)

    def forward(self, inputs, positions):
        x = self.tok_emb(inputs) + self.pos_emb(positions)
        x = nn.Dropout(p=0.1)(x)
        x = self.transformer_encoder(x)

        # x = x.view(-1) forward 할때 batch size 행렬 전체를 불러오는지 아니면 iteration 써서 input 하나씩 불러오는지 찾아보기

        # x = x.mean(dim=1)  # sos 토큰 없을 경우 평균으로 aggregation 후 classification 진행

        x = self.linear_1(x[:, 0])

        return x

# 기존 방식처럼 sin cos을 이용하여 Postional Encoding을 해주는 방식
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]

        return x
