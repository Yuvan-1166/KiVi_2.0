
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class TamilLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.embed_positions = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits
