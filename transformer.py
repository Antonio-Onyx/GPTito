import torch
import torch.nn as nn
#import torch.nn.functional as F

from layers import InputEmbeddings, PositionalEncoding, MultiHeadAttention, EncoderLayer, DecoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, tgt_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask)
        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.transformer_encoding = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length)
        self.transformer_decoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.softmax = torch.softmax()

    def forward(self, source, target):
        source_mask = torch.ones()
        output_encoder = self.transformer_encoding(source, source_mask())

    def source_mask(self, batch_size, max_seql_length):
        return torch.ones(batch_size, max_seql_length, max_seql_length)
    
    def target_mask(self, target):
        return torch.tril(torch.ones(target.shape[1]), target.shape[1])