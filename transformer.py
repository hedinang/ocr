from collections import OrderedDict
from torchvision import models
import math
import torch
from torch import nn


class LanguageTransformer(nn.Module):
    def __init__(self, vocab_size,
                 d_model, nhead,
                 num_encoder_layers, num_decoder_layers,
                 dim_feedforward, max_seq_length,
                 pos_dropout, trans_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        self.transformer = nn.Transformer(d_model, nhead,
                                          num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, trans_dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, beam_size=4):
        src = self.pos_enc(src*math.sqrt(self.d_model))
        memory = self.transformer.encoder(src)
        memory = memory.repeat(1, beam_size, 1)
        return memory

    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).permute(1, 0)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward_decoder(self, tgt, memory):
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        return self.fc(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


class Backbone(nn.Module):
    def __init__(self, ss, ks, hidden, dropout=0.5):
        super(Backbone, self).__init__()
        cnn = models.vgg19_bn()
        pool_idx = 0
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn.features[i] = torch.nn.AvgPool2d(
                    kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1
        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

    def forward(self, x):
        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv


class Transformer(nn.Module):
    def __init__(self, vocab_size,
                 backbone={
                     'ss': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                     'ks': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                     'hidden': 256},
                 transformer={
                     'd_model': 256,
                     'nhead': 8,
                     'num_encoder_layers': 6,
                     'num_decoder_layers': 6,
                     'dim_feedforward': 2048,
                     'max_seq_length': 1024,
                     'pos_dropout': 0.1,
                     'trans_dropout': 0.1
                 }):

        super(Transformer, self).__init__()
        self.cnn = Backbone(**backbone)
        self.transformer = LanguageTransformer(vocab_size, **transformer)

    def forward(self, img):
        src = self.cnn(img)
        return self.transformer(src)
