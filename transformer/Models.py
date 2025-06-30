import torch
import torch.nn as nn
import numpy as np

import transformer.Constants as Constants
from .Layers import FFTBlock
from text.symbols import symbols

import json
import os

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1                          #位置编码的最大序列长度加一。
        n_src_vocab = len(symbols) + 1                                  #源词汇表的大小，加上一个额外的单位（通常是用于填充的特殊标记）。
        d_word_vec = config["transformer"]["encoder_hidden"]            #单词嵌入的维度。
        n_layers = config["transformer"]["encoder_layer"]               #编码器中的层数。

        n_head = config["transformer"]["encoder_head"]                  #多头注意力中头的数量。
        d_k = d_v = (                                                   #（key）和值（value）的维度，通常是隐藏层维度除以头数的结果。
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]               #模型的维度
        d_inner = config["transformer"]["conv_filter_size"]             #卷积层的滤波器大小
        kernel_size = config["transformer"]["conv_kernel_size"]         #卷积核大小
        dropout = config["transformer"]["encoder_dropout"]              #用于正则化的 dropout 比率

        self.max_seq_len = config["max_seq_len"]                        #个值代表模型能够处理的最大序列长度。在自然语言处理任务中，这通常是输入文本的最大单词数或最大字符数。这个值对于确定位置编码的大小和嵌入层的输出维度非常重要。
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )                                                               #单词嵌入层，将词汇表中的单词索引转换为嵌入向量
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )                                                                       #位置编码，用于给模型提供序列中元素的位置信息。
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)           #unsqueeze（1）:升1个维度

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:               #不在训练模式
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[           #词嵌入+位置向量
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask        #4层FFT block
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask

class EmotionEncoder(nn.Module):

    def __init__(self, preprocess_config,config):
        super(EmotionEncoder, self).__init__()

        n_position = config["max_seq_len"] + 1                          #位置编码的最大序列长度加一。
        n_src_vocab = len(symbols) + 1                                  #源词汇表的大小，加上一个额外的单位（通常是用于填充的特殊标记）。
        d_word_vec = config["transformer"]["emotion_encoder_hidden"]            #单词嵌入的维度。
        #d_word_vec = preprocess_config["preprocessing"]["mel"]["n_mel_channels"] + 2
        n_layers = config["transformer"]["emotion_encoder_layer"]               #编码器中的层数。

        n_head = config["transformer"]["emotion_encoder_head"]                  #多头注意力中头的数量。
        d_k = d_v = (                                                   #（key）和值（value）的维度，通常是隐藏层维度除以头数的结果。
            config["transformer"]["emotion_encoder_hidden"]
            // config["transformer"]["emotion_encoder_head"]
        )
        d_model = config["transformer"]["emotion_encoder_hidden"]               #模型的维度
        d_model_output = config["transformer"]["emotion_encoder_output"]
        d_inner = config["transformer"]["conv_filter_size"]             #卷积层的滤波器大小
        kernel_size = config["transformer"]["conv_kernel_size"]         #卷积核大小
        dropout = config["transformer"]["emotion_encoder_dropout"]              #用于正则化的 dropout 比率

        self.max_seq_len = config["max_mel_len"]                        #个值代表模型能够处理的最大序列长度。在自然语言处理任务中，这通常是输入文本的最大单词数或最大字符数。这个值对于确定位置编码的大小和嵌入层的输出维度非常重要。
        self.d_model = d_model
        self.d_model_output = d_model_output

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )                                                               #单词嵌入层，将词汇表中的单词索引转换为嵌入向量
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )                                                               #位置编码，用于给模型提供序列中元素的位置信息。

        #FFT block                                                                      
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        #修改输出维度
        self.fc = nn.Linear(self.d_model, self.d_model_output)
        #映射m,logs
        self.proj= nn.Conv1d(self.d_model_output, self.d_model_output * 2, 1)

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:               #不在训练模式 且序列长度大于模型预设长度
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:   #词嵌入+位置向量
            enc_output = src_seq + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask        #4层FFT block
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        #linear mking dim from 80 to 256
        mask = (~mask).unsqueeze(1)
        enc_output = self.fc(enc_output) * mask.transpose(1, 2) # [b, h, t]

        #返回m和logs
        stats = self.proj(enc_output) * mask
        m, logs = torch.split(stats, self.d_model_output, dim=1)

        return enc_output, m, logs  # [b, h, t]

class Average(nn.Module):
    def __init__(self, dim=1):
        super(Average, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        # 应用掩码，将无效部分置为零
        x_masked = x * ((~mask).unsqueeze(-1)).float()

        # 计算有效元素的数量
        valid_count = (~mask).sum(dim=self.dim, keepdim=True).float()

        sumx = torch.sum(x_masked, dim=self.dim, keepdim=True)
        # 确保 valid_count 的形状与 sumx 一致
        if valid_count.dim() == 2:
            valid_count = valid_count.unsqueeze(-1)  # 添加一个新的维度
        valid_count = valid_count.expand_as(sumx)

        # 计算平均值，除以有效元素的数量
        mean_val = sumx / valid_count

        return mean_val