import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class GST(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        
        self.encoder = ReferenceEncoder(model_config)
        self.stl = STL(model_config)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths)
        style_embed = self.stl(enc_out)

        return style_embed.squeeze(1) # [N, 1, 256]


class ReferenceEncoder(nn.Module):
    """Reference Encoder
    Getting style embedding from Reference Audio
    - six 2D convolution layers, and a GRU layer
    """

    def __init__(self, model_config):
        super().__init__()

        K = len(model_config["gst"]["conv_filters"])
        filters = [1] + model_config["gst"]["conv_filters"]

        convs = [
            nn.Conv2d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=model_config["gst"]["conv_filters"][i]) for i in range(K)])

        out_channels = self.calculate_channels(model_config["gst"]["n_mel_channels"], 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=model_config["gst"]["conv_filters"][-1] * out_channels,
            hidden_size=model_config["gst"]["ref_enc_gru_size"],
            batch_first=True,
        )
        self.n_mel_channels = model_config["gst"]["n_mel_channels"]
        self.ref_enc_gru_size = model_config["gst"]["ref_enc_gru_size"]
        self.in_channels = model_config["gst"]["in_channels"]

    def forward(self, inputs, input_lengths=None):
        """
        Args:
            inputs: [N, length, 80]
            input_lengths: [N]
        Returns:
            embedding: [N, 128]
        """
        inputs = inputs.unsqueeze(1)
        out = inputs  #  -->[4,3,387,82]

        for conv, bn in zip(self.convs, self.bns):      #zip将这两个打包成元组，每次返回两个值
            # for conv in self.convs:
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]   [4,128,7,2]-->[4,7,128,2]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]      #view()方法用于改变张量的形状（shape）。
                                                                                    # 在这个方法中，你可以指定新的维度大小。如果某个维度被设置为-1，
                                                                                    # PyTorch 会自动计算这个维度的大小，以便保持总元素数量不变。

        # pack padded sequence
        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / 2 ** len(self.convs))
            input_lengths = input_lengths.cpu().numpy().astype(int)
            out = nn.utils.rnn.pack_padded_sequence(out, input_lengths, batch_first=True, enforce_sorted=False)     #这个函数的主要目的是让循环层仅处理序列的有效部分，而忽略填充的部分。
        
        self.gru.flatten_parameters()  # Resets parameter data pointer
        _, out = self.gru(out)
        emo_emb = out.squeeze(0)  # [N, 128]

        return emo_emb # [N, 128]

    @staticmethod
    def calculate_channels(L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, model_config):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(model_config["gst"]["token_num"], model_config["gst"]["E"] // model_config["gst"]["num_heads"]))
        d_q = model_config["gst"]["E"] // 2
        d_k = model_config["gst"]["E"] // model_config["gst"]["num_heads"]
        # self.attention = MultiHeadAttention(hp.num_heads, d_model, d_q, d_v)
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=model_config["gst"]["E"], num_heads=model_config["gst"]["num_heads"])

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class MultiHeadAttention2(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 # dropout_p=0.5,
                 h=8,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        # if query_dim != key_dim:
        #     raise ValueError("query_dim and key_dim must be the same")
        # if num_units % h != 0:
        #     raise ValueError("num_units must be dividable by h")
        # if query_dim != num_units:
        #     raise ValueError("to employ residual connection, the number of "
        #                      "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        # self._key_dim = torch.tensor(
        #     data=[key_dim], requires_grad=True, dtype=torch.float32)
        self._key_dim = key_dim
        # self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        # self.bn = nn.BatchNorm1d(num_units)

    def forward(self, query, keys):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention / (self._key_dim ** 0.5)
        # use masking (usually for decoder) to prevent leftward
        # information flow and retains auto-regressive property
        # as said in the paper
        if self._is_masked:
            diag_vals = attention[0].sign().abs()
            diag_mat = diag_vals.tril()
            diag_mat = diag_mat.unsqueeze(0).expand(attention.size())
            mask = torch.ones(diag_mat.size()) * (-2**32 + 1)
            # this is some trick that I use to combine the lower diagonal
            # matrix and its masking. (diag_mat-1).abs() will reverse the value
            # inside diag_mat, from 0 to 1 and 1 to zero. with this
            # we don't need loop operation andn could perform our calculation
            # faster
            attention = (attention * diag_mat) + (mask * (diag_mat - 1).abs())
        # put it to softmax
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        # attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)

        return attention
