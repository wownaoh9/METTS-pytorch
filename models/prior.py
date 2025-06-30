import math
import torch
from torch import nn
from torch.nn import functional as F

from models.commons import sequence_mask
import models.attentions

class TextEncoder(nn.Module):
  def __init__(self, model_config):
    super().__init__()
    n_vocab = model_config["max_seq_len"]
    self.out_channels = model_config["FG_predictor"]["inter_channels"]   #vits是这样的
    hidden_channels = model_config["FG_predictor"]["hidden_channels"]
    filter_channels = model_config["FG_predictor"]["filter_channels"]
    n_heads = model_config["FG_predictor"]["n_heads"]
    n_layers = model_config["FG_predictor"]["n_layers"]
    kernel_size = model_config["FG_predictor"]["kernel_size"]
    p_dropout = model_config["FG_predictor"]["p_dropout"]

    # self.emb = nn.Embedding(n_vocab, hidden_channels)
    # nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = models.attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, self.out_channels * 2, 1)

  def forward(self, x, x_lengths):  # x: [b, t, h]
    # x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask