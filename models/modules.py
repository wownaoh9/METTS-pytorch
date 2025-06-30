import os
import json
import copy
import math
from collections import OrderedDict

from click import style
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils.tools import get_mask_from_lengths,Concate_tensor
from utils.tools import get_mask_from_lengths, pad

from .GST import GST
from .conformer import MelConformer
from .FG import PhonemeLevelPredictor, PhonemeLevelMelAverage, PhonemeLevelEncoder
from models import flow

class PhoneLevelEmb(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(
            in_features,
            out_features,
        )
    def forward(self, inputs):
        return self.linear(inputs)

class UtterLevelEmb(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(
            in_features,
            out_features,
        )
    def forward(self, inputs):
        return self.linear(inputs)

class MelLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mel_linear = nn.Linear(
            in_features,
            out_features,
        )
    def forward(self, decoder_output):
        return self.mel_linear(decoder_output)
    
class SpeakerEncoder(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super().__init__()
        with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
        self.embedding = nn.Embedding(
                n_speaker,
                model_config["conformer"]["encoder_hidden"],
            )
    def forward(self, speaker):
        speaker_embedding = self.embedding(speaker)
        return speaker_embedding

class CoarseEmoExtractionModule(nn.Module): 
    def __init__(self, preprocess_config, model_config):
        super().__init__()
        self.GST = GST(model_config)
        with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                ),
                "r",
            ) as f:
                n_emotion = len(json.load(f))

        self.Classifier = Classifier(model_config["gst"]["E"], n_emotion)
    def forward(self, mels, mel_len):
        coarse_emo_emb = self.GST(mels, mel_len)
        cls_emb = coarse_emo_emb 
        return coarse_emo_emb, cls_emb

class Classifier(nn.Module):
    """Classifier
    - a Full-connected layer and a soft-max layer
    """

    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        N = 3  # 定义为3个全连接层

        # 创建一个列表来存储全连接层
        linears = [
            nn.Linear(in_features=self.in_features, out_features=self.in_features) for _ in range(N - 1)
        ]
        # 最后一层的输出特征数为out_features
        linears.append(nn.Linear(in_features=self.in_features, out_features=self.out_features))
        self.linears = nn.ModuleList(linears)

    def forward(self, x):
        """
        Args:
            x: [N, D] 输入特征，其中N是样本数，D是特征维度
        Returns:
            cls: [N, num_classes] 输出分类结果
        """
        for i, layer in enumerate(self.linears):
            x = layer(x)

        return x

class FineEmoExtractionModule(nn.Module):
    def __init__(self,  preprocess_config, model_config):
        super().__init__()
        with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                ),
                "r",
            ) as f:
                n_emotion = len(json.load(f))
        self.mel_conformer = MelConformer(model_config)
        self.phoneme_level_encoder = PhonemeLevelEncoder(idim = 80)
        self.phoneme_level_mel_average = PhonemeLevelMelAverage()
        self.Classifier = Classifier(model_config["FG_proj"]["out_channels"], n_emotion)
    
    def forward(self, mels, mel_mask, durations):
        fine_emo_emb = self.mel_conformer(mels, mel_mask)
        fine_emo_emb = self.phoneme_level_encoder(fine_emo_emb.transpose(1, 2), mel_mask)
        phone_emo_emb = self.phoneme_level_mel_average(fine_emo_emb, durations)
        cls_fine_emb = self.Classifier(torch.mean(fine_emo_emb, dim=1))
        cls_phoneme_emb = self.Classifier(torch.mean(phone_emo_emb, dim=1))
        return phone_emo_emb, cls_fine_emb, cls_phoneme_emb

class FineEmoPredictModule(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.PhonemeLevelPredictor = PhonemeLevelPredictor(idim = 384)
    def forward(self, x, x_mask):
        x = self.PhonemeLevelPredictor(x, x_mask)
        return x

class ResidualCouplingBlock(nn.Module):
  # flow
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(flow.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(flow.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["conformer"]["decoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["conformer"]["decoder_hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control, train):
        prediction = self.pitch_predictor(x, mask)
        if target is not None and train == True:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control, train):
        prediction = self.energy_predictor(x, mask)
        if target is not None and train == True:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        train = True
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control, train
            )
            if True:
                x = x + pitch_embedding

        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control, train
            )
            if True:
                x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            #max_len = torch.max(torch.sum(duration_rounded, dim=1))
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control, train
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control, train
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len=None):
        """
        Length Regulation (LR): Adjust the length of the input sequence based on duration.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, feature_dim).
            duration (Tensor): Duration tensor of shape (batch_size, seq_len), where each value specifies the expansion factor.
            max_len (int, optional): The maximum length of the output sequences. If None, it is determined by the maximum length of the expanded sequences.

        Returns:
            Tensor: Output tensor of shape (batch_size, max_len, feature_dim), where sequences are padded to max_len.
            Tensor: Tensor containing the lengths of each expanded sequence, shape (batch_size,).
        """
        output = []
        mel_len = []

        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        # 如果没有提供 max_len，则使用所有序列中的最大长度
        if max_len is None:
            max_len = max(mel_len)

        # 对所有序列进行填充到相同长度
        padded_output = pad_sequence(
            output, batch_first=True, padding_value=0.0
        )  # shape: (batch_size, max_len, feature_dim)
        
        # 如果填充长度小于 max_len，补充剩余的长度
        if padded_output.size(1) < max_len:
            padding_size = max_len - padded_output.size(1)
            padded_output = F.pad(padded_output, (0, 0, 0, padding_size), "constant", 0)

        device = x.device
        return padded_output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        """
        Expand the input sequence based on the predicted durations.

        Args:
            batch (Tensor): Input tensor of shape (seq_len, feature_dim).
            predicted (Tensor): Duration tensor of shape (seq_len,).

        Returns:
            Tensor: Expanded tensor.
        """
        out = []
        for i, vec in enumerate(batch):
            expand_size = max(int(predicted[i].item()), 0)  # 防止负值扩展
            if expand_size > 0:
                out.append(vec.unsqueeze(0).expand(expand_size, -1))
        if out:
            out = torch.cat(out, dim=0)  # 合并成一个张量
        else:
            out = torch.zeros((0, batch.size(1)), device=batch.device)  # 空张量处理
        return out

    def forward(self, x, duration, max_len=None):
        """
        Forward pass for the Length Regulator.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, feature_dim).
            duration (Tensor): Duration tensor of shape (batch_size, seq_len).
            max_len (int, optional): The maximum length of the output sequences.

        Returns:
            Tensor: Output tensor of shape (batch_size, max_len, feature_dim).
            Tensor: Tensor containing the lengths of each expanded sequence.
        """
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len
    
class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["conformer"]["decoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
