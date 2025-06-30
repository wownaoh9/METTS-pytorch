import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.commons import sequence_mask

class LayerNorm(torch.nn.Module):
    def __init__(self, nout: int):
        super(LayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(nout, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x.transpose(1, -1))
        x = x.transpose(1, -1)
        return x

class PhonemeLevelPredictor(nn.Module):

    def __init__(self, idim: int,
                 n_layers: int = 2,
                 n_chans: int = 384,
                 out: int = 16,
                 kernel_size: int = 3,
                 dropout_rate: float = 0.5,
                 stride: int = 1):
        super(PhonemeLevelPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

        self.linear = torch.nn.Linear(n_chans, out)

    def forward(self,
                xs: torch.Tensor,
                x_masks: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax)

        return xs

class PhonemeLevelEncoder(nn.Module):

    def __init__(self, idim: int,
                    n_layers: int = 2,
                    n_chans: int = 80,
                    out: int = 16,
                    kernel_size: int = 3,
                    dropout_rate: float = 0.5,
                    stride: int = 1):
        super(PhonemeLevelEncoder, self).__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

        self.linear = torch.nn.Linear(n_chans, out)

    def forward(self,
                xs: torch.Tensor,
                x_masks: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        for f in self.conv:
            xs = f(xs)  # (B, C, Lmax)


        xs = self.linear(xs.transpose(1, 2))  # (B, Lmax, 4)

        return xs

class MelEncoder(nn.Module):
  def __init__(self, model_config):
    super().__init__()
    n_vocab = model_config["max_seq_len"]
    self.out_channels = model_config["FG_encoder"]["inter_channels"]   #vits是这样的
    hidden_channels = model_config["FG_encoder"]["hidden_channels"]
    filter_channels = model_config["FG_encoder"]["filter_channels"]
    n_heads = model_config["FG_encoder"]["n_heads"]
    n_layers = model_config["FG_encoder"]["n_layers"]
    kernel_size = model_config["FG_encoder"]["kernel_size"]
    p_dropout = model_config["FG_encoder"]["p_dropout"]

    # self.emb = nn.Embedding(n_vocab, hidden_channels)
    # nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = models.attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.linear = nn.Linear(hidden_channels, self.out_channels)
    

  def forward(self, x, x_lengths):  # x: [b, t, h]
    # x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    x = self.linear(x.transpose(1, 2)).transpose(1, 2)
    # stats = self.proj(x) * x_mask

    # m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, x_mask

class PhonemeLevelMelAverage(nn.Module):
    """
    输入：
        - mel: 帧级别的 Mel 频谱特征，形状为 (B, T, D)
        - duration: 音素的持续时间，形状为 (B, P)
            duration[i, j] 表示第 i 个样本中第 j 个音素的持续时间（帧数）
    输出：
        - phoneme_mel: 音素级别的 Mel 频谱特征，形状为 (B, P, D)
    """
    def __init__(self):
        super(PhonemeLevelMelAverage, self).__init__()

    def forward(self, mel, duration):
        """
            mel: 帧级别的 Mel 频谱特征，形状为 (B, T, D)
            duration: 音素的持续时间，形状为 (B, P)
        """
        batch_size, total_frames, mel_dim = mel.size()
        batch_size, num_phonemes = duration.size()
        
        # 初始化音素级别的 Mel 频谱特征
        phoneme_mel = torch.zeros(batch_size, num_phonemes, mel_dim, device=mel.device)
        
        # 遍历每个样本
        for b in range(batch_size):
            pos = 0  # 当前帧的位置
            for p in range(num_phonemes):
                d = duration[b, p].item()  # 当前音素的持续时间
                if d > 0:
                    # 计算当前音素的 Mel 频谱平均值
                    phoneme_mel[b, p] = mel[b, pos:pos + d].mean(dim=0)
                pos += d  # 更新帧的位置
        
        return phoneme_mel
    
class ConvClassifier(nn.Module):
    """
    使用 Conv1d 的分类器模块。
    """
    def __init__(self, in_channels, num_classes, hidden_channels=128, kernel_size=3, padding=1):
        super(ConvClassifier, self).__init__()
        self.in_channels = in_channels  # 输入特征维度
        self.num_classes = num_classes  # 输出类别数
        self.hidden_channels = hidden_channels  # 隐藏层通道数
        self.kernel_size = kernel_size  # 卷积核大小
        self.padding = padding  # 填充大小

        # 定义卷积层
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                               out_channels=hidden_channels, 
                               kernel_size=kernel_size, 
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, 
                               out_channels=hidden_channels, 
                               kernel_size=kernel_size, 
                               padding=padding)

        # 定义全连接层
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        """
        Args:
            x: [b, dim, mel_len]  
        Returns:
            cls: [b, num_classes]  
        """
        x = torch.transpose(x, 1, -1)
        # 调整输入张量的形状以适应 Conv1d
        # Conv1d 的输入形状为 (batch_size, in_channels, sequence_length)
        x = x.permute(0, 2, 1)  # [b, dim, mel_len]

        # 通过卷积层
        x = F.relu(self.conv1(x))  # [b, hidden_channels, mel_len]
        x = F.relu(self.conv2(x))  # [b, hidden_channels, mel_len]

        # 对时间步进行池化，例如取全局平均池化
        x = x.mean(dim=2)  # [b, hidden_channels]

        # 通过全连接层进行分类
        x = self.fc(x)  # [b, num_classes]

        return x