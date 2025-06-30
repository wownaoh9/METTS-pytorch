import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms
# from pytorch_msssim import ssim, SSIM
from piq import SSIMLoss
from torch import Tensor

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(data_range=1.0)

    
    def forward(self, inputs, predictions):
        (
            mel_targets,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = (
            inputs[8],
            inputs[11],
            inputs[12],
            inputs[13],
        )
        (
            (output_mellinear_one, output_mellinear_two, output_mellinear_three, output_mellinear_four, output_mellinear_five, output_mellinear_six),
            output_ssim,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            emotions_emo,
            (h_i,h_j,h_k),
            (z_p, logs_q, m_p, logs_p, y_mask) # posterior ->flow-> prior
        ) = predictions

        device = mel_targets.device
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]    
        # mel_predictions = mel_predictions[:, : mel_masks.shape[1], :]
        output_mellinear_one = output_mellinear_one[:, : mel_masks.shape[1], :]
        output_mellinear_two = output_mellinear_two[:, : mel_masks.shape[1], :]
        output_mellinear_three = output_mellinear_three[:, : mel_masks.shape[1], :]
        output_mellinear_four = output_mellinear_four[:, : mel_masks.shape[1], :]
        output_mellinear_five = output_mellinear_five[:, : mel_masks.shape[1], :]
        output_mellinear_six = output_mellinear_six[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        #对pitch和energy的mask做选择从而为计算loss做准备  #masked_select是pytorch的一个函数
        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        #ssim loss
        output_ssim_normalized = (
            sample_wise_min_max(output_ssim).float().to(device)
        )
        mel_targets_normalized = (
            sample_wise_min_max(mel_targets).float().to(device)
        )
       
        ssim_loss = self.ssim_loss(output_ssim_normalized.unsqueeze(1), mel_targets_normalized.unsqueeze(1))

        if ssim_loss.item() > 1.0 or ssim_loss.item() < 0.0:
            ssim_loss = torch.tensor([1.0], device=device)

        #三维张量变成一维张量, 对梅尔谱图的mask做选择从而为计算L1 loss做准备
        #mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        output_mellinear_one = output_mellinear_one.masked_select(mel_masks.unsqueeze(-1))
        output_mellinear_two = output_mellinear_two.masked_select(mel_masks.unsqueeze(-1))
        output_mellinear_three = output_mellinear_three.masked_select(mel_masks.unsqueeze(-1))
        output_mellinear_four = output_mellinear_four.masked_select(mel_masks.unsqueeze(-1))
        output_mellinear_five = output_mellinear_five.masked_select(mel_masks.unsqueeze(-1))
        output_mellinear_six = output_mellinear_six.masked_select(mel_masks.unsqueeze(-1))
        output_ssim = output_ssim.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        #L1 loss
        mel_loss_one = self.mae_loss(output_mellinear_one, mel_targets)
        mel_loss_two = self.mae_loss(output_mellinear_two, mel_targets)
        mel_loss_three = self.mae_loss(output_mellinear_three, mel_targets)
        mel_loss_four = self.mae_loss(output_mellinear_four, mel_targets)
        mel_loss_five = self.mae_loss(output_mellinear_five, mel_targets)
        mel_loss_six = self.mae_loss(output_mellinear_six, mel_targets)
        mel_loss_first_five = mel_loss_one + mel_loss_two + mel_loss_three + mel_loss_four + mel_loss_five
        mel_loss_last_six = mel_loss_six


        # #L1 loss
        # mel_loss = self.mae_loss(mel_predictions, mel_targets)
        # postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        #mse loss
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        #ce loss
        y_emo = emotions_emo.to(device)
        ce_loss_i = F.cross_entropy(h_i, y_emo)
        ce_loss_j = F.cross_entropy(h_j, y_emo)
        ce_loss_k = F.cross_entropy(h_k, y_emo)
        ce_loss = (ce_loss_i + ce_loss_j + ce_loss_k) * 0.1

        #kl loss 
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * 0.05
        
        total_loss = (
            mel_loss_first_five + mel_loss_last_six + ssim_loss + pitch_loss + energy_loss + duration_loss + ce_loss + loss_kl
        )

        return (
            total_loss,
            mel_loss_first_five,
            mel_loss_last_six,
            ssim_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            ce_loss,
            loss_kl
        )
def sample_wise_min_max(x: Tensor) -> Tensor:
    r"""Applies sample-wise min-max normalization to a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_samples, num_features).

    Returns:
        torch.Tensor: Normalized tensor of the same shape as the input tensor.
    """
    # Compute the maximum and minimum values of each sample in the batch
    maximum = torch.amax(x, dim=(1, 2), keepdim=True)
    minimum = torch.amin(x, dim=(1, 2), keepdim=True)

    # Apply sample-wise min-max normalization to the input tensor
    return (x - minimum) / (maximum - minimum)
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p) #torch.exp(x) = e^x
  kl = torch.sum(kl * z_mask)
  # l = kl / torch.sum(z_mask) + 1e-1
  l = kl / torch.sum(z_mask)
  # l = torch.clamp(l, min=0.002)
  return l

# def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
#     """
#     z_p, logs_q: [b, h, t_t]
#     m_p, logs_p: [b, h, t_t]
#     """
#     # 确保输入为浮点类型
#     z_p = z_p.float()
#     logs_q = logs_q.float()
#     m_p = m_p.float()
#     logs_p = logs_p.float()
#     z_mask = z_mask.float()

#     # 数值稳定性：裁剪 logs_p 和 logs_q
#     logs_p = torch.clamp(logs_p, min=-10, max=10)
#     logs_q = torch.clamp(logs_q, min=-10, max=10)

#     # 计算 KL 散度
#     kl_1 = logs_p - logs_q - 0.5
#     kl_2 = kl_1 + 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)  # torch.exp(x) = e^x

#     # 应用掩码并归一化
#     kl = torch.sum(kl_2 * z_mask)
#     l = kl / torch.sum(z_mask)
#     l = torch.clamp(l, min=0)

#     return l

def kl_divergence_gaussians(m_p, logs_p, m_q, logs_q):
    """
    计算两个高斯分布之间的 KL 散度，使得先验分布 P 接近后验分布 Q
    :param m_p: 均值 m_p，形状为 (N, D)
    :param logs_p: 方差的对数 log(σ_p^2)，形状为 (N, D)
    :param m_q: 均值 m_q，形状为 (N, D)
    :param logs_q: 方差的对数 log(σ_q^2)，形状为 (N, D)
    :return: KL 散度，形状为 (N,)
    """
    sigma_p = torch.exp(0.5 * logs_p)
    sigma_q = torch.exp(0.5 * logs_q)
    
    kl_div = torch.log(sigma_q / sigma_p) + (sigma_p**2 + (m_p - m_q)**2) / (2 * sigma_q**2) - 0.5
    return kl_div.sum(dim=-1).mean()  # 对每个样本的 KL 散度求均值，得到一个标量