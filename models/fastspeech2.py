import os
import json

import torch
import torch.nn as nn

from .conformer import TextEncoder, Decoder
from .modules import SpeakerEncoder, MelLinear, VarianceAdaptor, ResidualCouplingBlock, CoarseEmoExtractionModule, FineEmoExtractionModule, FineEmoPredictModule, PhoneLevelEmb, UtterLevelEmb
# from transformer import PostNet

from .commons import sequence_mask
from utils.tools import get_mask_from_lengths

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        # FS2 Init
        self.model_config = model_config
        self.text_encoder = TextEncoder(model_config)
        self.speaker_encoder = SpeakerEncoder(preprocess_config, model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = MelLinear(model_config["conformer"]["decoder_hidden"],preprocess_config["preprocessing"]["mel"]["n_mel_channels"])
        # self.postnet = PostNet()
        
        #emo init
        self.coarse_emo_extraction_module = CoarseEmoExtractionModule(preprocess_config, model_config)
        self.utter_level_emb = UtterLevelEmb(model_config["gst"]["E"], model_config["conformer"]["encoder_hidden"])
        self.fine_emo_extraction_module = FineEmoExtractionModule(preprocess_config, model_config)
        self.fine_emo_predict_module = FineEmoPredictModule(model_config)
        self.FG_proj= nn.Conv1d(model_config["FG_proj"]["out_channels"], model_config["FG_proj"]["out_channels"] * 2, 1)
        self.out_channels = model_config["FG_proj"]["out_channels"]
        self.PhoneLevelEmb = PhoneLevelEmb(model_config["FG_proj"]["out_channels"], model_config["conformer"]["encoder_hidden"])
        self.flow = ResidualCouplingBlock(model_config["flow"]["inter_channels"], model_config["flow"]["hidden_channels"], 5, 1, 4, gin_channels=model_config["flow"]["gin_channels"])

        
    def forward(            
            self,
            srcs, src_len, max_src_len,
            speaker, emotion_emo, emotion_neu,
            mels = None, mel_len = None, max_mel_len = None,
            pitch = None, energy = None, duration = None,

            #using for emotion encoder
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,

            train = True
        ):
        device = srcs.device
        src_mask = get_mask_from_lengths(src_len, max_src_len, device)
        mel_mask = (
            get_mask_from_lengths(mel_len, max_mel_len, device)
            if mel_len is not None
            else None
        )

        # text encoder
        output = self.text_encoder(srcs, src_mask)
        
        # add speaker embedding
        speaker_emb = self.speaker_encoder(speaker)
        output = output + speaker_emb.unsqueeze(1).repeat(1, output.shape[1], 1)

        # coarse emo extraction
        coarse_emo_emb, cls_utter = self.coarse_emo_extraction_module(mels, mel_len)
        coarse_emo_emb = self.utter_level_emb(cls_utter)
        output = output + coarse_emo_emb.unsqueeze(1).repeat(1, output.size(1), 1)

        #prior
        FG_fea = self.fine_emo_predict_module(output.transpose(1,2), src_mask).transpose(1, 2)
        x_mask = torch.unsqueeze(sequence_mask(src_len, FG_fea.size(2)), 1).to(FG_fea.dtype)
        # compute predict pho_level m, logs
        predict_stats = self.FG_proj(FG_fea) * x_mask
        m_p, logs_p = torch.split(predict_stats, self.out_channels, dim=1)
        predict_pho_Level_fea = (m_p + torch.randn_like(m_p) * torch.exp(logs_p)) * x_mask

        #posterior
        phone_emo_emb, cls_fine, cls_phoneme = self.fine_emo_extraction_module(mels, mel_mask, duration)
        # compute pho_level m, logs
        stats = self.FG_proj(phone_emo_emb.transpose(1,2)) * x_mask
        m_q, logs_q = torch.split(stats, self.out_channels, dim=1)
        pho_Level_fea = (m_q + torch.randn_like(m_q) * torch.exp(logs_q)) * x_mask

        # flow
        z_p = self.flow(pho_Level_fea, x_mask, g=None) #后验-》flow-》
        
        #add output and pho_Level_fea
        output = output + self.PhoneLevelEmb(z_p.transpose(1, 2))

        #variance adapter
        (   output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            mel_lens, mel_mask
        ) = self.variance_adaptor(
            output, 
            src_mask,
            mel_mask, max_mel_len,
            pitch, energy, duration,
            p_control, e_control, d_control,
            train = train
        )

        #six conformer decoder to compute mel_loss seperately
        dec_output_one, dec_output_two, dec_output_three, dec_output_four, dec_output_five, dec_output_six, mel_mask = self.decoder(output, mel_mask)

        output_mellinear_one = self.mel_linear(dec_output_one)
        output_mellinear_two = self.mel_linear(dec_output_two)
        output_mellinear_three = self.mel_linear(dec_output_three)
        output_mellinear_four = self.mel_linear(dec_output_four)
        output_mellinear_five = self.mel_linear(dec_output_five)
        output_mellinear_six = self.mel_linear(dec_output_six)

        # postnet_output = self.postnet(output) + output
        output_ssim = output_mellinear_six

        return (
            (output_mellinear_one, output_mellinear_two, output_mellinear_three, output_mellinear_four, output_mellinear_five, output_mellinear_six),
            output_ssim,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            src_mask, mel_mask, src_len, mel_lens,       #9

            emotion_emo,
            (cls_utter, cls_fine, cls_phoneme),
            (z_p, logs_q, m_p, logs_p, x_mask) # posterior ->flow-> prior
        )

    def infer(            
            self,
            srcs, src_len, max_src_len,
            speaker, emotion_emo, emotion_neu,
            mels = None, mel_len = None, max_mel_len = None,
            pitch = None, energy = None, duration = None,

            #using for emotion encoder
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,

            train = True
        ):
        device = srcs.device
        src_mask = get_mask_from_lengths(src_len, max_src_len, device)
        mel_mask = (
            get_mask_from_lengths(mel_len, max_mel_len, device)
            if mel_len is not None
            else None
        )

        # text encoder
        output = self.text_encoder(srcs, src_mask)
        
        # add speaker embedding
        speaker_emb = self.speaker_encoder(speaker)
        output = output + speaker_emb.unsqueeze(1).repeat(1, output.shape[1], 1)

        # coarse emo extraction
        coarse_emo_emb, cls_utter = self.coarse_emo_extraction_module(mels, mel_len)
        coarse_emo_emb = self.utter_level_emb(cls_utter)
        output = output + coarse_emo_emb.unsqueeze(1).repeat(1, output.size(1), 1)

        #prior
        FG_fea = self.fine_emo_predict_module(output.transpose(1,2), src_mask).transpose(1, 2)
        x_mask = torch.unsqueeze(sequence_mask(src_len, FG_fea.size(2)), 1).to(FG_fea.dtype)
        # compute predict pho_level m, logs
        predict_stats = self.FG_proj(FG_fea) * x_mask
        m_p, logs_p = torch.split(predict_stats, self.out_channels, dim=1)
        predict_pho_Level_fea = (m_p + torch.randn_like(m_p) * torch.exp(logs_p)) * x_mask

        # #posterior
        # phone_emo_emb, cls_fine, cls_phoneme = self.fine_emo_extraction_module(mels, mel_mask, duration)
        # # compute pho_level m, logs
        # stats = self.FG_proj(phone_emo_emb.transpose(1,2)) * x_mask
        # m_q, logs_q = torch.split(stats, self.out_channels, dim=1)
        # pho_Level_fea = (m_q + torch.randn_like(m_q) * torch.exp(logs_q)) * x_mask

        # flow
        z_p = self.flow(predict_pho_Level_fea, x_mask, g=None, reverse=True) #后验-》flow-》
        
        #add output and pho_Level_fea
        output = output + self.PhoneLevelEmb(z_p.transpose(1, 2))

        #variance adapter
        mels = None
        mel_len = None
        max_mel_len = None
        mel_mask = (
            get_mask_from_lengths(mel_len, max_mel_len, device)
            if mel_len is not None
            else None
        )
        (   output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            mel_lens, mel_mask
        ) = self.variance_adaptor(
            output, 
            src_mask,
            mel_mask, max_mel_len,
            pitch, energy, duration,
            p_control, e_control, d_control,
            train = train
        )

        #six conformer decoder to compute mel_loss seperately
        dec_output_one, dec_output_two, dec_output_three, dec_output_four, dec_output_five, dec_output_six, mel_mask = self.decoder(output, mel_mask)

        output_mellinear_one = self.mel_linear(dec_output_one)
        output_mellinear_two = self.mel_linear(dec_output_two)
        output_mellinear_three = self.mel_linear(dec_output_three)
        output_mellinear_four = self.mel_linear(dec_output_four)
        output_mellinear_five = self.mel_linear(dec_output_five)
        output_mellinear_six = self.mel_linear(dec_output_six)

        # postnet_output = self.postnet(output) + output
        output_ssim = output_mellinear_six

        return (
            (output_mellinear_one, output_mellinear_two, output_mellinear_three, output_mellinear_four, output_mellinear_five, output_mellinear_six),
            output_ssim,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            src_mask, mel_mask, src_len, mel_lens,       #9

            emotion_emo,
            # (cls_utter, cls_fine, cls_phoneme),
            # (z_p, logs_q, m_p, logs_p, x_mask) # posterior ->flow-> prior
        )
