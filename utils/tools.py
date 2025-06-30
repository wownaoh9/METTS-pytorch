import os
import json
import random
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt

matplotlib.use("Agg")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_device(data, device):
    if len(data) == 14:
        (   
            id,    #1 
            raw_text,

            srcs,
            src_len,
            max_src_len,

            speaker,
            emotion_emo,
            emotion_neu,
            
            mels,
            mel_len,
            max_mel_len,

            pitch,
            energy,
            duration, #14

            #using for emotion encoder

        ) = data

        srcs = torch.from_numpy(srcs).long().to(device)
        src_len = torch.from_numpy(src_len).to(device)

        speaker = torch.from_numpy(speaker).long().to(device)
        emotion_emo = torch.from_numpy(emotion_emo).long().to(device)
        emotion_neu = torch.from_numpy(emotion_neu).long().to(device)

        mels = torch.from_numpy(mels).float().to(device)
        mel_len = torch.from_numpy(mel_len).int().to(device)
        max_mel_len = torch.tensor(max_mel_len).to(device)

        pitch = torch.from_numpy(pitch).float().to(device)
        energy = torch.from_numpy(energy).float().to(device)
        duration = torch.from_numpy(duration).int().to(device)

        return (
            id,
            raw_text,

            srcs,
            src_len,
            max_src_len,

            speaker,
            emotion_emo,
            emotion_neu,
            
            mels,
            mel_len,
            max_mel_len,

            pitch,
            energy,
            duration,

            #using for emotion encoder
        )
    
    #for inference
    if len(data) == 11:
        (   
            id,    #1 
            raw_text,

            srcs,
            src_len,
            max_src_len,

            speaker,
            emotion_emo,
            emotion_neu,

            mels,
            mel_len,
            max_mel_len,

            #using for emotion encoder
        ) = data

        srcs = torch.from_numpy(srcs).long().to(device)
        src_len = torch.from_numpy(src_len).to(device)

        speaker = torch.from_numpy(speaker).long().to(device)
        emotion_emo = torch.from_numpy(emotion_emo).long().to(device)
        emotion_neu = torch.from_numpy(emotion_neu).long().to(device)

        mels = torch.from_numpy(mels).float().to(device)
        mel_len = torch.from_numpy(mel_len).int().to(device)
        max_mel_len = torch.tensor(max_mel_len).to(device)

        return (
            id,
            raw_text,

            srcs,
            src_len,
            max_src_len,

            speaker,
            emotion_emo,
            emotion_neu,
            
            mels ,
            mel_len,
            max_mel_len,

            #using for emotion encoder
        )

def log_time_records(time_records, step, log_path):
    # 将时间数据保存到日志文件
    log_file_name = f"time_records_step_{step}.json"
    log_file_path = os.path.join(log_path, log_file_name)
    with open(log_file_path, "w") as log_file:
        json.dump(time_records, log_file, indent=4)
        
def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("1_Loss/total_loss", losses[0], step)
        logger.add_scalar("2_Loss/mel_loss_5", losses[1], step)
        logger.add_scalar("3_Loss/mel_loss_6", losses[2], step)
        logger.add_scalar("4_Loss/ssim_loss", losses[3], step)
        logger.add_scalar("5_Loss/pitch_loss", losses[4], step)
        logger.add_scalar("6_Loss/energy_loss", losses[5], step)
        logger.add_scalar("7_Loss/duration_loss", losses[6], step)
        logger.add_scalar("8_Loss/ce_loss", losses[7], step)
        logger.add_scalar("9_Loss/kl_loss", losses[8], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )

def get_mask_from_lengths(lengths, max_len=None, device=device):
    batch_size = lengths.shape[0]
    device = lengths.device
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask
def get_mask_from_lengths_tensor(lengths, max_len=None, device = None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

def get_mask_from_lengths_np(lengths, max_len=None):
    batch_size = lengths.shape[0]
    
    # 如果没有提供 max_len，使用 lengths 中的最大值
    if max_len is None:
        max_len = np.max(lengths)
    
    # 创建一个从 0 到 max_len 的索引数组
    ids = np.arange(max_len)
    
    # 扩展索引数组以匹配批次大小
    ids = np.tile(ids, (batch_size, 1))
    
    # 创建掩码，掩码中的每个元素 mask[i, j] 为 True 如果 j < lengths[i]
    mask = ids >= lengths[:, np.newaxis]
    
    return mask

def Concate_tensor(mel,pitches,energies):
    
    # 扩展 pitches 和 energies 的维度以匹配 mel
    pitches = pitches.unsqueeze(-1)
    energies = energies.unsqueeze(-1)

    # 沿着第三维度拼接 mel, pitches 和 energies
    X = torch.cat((mel, pitches, energies), dim=2)

    return X

def Concate_mel_pitch_energy(mel,pitches,energies, durations):
    #音素级到帧级的上采样
    pitches = expand_batch (pitches, durations)
    energies = expand_batch (energies, durations)
    
    # 扩展 pitches 和 energies 的维度以匹配 mel
    pitches = pitches.unsqueeze(-1).to(mel.device)
    energies = energies.unsqueeze(-1).to(mel.device)

    # 沿着第三维度拼接 mel, pitches 和 energies
    X = torch.cat((mel, pitches, energies), dim=2)

    return X

def Concate_np(mel, pitches, energies):

    pitches = np.expand_dims(pitches, axis=-1)
    energies = np.expand_dims(energies, axis=-1)

    # 沿着第三维度拼接 mel, pitches 和 energies
    X = np.concatenate((mel, pitches, energies), axis=2)

    return X

def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)

def expand_batch(tensorA, tensorB):
    # 获取批次大小和序列长度
    batch_size, seq_len = tensorA.size(0), tensorA.size(1)
    
    # 生成索引
    indices = torch.arange(seq_len).repeat(batch_size, 1) + torch.arange(batch_size).unsqueeze(1) * seq_len
    
    # 扩展索引
    expanded_indices = indices.unsqueeze(-1).repeat(1, 1, tensorB.max() + 1)
    
    # 根据扩展大小复制元素
    expanded_elements = tensorA.view(-1)[expanded_indices.view(-1)].view(batch_size, -1)
    
    return expanded_elements



def synth_infer_sample( predictions, vocoder, model_config, preprocess_config):

    src_len = predictions[8][0].item()

    mel_len_predic = predictions[9][0].item()
    mel_prediction = predictions[1][0, :mel_len_predic].detach().transpose(0, 1)

    duration_prediction = predictions[5][0, :src_len].detach().cpu().numpy()

    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch_prediction = predictions[2][0, :src_len].detach().cpu().numpy()
        pitch_prediction = expand(pitch_prediction, duration_prediction)

    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy_prediction = predictions[3][0, :src_len].detach().cpu().numpy()
        energy_prediction = expand(energy_prediction, duration_prediction)

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch_prediction, energy_prediction),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from.getModel_utils import vocoder_infer

        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_prediction


def synth_one_sample(batch, predictions, vocoder, model_config, preprocess_config):
    basename = batch[0][0]

    mels_emo = batch[8][0],
    mel_len_target = batch[9][0]
    pitches_emo = batch[11][0],
    energies_emo = batch[12][0],
    durations_emo = batch[13][0],

    mels_emo = mels_emo[0]
    pitches_emo = pitches_emo[0]
    energies_emo = energies_emo[0]
    durations_emo = durations_emo[0]
    
    basename = basename
    src_len = predictions[8][0].item()

    mel_len_predic = predictions[9][0].item()
    mel_target = mels_emo[:mel_len_target, ].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len_predic].detach().transpose(0, 1)

    duration_target = durations_emo[:src_len,].detach().cpu().numpy()
    duration_prediction = predictions[5][0, :src_len].detach().cpu().numpy()

    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch_target = pitches_emo[:src_len,].detach().cpu().numpy()
        pitch_target = expand(pitch_target, duration_target)

        pitch_prediction = predictions[2][0, :src_len].detach().cpu().numpy()
        pitch_prediction = expand(pitch_prediction, duration_prediction)

    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy_target = energies_emo[:src_len,].detach().cpu().numpy()
        energy_target = expand(energy_target, duration_target)

        energy_prediction = predictions[3][0, :src_len].detach().cpu().numpy()
        energy_prediction = expand(energy_prediction, duration_prediction)

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch_prediction, energy_prediction),
            (mel_target.cpu().numpy(), pitch_target, energy_target),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .getModel_utils import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename



def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        testS =datetime.now().strftime("%m-%d--%H-%M-%S")
        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(testS)))
        plt.close()

    from.getModel_utils import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    # 使用 NumPy 的 savetxt 函数保存数组，指定逗号作为分隔符
    #np.savetxt('large_array.txt', wav_predictions, fmt='%i', delimiter=',')
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(testS)), sampling_rate, wav)

def generate_durations(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        testS =datetime.now().strftime("%m-%d--%H-%M-%S")
        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(testS)))
        plt.close()

    from.getModel_utils import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    # 使用 NumPy 的 savetxt 函数保存数组，指定逗号作为分隔符
    #np.savetxt('large_array.txt', wav_predictions, fmt='%i', delimiter=',')
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(testS)), sampling_rate, wav)

def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):                              #range(len(data))生成一个从0到len(data) - 1的整数序列.只有一句话时只循环一次
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean

        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])                     #设置子图的y轴范围从0到梅尔频谱图的高度，即频率的维度。x会自动适配
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])               #设置ax1轴的x轴范围。mel.shape[1]是梅尔频谱图的宽度，即时间轴的维度。这确保音高曲线的x轴范围与梅尔频谱图的时间轴对齐。
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D_text(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_1D_p_e(inputs, max_len = None, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    if max_len is None:
        max_len = max([len(x) for x in inputs])
    else :
        max_len = max_len
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D_mel(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

import yaml

def get_configs_of(dataset):
    config_dir = os.path.join("./config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config

def sample_lambda(exclude_values, alpha=2.0, beta=2.0, size=None):
    # 如果没有指定 size，返回单个值
    if size is None:
        return np.random.beta(alpha, beta) if np.random.beta(alpha, beta) not in exclude_values else sample_lambda(exclude_values, alpha, beta, size)

    # 如果指定了 size，生成数组
    lambda_values = np.zeros(size)
    for i in np.ndindex(size):
        lambda_values[i] = np.random.beta(alpha, beta) if np.random.beta(alpha, beta) not in exclude_values else sample_lambda(exclude_values, alpha, beta, size=1)[0]
    return lambda_values

def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn"t know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx