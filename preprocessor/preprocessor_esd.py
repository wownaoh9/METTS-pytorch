import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio
from text import text_to_sequence

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.zh_text_cleaners = config["preprocessing"]["text"]["zh_text_cleaners"]
        self.en_text_cleaners = config["preprocessing"]["text"]["en_text_cleaners"]
        self.zh_speakers = {"0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010"}
        self.en_speakers = {"0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"}

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        emotions = {}
        for i, speaker_emotion in enumerate(tqdm(os.listdir(self.in_dir))):
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker_emotion)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                speaker = speaker_emotion.split("_")[0]
                emotion = speaker_emotion.split("_")[1]

                if speaker in self.en_speakers or speaker in self.zh_speakers:
                    # 将 speaker 和 emotion 添加到字典中
                    if speaker not in speakers:
                        speakers[speaker] = len(speakers)
                    if emotion not in emotions:
                        emotions[emotion] = len(emotions)
                        
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", speaker_emotion, "{}.TextGrid".format(basename)
                    )
                    if os.path.exists(tg_path):
                        ret = self.process_utterance(speaker_emotion, basename, speaker, emotion)    #对每一条音频处理
                        if ret is None:
                            continue
                        else:
                            info, pitch, energy, n = ret
                        out.append(info)

                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

                    n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))
        with open(os.path.join(self.out_dir, "emotions.json"), "w") as f:
            f.write(json.dumps(emotions))
          
        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write shuffled metadata
        with open(os.path.join(self.out_dir, "shuffled_train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "shuffled_val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker_emotion, basename, speaker, emotion):   #用于处理单个发音（utterance）的数据
        wav_path = os.path.join(self.in_dir, speaker_emotion, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker_emotion, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker_emotion, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"        #音素序列
        if start >= end:
            return None
        if speaker in self.zh_speakers:
            text_seq = " ".join([str(item) for item in text_to_sequence(text, self.zh_text_cleaners)])
        elif speaker in self.en_speakers:
            text_seq = " ".join([str(item) for item in text_to_sequence(text, self.en_text_cleaners)])

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)                   #长度变了，尾部裁剪

        # Read raw text
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency 计算音频信号的基本频率，通常也被称为音高
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)#去除音高轨迹中的噪声和假峰

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:       #检查非零元素的数量是否小于或等于1。如果非零元素的数量非常少，这可能表明音高检测结果不可靠，
            return None                   #或者音频数据中没有足够的有效音高信息。

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation   音素级别的线性插值的条件语句。这个步骤通常用于平滑音高轨迹，特别是在处理语音数据时，可以提高音高估计的连续性和可读性。
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]   #依据之前的duration对pitch做切片

        if self.energy_phoneme_averaging:
            # Phoneme-level average  音素级别的能量平均处理，它是基于每个音素的持续时间（duration）来计算能量（energy）的平均值
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-{}-duration-{}.npy".format(speaker, emotion, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-{}-pitch-{}.npy".format(speaker, emotion, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-{}-energy-{}.npy".format(speaker, emotion, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-{}-mel-{}.npy".format(speaker, emotion, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        #wav名称，LJSpeech,音素序列，文本序列，
        return (
            "|".join([basename, speaker, emotion, text_seq, text, raw_text]),
            self.remove_outlier(pitch),                         #移除异常值（outliers）
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],              #在梅尔频谱的上下文中，第一个维度通常表示时间帧（frames），而第二个维度表示频率bin
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            # Trim leading silences 只在前面执行
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
