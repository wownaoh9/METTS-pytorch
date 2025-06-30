import json
import math
import os
from functools import lru_cache
import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D_text,pad_1D_p_e, pad_2D_mel
from utils.tools import get_mask_from_lengths_np,Concate_np,sample_lambda

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.emotion, self.text_seq, self.text, self.raw_text = self.process_meta(
            filename
        )
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            self.emotion_map = json.load(f)
        self.zh_text_cleaners = preprocess_config["preprocessing"]["text"]["zh_text_cleaners"]
        self.en_text_cleaners = preprocess_config["preprocessing"]["text"]["en_text_cleaners"]
        self.zh_speakers = {"0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "biaobei"}
        self.en_speakers = {"0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020", "ljspeech"}
        self.sort = sort
        self.drop_last = drop_last

        # 缓存已加载的特征
        self.cached_spk_emb = {}
        self.cached_fs2data = {}

    def __len__(self):
        return len(self.text)

    # def _load_spk_emb(self, speaker, emotion, basename):
    #     """缓存加载speaker的embeddings"""
    #     if (speaker, emotion, basename) in self.cached_spk_emb:
    #         return self.cached_spk_emb[(speaker, emotion, basename)]
         
    #     if emotion is None:
    #         spk_emb_path = os.path.join(
    #             self.preprocessed_path,
    #             "spk_emb",
    #             speaker,
    #             "{}.wav.npy".format(basename),
    #         )
    #     else:
    #         spk_emb_path = os.path.join(
    #             self.preprocessed_path,
    #             "spk_emb",
    #             speaker,
    #             emotion,
    #             "{}.wav.npy".format(basename),
    #         )
    #     spk_emb = np.load(spk_emb_path)
    #     self.cached_spk_emb[(speaker, emotion, basename)] = spk_emb
    #     return spk_emb

    @lru_cache(maxsize=20000)  # 同样设置缓存大小
    def _load_fs2data(self, speaker, emotion, data, basename):
        fs2data_path = os.path.join(
            self.preprocessed_path,
            data,
            "{}-{}-{}-{}.npy".format(speaker, emotion, data, basename)
        )
        return np.load(fs2data_path)
    
    # def _find_existing_fs2data(self, data, speaker, emotion):
    #     """在data目录下查找一个存在的文件"""
    #     data_dir = os.path.join(self.preprocessed_path, data, speaker, emotion)
    #     for file_name in os.listdir(data_dir):
    #         # 检查文件是否符合命名规则
    #         if file_name.startswith(speaker) :
    #             # 找到文件后返回
    #             print(f"Warning: Specified file not found, returning existing file: {file_name}")
    #             return np.load(os.path.join(data_dir, file_name))
    #     raise FileNotFoundError(f"No valid file found for speaker {speaker} in {data}-{speaker}-{emotion} directory.")

    def _getitem_(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        emotion = self.emotion[idx]
        emotion_emo_id = self.emotion_map[emotion]
        emotion_neu = "Neutral"
        emotion_neu_id = self.emotion_map[emotion_neu]
        raw_text = self.raw_text[idx]
        srcs = np.array(self.text_seq[idx])

        # # 加载 speaker embedding
        # if speaker in self.speaker_set:
        #     spk_emb = self._load_spk_emb(speaker, emotion, basename)
        # else:
        #     spk_emb = self._load_spk_emb(speaker, None, basename)

        # 加载 mel, pitch, energy, duration 数据
        mels = self._load_fs2data(speaker, emotion, "mel", basename)
        pitch = self._load_fs2data(speaker, emotion, "pitch", basename)
        energy = self._load_fs2data(speaker, emotion, "energy", basename)
        duration = self._load_fs2data(speaker, emotion, "duration", basename)

        sample = {
            "id": basename,
            "raw_text": raw_text,
            "srcs": srcs,

            "speaker": speaker_id,
            "emotion_emo": emotion_emo_id,
            "emotion_neu": emotion_neu_id,

            "mels": mels,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,

            # using for emotion encoder
            # spk
            # "spk_emb": spk_emb,
        }
        return sample
    
    def __getitem__(self, idx):
        sample = self._getitem_(idx)
        
        return sample

    #get items from "train.txt" or "val.txt"
    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            emotion = []
            text_seq = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, e, t_s, t, r = line.strip("\n").split("|")    
                if s not in ["biaobei", "ljspeech"]:                 
                    name.append(n)
                    speaker.append(s)
                    emotion.append(e)
                    text_seq.append([int(item) for item in t_s.split()])
                    text.append(t)
                    raw_text.append(r)                  
            return name, speaker, emotion,  text_seq, text, raw_text

    #after getitems
    def reprocess(self, data, idxs):
        id = [data[idx]["id"] for idx in idxs]
        raw_text = [data[idx]["raw_text"] for idx in idxs]

        srcs = [data[idx]["srcs"] for idx in idxs]
        src_len = np.array([src.shape[0] for src in srcs])
        max_src_len = max(src_len)

        speaker = np.array([data[idx]["speaker"] for idx in idxs])
        emotion_emo = np.array([data[idx]["emotion_emo"] for idx in idxs])
        emotion_neu = np.array([data[idx]["emotion_neu"] for idx in idxs])
        
        mels = [data[idx]["mels"] for idx in idxs]
        mel_len = np.array([mel.shape[0] for mel in mels])
        max_mel_len = max(mel_len)

        pitch = [data[idx]["pitch"] for idx in idxs]
        energy = [data[idx]["energy"] for idx in idxs]
        duration = [data[idx]["duration"] for idx in idxs]

        #using for emotion encoder

        #spk 
        # spk_emb = np.array([data[idx]["spk_emb"] for idx in idxs])

        #做pad
        srcs = pad_1D_text(srcs)

        mels = pad_2D_mel(mels)
        pitch = pad_1D_p_e(pitch)
        energy = pad_1D_p_e(energy)
        duration = pad_1D_p_e(duration)

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

            #spk
            # spk_emb,
        )

    def collate_fn(self, data):   #自动获得batchsize个items
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["srcs"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()   #将剩余部分之外的索引按照 batch_size 分组成多个子列表，并转换为列表的形式。
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output



if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.tools import to_device_mix

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    preprocess_config = yaml.load(
        open("./config/ESD_neuDOE/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/ESD_neuDOE/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = DatasetFsWithMixFor_ESD_neuDOE(
        "ESD_neuDOE.txt", preprocess_config, train_config, sort=True, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        prefetch_factor=8,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,  # 持久化 worker
    )


    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device_mix(batch, device)
            n_batch += 1
            print(n_batch)
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )