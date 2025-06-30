# coding=gb2312
import argparse
import os

import librosa
import numpy as np
import yaml
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text

def prepare_align():
    in_dir = "/home/wl/aaa_code/METTS-delight-end/preprocessed_data/esd_lj_biaobei_corpus/wl/dataset/ESD-en"
    out_dir = "raw_data"
    sampling_rate = 22050
    max_wav_value = 32768.0
    cleaners = ["english_cleaners"]

    for speaker in ["0011","0012","0013","0014","0015","0016","0017","0018","0019","0020"]:
        print("Processing {} set...".format(speaker))
        for emotion in ["Angry","Happy","Neutral","Sad","Surprise"]:
            print("Processing {}-{} set...".format(speaker,emotion))
            if speaker in ["0011", "0015", "0020"]:
                with open(os.path.join(in_dir,speaker, "{}.txt".format(speaker)), encoding="GBK") as f:
                    for line in tqdm(f):
                        parts = line.strip().split("\t")  
                        if len(parts) == 3 and parts[2] == emotion:
                            wav_name=parts[0]
                            text=parts[1]
                            text = _clean_text(text, cleaners)
                            wav_path = os.path.join(in_dir, speaker,emotion, "{}.wav".format(wav_name))
                            if os.path.exists(wav_path):
                                os.makedirs(os.path.join(out_dir, f"{speaker}_{emotion}"), exist_ok=True)
                                wav, _ = librosa.load(wav_path, sampling_rate)
                                wav = wav / max(abs(wav)) * max_wav_value
                                wavfile.write(
                                    os.path.join(os.path.join(out_dir, f"{speaker}_{emotion}"), "{}.wav".format(wav_name)),
                                    sampling_rate,
                                    wav.astype(np.int16),
                                )
                                with open(
                                    os.path.join(os.path.join(out_dir, f"{speaker}_{emotion}"), "{}.lab".format(wav_name[:11])),
                                    "w",encoding="UTF-8"
                                ) as f1:
                                    f1.write(text)
            else:
                with open(os.path.join(in_dir,speaker, "{}.txt".format(speaker)), encoding="UTF-8") as f:
                    for line in tqdm(f):
                        parts = line.strip().split("\t")  
                        if len(parts) == 3 and parts[2] == emotion:
                            wav_name=parts[0]
                            text=parts[1]
                            text = _clean_text(text, cleaners)
                            wav_path = os.path.join(in_dir, speaker,emotion, "{}.wav".format(wav_name))
                            if os.path.exists(wav_path):
                                os.makedirs(os.path.join(out_dir, f"{speaker}_{emotion}"), exist_ok=True)
                                wav, _ = librosa.load(wav_path, sampling_rate)
                                wav = wav / max(abs(wav)) * max_wav_value
                                wavfile.write(
                                    os.path.join(os.path.join(out_dir, f"{speaker}_{emotion}"), "{}.wav".format(wav_name)),
                                    sampling_rate,
                                    wav.astype(np.int16),
                                )
                                with open(
                                    os.path.join(os.path.join(out_dir, f"{speaker}_{emotion}"), "{}.lab".format(wav_name[:11])),
                                    "w",encoding="UTF-8"
                                ) as f1:
                                    f1.write(text)

if __name__ == "__main__":
    prepare_align()
