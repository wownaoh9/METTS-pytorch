import argparse
import os
import numpy as np
import json
import matplotlib.pyplot as plt

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy
import logging

from utils.getModel_utils import get_model_fs, get_vocoder
from utils.tools import to_device, synth_one_sample, synth_infer_sample
from utils.text_utils import preprocess_english,preprocess_mandarin
from text import text_to_sequence


# 全局缓存
cached_data = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _load_fs2data(preprocessed_path, data_name, speaker, emotion, basename):
    """缓存加载数据"""
    if (data_name, speaker, emotion, basename) in cached_data:
        return cached_data[(data_name, speaker, emotion, basename)]

    data_name_path = os.path.join(
        preprocessed_path,
        data_name,
        "{}-{}-{}-{}_{}.npy".format(speaker, emotion, data_name, speaker, basename)
    )

    data = np.load(data_name_path)

    # 缓存和返回数据
    cached_data[(speaker, emotion, data_name, basename)] = data
    return data

def get_ref_emotion(preprocessed_path, refer_speaker, refer_emotion, refer_basename):
    mels = _load_fs2data(preprocessed_path, "mel", refer_speaker, refer_emotion, refer_basename)

    mel_len = np.array([mels.shape[0]])
    max_mel_len = max(mel_len)

    return mels, mel_len, max_mel_len

def main(args, configs):    #args是命令行输入，configs是配置文件
    print("Preparing for training...")

    #get device
    device = torch.device(args.device)
    print("Using Device:", device)

    #get step
    step = args.restore_step

    model= get_model_fs(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    #get infer emotion audio
    emotion_list = ["Neutral","Angry","Happy","Sad","Surprise"]
    basename_list = ["000326", "000676", "001026", "001376", "001726"]
    refer_speaker_list = ["0003", "0004", "0007", "0008", ]
    infer_speaker_list = ["0003", "0004", "0007", "0008", "001"]
    for infer_speaker_str in infer_speaker_list:
        for refer_speaker_str in refer_speaker_list:
            for (emotion_str, basename_str) in zip(emotion_list, basename_list):
                basename_str = f"{refer_speaker_str}_{basename_str}" 
                inference_zh(model, step = step, configs = configs,
                            infer_speaker_str = infer_speaker_str,
                            refer_speaker_str = refer_speaker_str, emotion_str = emotion_str, basename_str = basename_str, 
                            logger=None, vocoder=vocoder, device=device)

def inference_zh(model, step, configs, vocoder, device,
                infer_speaker,
                refer_speaker, refer_emotion, refer_basename,
                ):

    preprocess_config, model_config, train_config = configs
  
    preprocessed_path = preprocess_config["path"]["preprocessed_path"]
    with open(os.path.join(preprocessed_path, "speakers.json")) as f:
        speaker_map = json.load(f)
    with open(os.path.join(preprocessed_path, "emotions.json")) as f:
        emotion_map = json.load(f)
    
    # get text
    raw_text = "不久以后，王后果然生下了一个可爱的小公主。"
    id = raw_text[:100]
    phones = preprocess_mandarin(raw_text, preprocess_config)
    srcs = np.array(text_to_sequence(phones, preprocess_config["preprocessing"]["text"]["zh_text_cleaners"]))
    src_len = np.array([len(srcs)])

    #get ref emo wav
    mels, mel_len, max_mel_len = get_ref_emotion(preprocessed_path, refer_speaker, refer_emotion, refer_basename)
    
    #get infer speaker and emotion
    speaker = np.array([speaker_map[infer_speaker]])
    emotion_emo = np.array([emotion_map[refer_emotion]])
    emotion_neu = np.array([0])

    batchs = [(id, raw_text, 
                srcs, src_len, max(src_len), 
                speaker, emotion_emo, emotion_neu,
                mels, mel_len, max_mel_len,  
               )]

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():

            output = model.infer(
                            srcs = batch[2].unsqueeze(0),
                            src_len = batch[3],
                            max_src_len = batch[4],

                            speaker = batch[5],
                            emotion_emo = batch[6],
                            emotion_neu = batch[7],
                            
                            mels = batch[8].unsqueeze(0),
                            mel_len = batch[9],
                            max_mel_len = batch[10],

                            pitch = None,
                            energy = None,
                            duration = None,

                            #using for emotion encoder
                            # MPE_emo = batch[8].unsqueeze(0),
                            # MPE_neu = batch[9].unsqueeze(0),
                            # MPE_len = batch[10],

                            p_control=1.0,
                            e_control=1.0,
                            d_control=1.0,

                            train = False
            ) 


    #syn
    fig, wav_prediction = synth_infer_sample(
        output,
        vocoder,
        model_config,
        preprocess_config,
    )

    tag = "infer" + infer_speaker + "-" + "refer" + refer_speaker + "-" + refer_emotion + "-" + str(refer_basename) + "-" + raw_text

    #save audio
    if infer_speaker in ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010"]:
        save_dir = os.path.join(train_config["path"]["result_path"], f"{step}", "infer-zh", "ESD-zh", infer_speaker, refer_speaker)
    elif infer_speaker in ["0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"]:
        save_dir = os.path.join(train_config["path"]["result_path"], f"{step}", "infer-zh", "ESD-en", infer_speaker, refer_speaker)
    elif infer_speaker in ["ljspeech", "biaobei"]:
        save_dir = os.path.join(train_config["path"]["result_path"], f"{step}", "infer-zh", "ljs_biaobei", infer_speaker, refer_speaker)
    os.makedirs(save_dir, exist_ok=True)
    audio = wav_prediction
    scipy.io.wavfile.write(
        filename=os.path.join(save_dir, f"{tag}.wav"),
        rate=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        data=audio,
    )
    # save fig
    filename = f"{tag}.png"  # 图像文件名
    fig.savefig(os.path.join(save_dir, filename))  # 保存图像
    plt.close(fig)  # 关闭图形对象，释放内存

def inference_en(model, step, configs, vocoder, device,
                infer_speaker,
                refer_speaker, refer_emotion, refer_basename,
                ):

    preprocess_config, model_config, train_config = configs
  
    preprocessed_path = preprocess_config["path"]["preprocessed_path"]
    with open(os.path.join(preprocessed_path, "speakers.json")) as f:
        speaker_map = json.load(f)
    with open(os.path.join(preprocessed_path, "emotions.json")) as f:
        emotion_map = json.load(f)
    
    # get text
    raw_text = "Not long after, the queen gave birth to a lovely little princess."
    id = raw_text[:100]
    phones = preprocess_english(raw_text, preprocess_config)
    srcs = np.array(text_to_sequence(phones, preprocess_config["preprocessing"]["text"]["en_text_cleaners"]))
    src_len = np.array([len(srcs)])

    #get ref emo wav
    mels, mel_len, max_mel_len = get_ref_emotion(preprocessed_path, refer_speaker, refer_emotion, refer_basename)
    
    #get infer speaker and emotion
    speaker = np.array([speaker_map[infer_speaker]])
    emotion_emo = np.array([emotion_map[refer_emotion]])
    emotion_neu = np.array([0])

    batchs = [(id, raw_text, 
                srcs, src_len, max(src_len), 
                speaker, emotion_emo, emotion_neu,
                mels, mel_len, max_mel_len,  
               )]

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():

            output = model.infer(
                            srcs = batch[2].unsqueeze(0),
                            src_len = batch[3],
                            max_src_len = batch[4],

                            speaker = batch[5],
                            emotion_emo = batch[6],
                            emotion_neu = batch[7],
                            
                            mels = batch[8].unsqueeze(0),
                            mel_len = batch[9],
                            max_mel_len = batch[10],

                            pitch = None,
                            energy = None,
                            duration = None,

                            #using for emotion encoder
                            # MPE_emo = batch[8].unsqueeze(0),
                            # MPE_neu = batch[9].unsqueeze(0),
                            # MPE_len = batch[10],

                            p_control=1.0,
                            e_control=1.0,
                            d_control=1.0,

                            train = False
            ) 


    #syn
    fig, wav_prediction = synth_infer_sample(
        output,
        vocoder,
        model_config,
        preprocess_config,
    )

    tag = "infer" + infer_speaker + "-" + "refer" + refer_speaker + "-" + refer_emotion + "-" + str(refer_basename) + "-" + raw_text

    #save audio
    if infer_speaker in ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010"]:
        save_dir = os.path.join(train_config["path"]["result_path"], f"{step}", "infer-en", "ESD-zh", infer_speaker, refer_speaker)
    elif infer_speaker in ["0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"]:
        save_dir = os.path.join(train_config["path"]["result_path"], f"{step}", "infer-en", "ESD-en", infer_speaker, refer_speaker)
    elif infer_speaker in ["ljspeech", "biaobei"]:
        save_dir = os.path.join(train_config["path"]["result_path"], f"{step}", "infer-en", "ljs_biaobei", infer_speaker, refer_speaker)
    os.makedirs(save_dir, exist_ok=True)
    audio = wav_prediction
    scipy.io.wavfile.write(
        filename=os.path.join(save_dir, f"{tag}.wav"),
        rate=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        data=audio,
    )
    # save fig
    filename = f"{tag}.png"  # 图像文件名
    fig.savefig(os.path.join(save_dir, filename))  # 保存图像
    plt.close(fig)  # 关闭图形对象，释放内存

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("-r","--restore_step", type=int, default=200000, help="path to **.tar")

    parser.add_argument(
        "-p","--preprocess_config",type=str,required=False,default="config/ESD_zh/preprocess.yaml",help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=False,default="config/ESD_zh/model.yaml", help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=False,default="config/ESD_zh/train.yaml", help="path to train.yaml"
    )

    args = parser.parse_args()

    # 确保 restore_step 属性存在
    if hasattr(args, 'restore_step'):
        print(f"Restore step: {args.restore_step}")
    else:
        print("Restore step not specified")

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)