import os
import json

import torch
import torch.nn as nn
import numpy as np

from models.optimizer import ScheduledOptim

import hifigan
from models.fastspeech2 import FastSpeech2
from models.optimizer import ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = Finegrained(preprocess_config, model_config).to(device)

    #如果提供了恢复步骤(args.restore_step)，则加载相应的检查点(checkpoint)。
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_model_fs(args, configs, device, strict = False, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    
    #if args.restore_step!=0, loading checkpoint
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        # 加载checkpoint
        checkpoint = torch.load(ckpt_path,  map_location=device)
        print(checkpoint.keys())
        # 获取模型当前的状态字典
        current_state_dict = model.state_dict()

        # 获取checkpoint的状态字典
        checkpoint_state_dict = checkpoint['model']

        # 创建一个新的状态字典，只包含形状匹配的参数
        new_state_dict = {}

        for name, param in current_state_dict.items():
            if name in checkpoint_state_dict:
                # 检查形状是否匹配
                if param.size() == checkpoint_state_dict[name].size():
                    new_state_dict[name] = checkpoint_state_dict[name]
                else:
                    print(f"Skipping loading parameter {name} due to size mismatch.")
            else:
                print(f"Parameter {name} not found in checkpoint.")

        # 加载新的状态字典到模型
        model.load_state_dict(new_state_dict, strict=False)

        # 恢复 epoch 和 step
        if "epoch" in checkpoint and "step" in checkpoint:
            epoch = checkpoint["epoch"] + 1
            step = checkpoint["step"] + 1
            print(f"Resuming training from epoch {epoch}, step {step}")
        else:
            # 如果 epoch 和 step 不存在
            epoch = 1
            step = args.restore_step + 1
            print(f"No epoch and step in checkpoint. Starting from epoch {epoch}, step {step}")

        #
        if args.restore_step == 1:
            epoch = 1
            step = args.restore_step + 1
            print(f"New traing continuing from other ckpt. Starting from epoch {epoch}, step {step}")    

    else:
        epoch = 1
        step = args.restore_step + 1
        print(f"New training. Starting from epoch {epoch}, step {step}")
  
    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        # if args.restore_step and strict = True:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim, epoch, step

    model.eval()
    model.requires_grad_ = False
    return model

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)       #json.load表示转换成字典
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar",map_location=device)
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar",map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder

def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs