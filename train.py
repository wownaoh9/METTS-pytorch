import argparse
import os
import json

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.getModel_utils import get_model_fs, get_param_num, get_vocoder
from utils.tools import to_device, log, log_time_records, synth_one_sample
from models import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate
from inference import inference_zh, inference_en
import scipy.io.wavfile
from torch.nn import DataParallel


import time
from datetime import datetime


def main(args, configs):    # args are command line arguments, configs are configuration files

    #get device
    device = torch.device(args.device)
    print("Using Device:", device)

    preprocess_config, model_config, train_config= configs

    # Get dataset
    train_dataset = Dataset(
        "shuffled_train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4   # Set a group_size greater than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        prefetch_factor=8,
        pin_memory=True, 
        num_workers=8,
        persistent_workers=True,  # 持久化 worker
    )

    # Prepare model and optimizer
    model, optimizer, epoch, step = get_model_fs(args, configs, device, train=True)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of parameters in FastSpeech2 model:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Initialize loggers
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    time_log_path = os.path.join(train_config["path"]["log_path"], "time")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    os.makedirs(time_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training loop
    # step = args.restore_step + 1
    # epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    eval_step = train_config["step"]["eval_step"]
    synth_step = train_config["step"]["synth_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    # 初始化用于存储时间数据的字典
    time_records = {
        "total_data_load_time": [],  # 添加数据加载总时间记录
        "total_forward_time": [],    # 添加前向传播总时间记录
        "total_backward_time": [],   # 添加反向传播总时间记录
        "data_load_time": [],
        "forward_time": [],
        "backward_time": []
    }

    while True:
        epoch_start_time = time.time()
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Log epoch start time
        epoch_start_message = f"Epoch {epoch}:{formatted_now}"
        outer_bar.write(epoch_start_message)
        
        # Write to log file
        with open(os.path.join(train_log_path, "log.txt"), "a") as log_file:
            log_file.write(epoch_start_message + "\n")

        inner_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", position=1)

        for batchs in train_loader:
            # 记录读取数据时间
            data_load_start_time = time.time()
            
            for batch in batchs:
                # 数据加载完成，记录时间
                data_load_end_time = time.time()
                data_load_time = data_load_end_time - data_load_start_time
                time_records["data_load_time"].append(data_load_time)

                batch = to_device(batch, device)

                # Forward pass
                forward_start_time = time.time()
                output = model(*(batch[2:]))
                forward_end_time = time.time()
                forward_time = forward_end_time - forward_start_time
                time_records["forward_time"].append(forward_time)

                # Compute loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward pass
                backward_start_time = time.time()
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()
                backward_end_time = time.time()
                backward_time = backward_end_time - backward_start_time
                time_records["backward_time"].append(backward_time)
                # 重置数据加载开始时间，为加载下一批数据准备
                data_load_start_time = time.time()

                #log txt and tensorboard
                if step == 2 or step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Train Total Loss: {:.4f}, Mel Loss 5: {:.4f}, Mel Loss 6: {:.4f}, Ssim_Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f},\n"\
                        "Ce Loss: {:.4f},Kl Loss: {:.4f}".format(
                        *losses
                    )

                    #写日志 
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + "\n" + message2 + "\n")

                    outer_bar.write(message1 + message2)
                    
                    #写tensorboard
                    log(train_logger, step, losses=losses)

                if step % ( log_step * 10 ) == 0:
                    # 当需要记录日志时，将时间数据保存到文件
                    # 计算总时间并记录
                    total_data_load_time = sum(time_records["data_load_time"][-100:])
                    total_forward_time = sum(time_records["forward_time"][-100:])
                    total_backward_time = sum(time_records["backward_time"][-100:])
                    time_records["total_data_load_time"].append(total_data_load_time)
                    time_records["total_forward_time"].append(total_forward_time)
                    time_records["total_backward_time"].append(total_backward_time)

                    log_time_records(time_records, step, time_log_path)
                    # 清空时间数据字典，为下一次记录准备
                    time_records = {
                        "total_data_load_time": [],  # 添加数据加载总时间记录
                        "total_forward_time": [],    # 添加前向传播总时间记录
                        "total_backward_time": [],   # 添加反向传播总时间记录
                        "data_load_time": [],
                        "forward_time": [],
                        "backward_time": []
                    }

                if  step % save_step == 0:

                    #save model
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                            "epoch": epoch,
                            "step": step,
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )
                    outer_bar.write(f"Save model at step: {step}")

                if  step % eval_step == 0:
                    #syn sample
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                    )
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )           
                    #save audio
                    save_dir = os.path.join(train_config["path"]["result_path"], f"{step}", "train")
                    os.makedirs(save_dir, exist_ok=True)
                    audio = wav_prediction
                    scipy.io.wavfile.write(
                        filename=os.path.join(save_dir, f"{step}trian_pre.wav"),
                        rate=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                        data=audio,
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    audio = wav_reconstruction
                    scipy.io.wavfile.write(
                        filename=os.path.join(save_dir, f"{step}trian_gt.wav"),
                        rate=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                        data=audio,
                    )
                    # save fig
                    filename = f"{step}trian.png"  # 图像文件名
                    fig.savefig(os.path.join(save_dir, filename))  # 保存图像
                    plt.close(fig)  # 关闭图形对象，释放内存
                    outer_bar.write(f"Save train example at step: {step}")
            
                    #evaluate
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder, device)
                    outer_bar.write(f"Save val example at step: {step}")
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                    model.train()
                    
                    # infernece
                
                if step % synth_step == 0:
                    model.eval()
                    infer_speaker_list = ["0003", "0004", "0007", "0008", "0013", "0014", "0017", "0018"]
                    emotion_list = ["Neutral","Angry","Happy","Sad","Surprise"]
                    basename_list = ["000326", "000676", "001026", "001376", "001726"]
                    refer_speaker_list = ["0007","0003", "0017","0013"]
                    for infer_speaker in infer_speaker_list:
                        for refer_speaker in refer_speaker_list:
                            for (refer_emotion, refer_basename) in zip(emotion_list, basename_list):
                                inference_zh(model, step, configs, vocoder, device,
                                            infer_speaker,
                                            refer_speaker, refer_emotion, refer_basename,
                                            )
                                inference_en(model, step, configs, vocoder, device,
                                            infer_speaker,
                                            refer_speaker, refer_emotion, refer_basename,
                                            )
                    model.train()

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("-r","--restore_step", type=int, default=378000, help="path to **.tar")

    parser.add_argument(
        "-p","--preprocess_config",type=str,required=False,default="config/cl_esd/preprocess.yaml",help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=False,default="config/cl_esd/model.yaml", help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=False,default="config/cl_esd/train.yaml", help="path to train.yaml"
    )

    args = parser.parse_args()

    print("Preparing for training...")

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