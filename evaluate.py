import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy
import matplotlib.pyplot as plt

from utils.getModel_utils import get_model_fs, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from models import FastSpeech2Loss
from dataset import Dataset


def evaluate(model, step, configs, logger=None, vocoder=None, device=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "shuffled_val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        pin_memory=True, 
        num_workers=8
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(10)]  #init a list with six "zero elements"
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(
                                *batch[2:],
                                train = True
                ) 

                # Cal Loss
                losses= Loss(batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message1 = "step{}, Val total Loss: {:.4f}, Mel Loss 5: {:.4f}, Mel Loss 6: {:.4f}, Ssim Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f},\n"\
        "Ce Loss: {:.4f}, Kl loss: {:.4f}".format(
        step, *losses
    )
    message = message1

    #
    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
        batch,
        output,
        vocoder,
        model_config,
        preprocess_config,
    )
    #save audio
    save_dir = os.path.join(train_config["path"]["result_path"], f"{step}", "val")
    os.makedirs(save_dir, exist_ok=True)
    audio = wav_reconstruction
    scipy.io.wavfile.write(
        filename=os.path.join(save_dir, f"{step}val_gt.wav"),
        rate=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        data=audio,
    )
    os.makedirs(save_dir, exist_ok=True)
    audio = wav_prediction
    scipy.io.wavfile.write(
        filename=os.path.join(save_dir, f"{step}val_pre.wav"),
        rate=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        data=audio,
    )
    # save fig
    filename = f"{step}val.png"  # 图像文件名
    fig.savefig(os.path.join(save_dir, filename))  # 保存图像
    plt.close(fig)  # 关闭图形对象，释放内存

    if True:
        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--restore_step", type=int, default=0, help="path to **.tar")
    parser.add_argument(
        "-p","--preprocess_config",type=str,required=False,default="config/ESD-en/preprocess.yaml",help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=False,default="config/ESD-en/model.yaml", help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=False,default="config/ESD-en/train.yaml", help="path to train.yaml"
    )

    args = parser.parse_args()
    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model_fs(args, configs, device, train=False).to(device)
    vocoder = get_vocoder(model_config, device)

    message = evaluate(model, args.restore_step, configs, logger=None, vocoder=vocoder)
    print(message)