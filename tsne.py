import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import torch
import argparse
import yaml
import os
from torch.utils.data import DataLoader
from dataset import DatasetFsWithMixFor_ESD_neuDOE_tsne
from utils.tools import to_device_mix
from utils.getModel_utils import get_model_fs


def plot(x, emotions, speakers, colors, train_config, step, use_speaker_symbols=True):
    """
    绘制情感和说话人嵌入的 t-SNE 降维结果。

    Args:
        x: 降维后的嵌入数据，形状为 (n_samples, 2)。
        emotions: 情感标签列表，用于颜色区分。
        speakers: 说话人标签列表，用于符号区分。
        colors: 用于映射情感标签的颜色。
        train_config: 训练配置字典。
        step: 当前步数，用于保存路径。
        use_speaker_symbols: 是否依据说话人使用不同符号，默认 True。
    """
    # 情感映射规则
    emotion_map = {0: "angry", 1: "happpy", 2: "neutral", 3: "sad", 4: "surprise"}
    # 说话人映射规则
    speaker_map = {
        1: "esd.001",  # esd
        2: "esd.002",   
        3: "esd.003",   
        4: "esd.004",   
        5: "esd.005",   
        6: "esd.006",   
        7: "esd.007",   
        8: "esd.008",   
        9: "esd.009",   
        10: "esd.010",
        11: "old.fema",
        12: "old.male",
        13: "you.male",
        14: "you.fema",
    }


    palette = np.array(sns.color_palette("husl", len(np.unique(emotions))))
    speaker_symbols = ['s', '^', 'D', 'P', 'o']  # 符号数量应覆盖所有说话人

    unique_speakers = np.unique(speakers)
    unique_emotions = np.unique(emotions)

    plt.figure(figsize=(8, 8))  # 增加图像大小
    ax = plt.subplot(aspect="equal")

    # 绘制每个情感和说话人的点
    for speaker_id in unique_speakers:
        for emotion_id in unique_emotions:
            # 筛选当前说话人和情感的点
            indices = (speakers == speaker_id) & (emotions == emotion_id)
            num_points = np.sum(indices)  # 匹配点数量
            print(f"Speaker {speaker_map[speaker_id]}, Emotion {emotion_map[emotion_id]}, Data points: {num_points}")

            if num_points > 0:
                ax.scatter(
                    x[indices, 0], x[indices, 1],
                    lw=0, s=30, c=[palette[emotion_id]],  # 使用映射的情感颜色
                    marker=speaker_symbols[list(unique_speakers).index(speaker_id)] if use_speaker_symbols else 'o',
                    label=f'{emotion_map[emotion_id]}, {speaker_map[speaker_id]}',
                    alpha=0.7  # 增加透明度
                )

    # 添加图例
    legend_elements = []
    for emotion_id, color in enumerate(palette):
        legend_elements.append(Line2D(
            [0], [0], color=color, marker='o', linestyle='None',
            markersize=6, label=f'{emotion_map[emotion_id]}'  # 缩小符号
        ))

    if use_speaker_symbols:
        for speaker_id, symbol in enumerate(speaker_symbols[:len(unique_speakers)]):
            legend_elements.append(Line2D(
                [0], [0], color='black', marker=symbol, linestyle='None',
                markersize=6, label=f'{speaker_map[speaker_id]}'  # 缩小符号
            ))

    # 控制图例布局和位置
    ax.legend(
        handles=legend_elements,
        loc='upper left',              # 图例位置：右上角
        bbox_to_anchor=(0.0, 1.0),      # 调整图例相对于绘图区域的位置
        fontsize=8,                     # 缩小字体
        ncol=1,                         # 设置为1列显示
        frameon=False                   # 是否显示边框
    )

    # 是否去掉 t-SNE 图的外边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 去掉坐标轴上的数字和刻度
    ax.set_xticks([])  # 去掉 x 轴刻度
    ax.set_yticks([])  # 去掉 y 轴刻度

    # 保存图像
    result_path = train_config["path"]["result_path"]
    save_path = os.path.join(result_path, str(step), "tsne", "emotion_tsne_val.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)  # 提高分辨率
    plt.close()

    print(f"t-SNE plot saved to {save_path}")


def t_SNE(model, configs, device=None, step=None):
    preprocess_config, model_config, train_config = configs

    # 获取数据集
    dataset = DatasetFsWithMixFor_ESD_neuDOE_tsne(
        "val_shuffled_2.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = 64
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # 收集嵌入和标签
    emotion_embedding_list = []
    emotion_target_list = []
    speaker_target_list = []

    for batches in loader:
        for batch in batches:
            batch = to_device_mix(batch, device)
            with torch.no_grad():
                output = model.tsne(*(batch[2:]), train=False)
            emotion_target = output[-3].squeeze(0).cpu().numpy()
            speaker_target = batch[5].squeeze(0).cpu().numpy()
            emotion_embedding = output[-2].squeeze(0, 1).cpu().numpy()

            emotion_embedding_list.extend(emotion_embedding)
            speaker_target_list.extend(speaker_target)
            emotion_target_list.extend(emotion_target)

    # 转换为 NumPy 数组
    emo_emb = np.array(emotion_embedding_list)
    spk_target = np.array(speaker_target_list)
    emo_target = np.array(emotion_target_list)

    print("Emotion embedding shape:", emo_emb.shape)
    print("Emotion target shape:", emo_target.shape)
    print("Speaker target shape:", spk_target.shape)

    # 打印唯一值
    print("Unique speaker IDs:", np.unique(spk_target))
    print("Unique emotion IDs:", np.unique(emo_target))

    # t-SNE降维
    print("Performing t-SNE...")
    #emo_emb_final = TSNE(perplexity=30, n_iter=1000).fit_transform(emo_emb)
    emo_emb_final = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(emo_emb) #cuda加速

    # 绘制降维结果
    plot(emo_emb_final, emo_target, spk_target, emo_target, train_config, step, use_speaker_symbols=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--device", type=str, default="cuda:2", help="Device to use")
    parser.add_argument("-r","--restore_step", type=int, default=382000, help="path to **.tar")

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

    #get device
    device = torch.device(args.device)
    print("Using Device:", device)

    # 读取配置文件
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)

    configs = (preprocess_config, model_config, train_config)

    # 加载模型
    model = get_model_fs(args, configs, device, train=False).to(device)

    # 执行 t-SNE 可视化
    t_SNE(model, configs, device, args.restore_step)
