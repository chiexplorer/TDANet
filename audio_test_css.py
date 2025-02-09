###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Please set LastEditors
# LastEditTime: 2022-09-26 11:14:20
# 评估LibriCSS测试集性能指标
# 输出：与混合音频同名的两个分离通道音频，存储到指定文件夹下的s1和s2文件夹中
###

import os
import random
from typing import Union
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import yaml
import json
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from scipy.io import wavfile
import warnings
# import torchaudio
warnings.filterwarnings("ignore")
import look2hear.models
import look2hear.datas
from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir",
                    default="local/mixit_conf.yml",
                    help="Full path to save best validation model")
parser.add_argument("--ckpt_path",
                    default="",
                    help="save path of checkpoint file")
parser.add_argument("--save_output",
                    default="False",
                    help="whether to save the skim audios")
parser.add_argument("--save_path",
                    default="./skim",
                    help="save path of skim audios")



compute_metrics = ["si_sdr", "sdr"]
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main(config):
    # import pdb; pdb.set_trace()
    config["train_conf"]["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
    )
    model_path = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "best_model.pth")
    # import pdb; pdb.set_trace()
    # conf["train_conf"]["masknet"].update({"n_src": 2})
    # 预训练模型：JusperLee/TDANetBest-4ms-LRS2
    model = getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"]).from_pretrain(
        config["train_conf"]["audionet"]["audionet_name"],
        config["ckpt_path"],
        sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
        **config["train_conf"]["audionet"]["audionet_config"],
    )
    if config["train_conf"]["training"]["gpus"]:
        device = "cuda"
        model.to(device)
    model_device = next(model.parameters()).device
    print(config["train_conf"]["datamodule"]["data_config"])
    datamodule: object = getattr(look2hear.datas, config["train_conf"]["datamodule"]["data_name"])(
        **config["train_conf"]["datamodule"]["data_config"]
    )
    datamodule.setup()
    _, _ , test_set = datamodule.make_sets
   
    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(config["train_conf"]["save_dir"], "results/")
    os.makedirs(ex_save_dir, exist_ok=True)
    torch.no_grad().__enter__()
    # 准备分离音频存储文件夹
    if config["save_output"] == "True":
        save_dir = config['save_path']
        s1_path = os.path.join(save_dir, "s1")
        s2_path = os.path.join(save_dir, "s2")
        os.makedirs(s1_path, exist_ok=True)
        os.makedirs(s2_path, exist_ok=True)
    # 分片重叠的长度
    overlap_len = int(config['train_conf']['datamodule']['data_config']['sample_rate'] *
                      config['train_conf']['datamodule']['data_config']['segment'] *
                      config['train_conf']['datamodule']['data_config']['overlap'])
    start_time = time.time()
    for idx in tqdm(range(len(test_set))):
        f_name = test_set[idx][0]
        pad_len = test_set[idx][2]
        # Forward the network on the segments.
        for k, seg in enumerate(test_set[idx][1]):

            seg = tensors_to_device(seg, device=model_device)
            est_sources = model(seg)
            est_sources = est_sources.detach().cpu()
            est_s1 = est_sources[0, :]
            est_s2 = est_sources[1, :]
            # 第一次迭代时，执行初始化
            if k == 0:
                output1 = torch.unsqueeze(est_s1, dim=0)
                output2 = torch.unsqueeze(est_s2, dim=0)
                s1_t_minus_1 = est_s1[-overlap_len:]
                s2_t_minus_1 = est_s2[-overlap_len:]
            else:
                # 计算排列得分，组合的余弦相似度之和
                comb1_score = F.cosine_similarity(s1_t_minus_1, est_s1[0:overlap_len], dim=0) + F.cosine_similarity(s2_t_minus_1, est_s2[0:overlap_len], dim=0)
                comb2_score = F.cosine_similarity(s1_t_minus_1, est_s2[0:overlap_len], dim=0) + F.cosine_similarity(s2_t_minus_1, est_s1[0:overlap_len], dim=0)
                # 根据排列相似度决定拼接关系，追加到输出流
                if comb1_score > comb2_score:
                    output1 = torch.cat([output1, torch.unsqueeze(est_s1[overlap_len:], dim=0)], dim=1)
                    output2 = torch.cat([output2, torch.unsqueeze(est_s2[overlap_len:], dim = 0)], dim = 1)
                else:
                    output1 = torch.cat([output1, torch.unsqueeze(est_s2[overlap_len:], dim=0)], dim=1)
                    output2 = torch.cat([output2, torch.unsqueeze(est_s1[overlap_len:], dim = 0)], dim = 1)
        # 去除末端的补零，并存为音频
        output1 = output1[:, :-pad_len]
        output2 = output2[:, :-pad_len]
        torchaudio.save(os.path.join(s1_path, f_name), output1, config["train_conf"]["datamodule"]["data_config"]["sample_rate"])
        torchaudio.save(os.path.join(s2_path, f_name), output2, config["train_conf"]["datamodule"]["data_config"]["sample_rate"])
    print(f"Deal time: [{time.time() - start_time}] seconds for [{idx}] items.")


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    # print(arg_dic)
    main(arg_dic)
