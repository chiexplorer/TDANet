###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Please set LastEditors
# LastEditTime: 2022-09-26 11:14:20
# 评估测试集性能指标
###

import os
import random
from typing import Union
import soundfile as sf
import torch
import torchaudio
import yaml
import json
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
                    help="whether to save the separated audios")
parser.add_argument("--save_path",
                    default="./separated",
                    help="save path of separated audios")


compute_metrics = ["si_sdr", "sdr"]
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main(config):
    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "•",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress), 
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metricscolumn
    )
    # import pdb; pdb.set_trace()
    config["train_conf"]["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
    )
    model_path = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "best_model.pth")
    # import pdb; pdb.set_trace()
    # conf["train_conf"]["masknet"].update({"n_src": 2})
    # 预训练模型：JusperLee/TDANetBest-4ms-LRS2
    model = getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"]).from_pretrain(
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
    ex_save_dir = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "results/")
    os.makedirs(ex_save_dir, exist_ok=True)
    metrics = MetricsTracker(
        save_file=os.path.join(ex_save_dir, "metrics.csv"))
    torch.no_grad().__enter__()
    # 准备分离音频存储文件夹
    if config["save_output"] == "True":
        save_dir = os.path.dirname(config['conf_dir'])
        s1_path = os.path.join(save_dir, "separated", "s1")
        s2_path = os.path.join(save_dir, "separated", "s2")
        os.makedirs(s1_path, exist_ok=True)
        os.makedirs(s2_path, exist_ok=True)
    with progress:
        for idx in progress.track(range(len(test_set))):
            # Forward the network on the mixture.
            mix, sources, key = tensors_to_device(test_set[idx],
                                                    device=model_device)
            est_sources = model(mix[None])
            mix_np = mix
            sources_np = sources
            est_sources_np = est_sources.squeeze(0)
            metrics(mix=mix_np,
                    clean=sources_np,
                    estimate=est_sources_np,
                    key=key)
            if config["save_output"] == "True":
                torchaudio.save(os.path.join(s1_path, os.path.basename(test_set[idx][2])), est_sources[:, 0, :], config["train_conf"]["datamodule"]["data_config"]["sample_rate"])
                torchaudio.save(os.path.join(s2_path, os.path.basename(test_set[idx][2])), est_sources[:, 1, :], config["train_conf"]["datamodule"]["data_config"]["sample_rate"])
            # save_dir = "./TDANet"
            # # est_sources_np = normalize_tensor_wav(est_sources_np)
            # for i in range(est_sources_np.shape[0]):
            #     os.makedirs(os.path.join(save_dir, "s{}/".format(i + 1)), exist_ok=True)
                # torchaudio.save(os.path.join(save_dir, "s{}/".format(i + 1)) + key, est_sources_np[i].unsqueeze(0).cpu(), 16000)
            if idx % 50 == 0:
                metricscolumn.update(metrics.update())
            metricscolumn.update(metrics.update())
    metrics.final()


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    # print(arg_dic)
    main(arg_dic)
