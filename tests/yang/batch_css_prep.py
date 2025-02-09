import os
import tqdm
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
from collections import OrderedDict
from look2hear.models.TDANet_intergral_v1_6 import TDANetEMCADv1_6

"""
    对齐audio至指定长度
    @params:
        tensor: 待对齐的音频，shape(1, n)
        size: 目标长度, int
"""
def align_tensor_to_size(tensor, size):
    # 获取当前 tensor 的尺寸
    current_size = tensor.size(1)

    if current_size > size:
        # 截断 tensor 到目标大小
        return tensor[:, :size]
    elif current_size < size:
        # 补零到目标大小
        padding_size = size - current_size
        return F.pad(tensor, (0, padding_size), mode='constant', value=0)
    else:
        # 如果已经是目标大小，直接返回原 tensor
        return tensor


if __name__ == '__main__':
    dir_path = r"H:\exp\dataset\LibriCSS\for_release"
    output_path = r"H:\exp\dataset\LibriCSS\CSS_Whisper"
    utter_dir_name = os.path.join("record", "utterances")  # 话语路径(相对于session)
    output_channel1_name = "s1"
    output_channel2_name = "s2"
    excuded_sessions = ["0L", "0S"]
    overlap_ratios = []
    sessions = []
    # 创建一级输出目录(重叠率)
    for root, dirs, files in os.walk(dir_path):
        if root == dir_path:
            for d in dirs:
                os.makedirs(os.path.join(output_path, d), exist_ok=True)
                overlap_ratios.append(d)

    # 创建二级输出目录(session)
    for overlap_ratio in overlap_ratios:
        for root, dirs, files in os.walk(os.path.join(dir_path, overlap_ratio)):
            if root == os.path.join(dir_path, overlap_ratio):
                for d in dirs:
                    os.makedirs(os.path.join(output_path, overlap_ratio, d), exist_ok=True)  # session
                    os.makedirs(os.path.join(output_path, overlap_ratio, d, output_channel1_name), exist_ok=True)  # CSS输出通道1
                    os.makedirs(os.path.join(output_path, overlap_ratio, d, output_channel2_name), exist_ok=True)

    # CSS准备工作
    model_path = r"D:\Projects\pyprog\TDANet\Experiments\checkpoint\TDANet_EMCAD_before_Decoder_v1_6_noInit_full\epoch=164.ckpt"
    orig_sr = 16000  # 原始采样率
    tgt_sr = 8000  # 目标采样率
    wav_dtype = torch.float32  # 音频数据类型
    seg_len = 3  # 分段长度
    # 下采样&上采样器
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=tgt_sr, dtype=wav_dtype)  # 上采样
    rev_resampler = torchaudio.transforms.Resample(orig_freq=tgt_sr, new_freq=orig_sr, dtype=wav_dtype)  # 下采样
    ## 模型
    model = TDANetEMCADv1_6(
        out_channels=128,
        in_channels=512,
        num_blocks=8,
        upsampling_depth=5,
        enc_kernel_size=4,
        num_sources=2,
        sample_rate=8000,
        feat_len=3010
    )
    conf = torch.load(model_path)['state_dict']
    conf = OrderedDict({key.replace('audio_model.', ''): value for key, value in conf.items()})
    model.load_state_dict(conf, strict=False)
    if torch.cuda.is_available():
        model = model.cuda()

    print("开始执行CSS...")
    # 处理话语，保存CSS结果
    for overlap_ratio in overlap_ratios:
        if overlap_ratio in excuded_sessions:
            print("跳过session [{}]...".format(overlap_ratio))
            continue
        print("当前overlap ratio: ", overlap_ratio)
        sessions = os.listdir(os.path.join(dir_path, overlap_ratio))
        for session in tqdm.tqdm(sessions):
            utter_dir = os.path.join(dir_path, overlap_ratio, session, utter_dir_name)
            utter_names = os.listdir(utter_dir)
            output_dir = os.path.join(output_path, overlap_ratio, session)  # 输出路径(至session)
            for utter_name in utter_names:
                try:
                    ### 1 reinit
                    wav_len = 0  # 音频长度
                    start_idx = 0
                    pad_len = 0  # 末端补零长度
                    segs = []
                    est_segs = []
                    outputs = []
                    wav_path = os.path.join(utter_dir, utter_name)
                    ### 2 载入音频&预处理
                    y, sr = torchaudio.load(wav_path)
                    wav_len = y.shape[1]
                    # 7ch->1ch
                    y = torch.mean(y, dim=0, keepdim=True) * 7
                    # 下采样至8kHz
                    y_downsampled = resampler(y)
                    y_downsampled = y_downsampled.cuda() if torch.cuda.is_available() else y_downsampled
                    # 分割为3s片段
                    while start_idx < y_downsampled.shape[1]:
                        segs.append(y_downsampled[:, start_idx: start_idx + seg_len * tgt_sr])
                        start_idx += seg_len * tgt_sr
                    if start_idx > y_downsampled.shape[1]:
                        pad_len = start_idx - y_downsampled.shape[1]
                        segs[-1] = torch.nn.functional.pad(
                            segs[-1], (0, pad_len)
                        )

                    ### 3 TDMGNet分离
                    for seg in segs:
                        with torch.no_grad():
                            est_segs.append(model(seg))

                    ### 4 Stitching
                    for i, est_seg in enumerate(est_segs):
                        est_1 = est_seg[:, 0, :]
                        est_2 = est_seg[:, 1, :]
                        if i == 0:
                            outputs.append(est_1)
                            outputs.append(est_2)
                        else:
                            # 计算余弦相似度
                            cos_simi_1 = F.cosine_similarity(est_1, est_segs[i-1][:, 0, :], dim=1)
                            cos_simi_2 = F.cosine_similarity(est_1, est_segs[i-1][:, 1, :], dim=1)
                            # 根据chan_1和前一分段上两通道的相似度，决定拼接方式
                            if cos_simi_1 > cos_simi_2:
                                outputs[0] = torch.cat((outputs[0], est_1), dim=1)  # 正序拼接
                                outputs[1] = torch.cat((outputs[1], est_2), dim=1)
                            else:
                                outputs[0] = torch.cat((outputs[0], est_2), dim=1)  # 交叉拼接
                                outputs[1] = torch.cat((outputs[1], est_1), dim=1)
                    ### 5 后处理
                    y_upsampled = []
                    for output in outputs:
                        output = output.detach().cpu()
                        output = output[:, :-pad_len] if pad_len != 0 else output  # 去除末端补零
                        output = rev_resampler(output)  # 上采样至16kHz
                        output = align_tensor_to_size(output, wav_len)  # 对齐长度
                        y_upsampled.append(output)

                    # 保存音频
                    torchaudio.save(os.path.join(output_dir, output_channel1_name, utter_name), y_upsampled[0], orig_sr)
                    torchaudio.save(os.path.join(output_dir, output_channel2_name, utter_name), y_upsampled[1], orig_sr)
                except Exception as e:
                    print("处理音频{}时出错: {}".format(utter_name, e))
    print("CSS执行完毕...")