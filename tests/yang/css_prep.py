import os
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
    fpath = r"H:\exp\dataset\LibriCSS\for_release\OV10\overlap_ratio_10.0_sil0.1_1.0_session2_actual10.0\record\utterances\utterance_59.wav"
    model_path = r"D:\Projects\pyprog\TDANet\Experiments\checkpoint\TDANet_EMCAD_before_Decoder_v1_6_noInit_full\epoch=164.ckpt"
    save_path = r"H:\exp\dataset\LibriCSS\CSS_Whisper\OV10\overlap_ratio_10.0_sil0.1_1.0_session2_actual10.0"
    s1 = os.path.join(save_path, "s1")
    s2 = os.path.join(save_path, "s2")
    y, sr = torchaudio.load(fpath)
    tgt_sr = 8000  # 目标采样率
    wav_len = 0  # 音频长度 
    seg_len = 3  # 分段长度
    pad_len = 0  # 末端补零长度
    segs = []
    est_segs = []
    outputs = []
    start_idx = 0
    fname = os.path.basename(fpath)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=tgt_sr, dtype=y.dtype)
    rev_resampler = torchaudio.transforms.Resample(orig_freq=tgt_sr, new_freq=sr, dtype=y.dtype)

    wav_len = y.shape[1]
    print("orig wav info:\n", y.shape, sr)
    # 7ch->1ch
    y = torch.mean(y, dim=0, keepdim=True) * 7
    print("mono wav info:\n", y.shape, sr)

    # 下采样至8kHz
    y_downsampled = resampler(y)
    y_downsampled = y_downsampled.cuda() if torch.cuda.is_available() else y_downsampled
    print("downsampled wav info:\n", y_downsampled.shape, tgt_sr)

    # 分割为3s片段
    while start_idx < y_downsampled.shape[1]:
        segs.append(y_downsampled[:, start_idx: start_idx + seg_len * tgt_sr])
        start_idx += seg_len * tgt_sr
    if start_idx > y_downsampled.shape[1]:
        pad_len = start_idx - y_downsampled.shape[1]
        segs[-1] = torch.nn.functional.pad(
            segs[-1], (0, pad_len)
        )

    # TDMGNet推理
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
    conf = OrderedDict(
        {key.replace('audio_model.', ''): value for key, value in conf.items()})
    model.load_state_dict(conf, strict=False)
    model = model.cuda() if torch.cuda.is_available() else model

    for seg in segs:
        with torch.no_grad():
            est_segs.append(model(seg))

    # Stitching
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

    y_upsampled = []
    for output in outputs:
        output = output.detach().cpu()
        output = output[:, :-pad_len] if pad_len != 0 else output  # 去除末端补零
        output = rev_resampler(output)  # 上采样至16kHz
        output = align_tensor_to_size(output, wav_len)  # 对齐长度
        y_upsampled.append(output)

    # 保存音频
    torchaudio.save(os.path.join(s1, fname), y_upsampled[0], sr)
    torchaudio.save(os.path.join(s2, fname), y_upsampled[1], sr)