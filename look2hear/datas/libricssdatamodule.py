import os
import json
from tkinter.tix import Tree
import numpy as np
from typing import Any, Tuple, Union
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing import Dict, Iterable, List, Iterator
from rich import print
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_(message: str):
    print(message)


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class LibriCSSDataset(Dataset):
    """
        Dataset for LibriCSS.

        inputs:
            input_dir: str, 子集所在的文件夹，文件夹包含音频文件
            n_src: int
            sample_rate: int
            segment: float，分片长度，单位秒
            overlap: float，重叠率，~[0, 1]
            normalize_audio: bool
            audio_only: bool

        returns:
            segments: List[List[str, List[Tensor], int]]，深度2中的三个元素分别是：音频名、分片、补零长度
    """
    def __init__(
        self,
        input_dir: str = "",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        overlap: float = 0.25,
        normalize_audio: bool = False,
        audio_only: bool = True,
    ) -> None:
        super().__init__()
        self.EPS = 1e-8
        if input_dir == None or input_dir == "":
            raise ValueError("Input DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        self.input_dir = input_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.audio_only = audio_only
        if segment is None:
            self.seg_len = None
        else:
            self.seg_len = int(segment * sample_rate)
        self.overlap = overlap
        self.n_src = n_src
        self.test = self.seg_len is None
        self.segments = []  # [音频 <[音频名, [分片 <tensor>], 补零长度]>]
        drop_utt = 0  # 丢弃音频数
        hop_len = int(self.seg_len * (1 - self.overlap))
        # 读长音频
        audio_names = os.listdir(input_dir)
        for audio_name in audio_names:
            if not audio_name.endswith(".wav"):
                continue
            start_idx = 0
            pad_len = 0
            audio_segs = [audio_name, [], pad_len]
            audio_path = os.path.join(input_dir, audio_name)

            audio_len = sf.info(audio_path).frames
            if audio_len < self.seg_len:
                drop_utt += 1
            while start_idx < audio_len:
                seg, _ = sf.read(
                    audio_path, start=start_idx, stop=start_idx + self.seg_len, dtype="float32"
                )  # 分割
                seg = torch.from_numpy(seg)

                # 补零
                if start_idx + self.seg_len > audio_len:
                    pad_len = start_idx + self.seg_len - audio_len
                    seg = torch.cat(
                        [seg, torch.zeros(pad_len, dtype=seg.dtype)]
                    )
                    audio_segs[2] = pad_len  # 记录
                    start_idx += pad_len  # pad计入起始下标
                if self.normalize_audio:
                    m_std = seg.std(-1, keepdim=True)
                    seg = normalize_tensor_wav(seg, eps=self.EPS, std=m_std)
                audio_segs[1].append(seg)
                start_idx += hop_len
            self.segments.append(audio_segs)
        print_(
            "Drop {} utts(shorter than {} samples)".format(
                drop_utt, self.seg_len
            )
        )
        self.length = len(self.segments)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return self.segments[index]


class LibriCSSDataModule(object):
    def __init__(
        self,
        train_dir: str = "",
        valid_dir: str = "",
        test_dir: str = "",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        overlap: float = 0.25,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        audio_only: bool = True,
    ) -> None:
        super().__init__()
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))

        # this line allows to access init params with 'self.hparams' attribute
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.audio_only = audio_only

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = LibriCSSDataset(
            input_dir=self.train_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            overlap=self.overlap,
            normalize_audio=self.normalize_audio,
            audio_only=self.audio_only,
        ) if self.train_dir != None and self.train_dir != "" else None
        self.data_val = LibriCSSDataset(
            input_dir=self.valid_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            overlap=self.overlap,
            normalize_audio=self.normalize_audio,
            audio_only=self.audio_only,
        ) if self.valid_dir != None and self.valid_dir != "" else None
        self.data_test = LibriCSSDataset(
            input_dir=self.test_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            overlap=self.overlap,
            normalize_audio=self.normalize_audio,
            audio_only=self.audio_only,
        ) if self.test_dir != None and self.test_dir != "" else None

    def train_dataloader(self) -> Union[DataLoader, None]:
        if self.data_train is None:
            return None
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> Union[DataLoader, None]:
        if self.data_val is None:
            return None
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> Union[DataLoader, None]:
        if self.data_test is None:
            return None
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test


if __name__ == '__main__':
    fpath = r"D:\Projects\pyprog\TDANet\DataPreProcess\LibriCSS\debug"
    data_configs = {
        "train_dir": "",
        "valid_dir": "",
        "test_dir": fpath,
        "n_src": 2,
        "sample_rate": 8000,
        "segment": 3.0,
        "overlap": 0.25,
        "normalize_audio": False,
        "batch_size": 1,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": False
    }

    datamodule = LibriCSSDataModule(**data_configs)
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    for item in test_loader:
        print(item)
        break
