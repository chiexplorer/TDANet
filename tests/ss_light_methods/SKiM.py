from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import time
import torch
import torch.nn as nn
from torch_complex.tensor import ComplexTensor
from espnet_utils import SingleRNN, merge_feature, split_feature, is_complex, choose_norm, AbsSeparator
from espnet_utils import ConvEncoder, ConvDecoder

class MemLSTM(nn.Module):
    """the Mem-LSTM of SkiM

    args:
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the LSTM layers are bidirectional.
            Default is False.
        mem_type: 'hc', 'h', 'c' or 'id'.
            It controls whether the hidden (or cell) state of
            SegLSTM will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states will
            be identically returned.
        norm_type: gLN, cLN. cLN is for causal implementation.
    """

    def __init__(
        self,
        hidden_size,
        dropout=0.0,
        bidirectional=False,
        mem_type="hc",
        norm_type="cLN",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = (int(bidirectional) + 1) * hidden_size
        self.mem_type = mem_type

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
        ], f"only support 'hc', 'h', 'c' and 'id', current type: {mem_type}"

        if mem_type in ["hc", "h"]:
            self.h_net = SingleRNN(
                "LSTM",
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.h_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )
        if mem_type in ["hc", "c"]:
            self.c_net = SingleRNN(
                "LSTM",
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.c_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )

    def extra_repr(self) -> str:
        return f"Mem_type: {self.mem_type}, bidirectional: {self.bidirectional}"

    def forward(self, hc, S):
        # hc = (h, c), tuple of hidden and cell states from SegLSTM
        # shape of h and c: (d, B*S, H)
        # S: number of segments in SegLSTM

        if self.mem_type == "id":
            ret_val = hc
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
        else:
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
            h = h.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            c = c.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            if self.mem_type == "hc":
                h = h + self.h_norm(self.h_net(h)[0])
                c = c + self.c_norm(self.c_net(c)[0])
            elif self.mem_type == "h":
                h = h + self.h_norm(self.h_net(h)[0])
                c = torch.zeros_like(c)
            elif self.mem_type == "c":
                h = torch.zeros_like(h)
                c = c + self.c_norm(self.c_net(c)[0])

            h = h.view(B * S, d, H).transpose(1, 0).contiguous()
            c = c.view(B * S, d, H).transpose(1, 0).contiguous()
            ret_val = (h, c)

        if not self.bidirectional:
            # for causal setup
            causal_ret_val = []
            for x in ret_val:
                x = x.transpose(1, 0).contiguous().view(B, S, d * H)
                x_ = torch.zeros_like(x)
                x_[:, 1:, :] = x[:, :-1, :]
                x_ = x_.view(B * S, d, H).transpose(1, 0).contiguous()
                causal_ret_val.append(x_)
            ret_val = tuple(causal_ret_val)

        return ret_val

    def forward_one_step(self, hc, state):
        if self.mem_type == "id":
            pass
        else:
            h, c = hc
            d, B, H = h.shape
            h = h.transpose(1, 0).contiguous().view(B, 1, d * H)  # B, 1, dH
            c = c.transpose(1, 0).contiguous().view(B, 1, d * H)  # B, 1, dH
            if self.mem_type == "hc":
                h_tmp, state[0] = self.h_net(h, state[0])
                h = h + self.h_norm(h_tmp)
                c_tmp, state[1] = self.c_net(c, state[1])
                c = c + self.c_norm(c_tmp)
            elif self.mem_type == "h":
                h_tmp, state[0] = self.h_net(h, state[0])
                h = h + self.h_norm(h_tmp)
                c = torch.zeros_like(c)
            elif self.mem_type == "c":
                h = torch.zeros_like(h)
                c_tmp, state[1] = self.c_net(c, state[1])
                c = c + self.c_norm(c_tmp)
            h = h.transpose(1, 0).contiguous()
            c = c.transpose(1, 0).contiguous()
            hc = (h, c)

        return hc, state


class SegLSTM(nn.Module):
    """the Seg-LSTM of SkiM

    args:
        input_size: int, dimension of the input feature.
            The input should have shape (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the LSTM layers are bidirectional.
            Default is False.
        norm_type: gLN, cLN. cLN is for causal implementation.
    """

    def __init__(
        self, input_size, hidden_size, dropout=0.0, bidirectional=False, norm_type="cLN"
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.norm = choose_norm(
            norm_type=norm_type, channel_size=input_size, shape="BTD"
        )

    def forward(self, input, hc):
        # input shape: B, T, H

        B, T, H = input.shape

        if hc is None:
            # In fist input SkiM block, h and c are not available
            d = self.num_direction
            h = torch.zeros(d, B, self.hidden_size, dtype=input.dtype).to(input.device)
            c = torch.zeros(d, B, self.hidden_size, dtype=input.dtype).to(input.device)
        else:
            h, c = hc

        output, (h, c) = self.lstm(input, (h, c))
        output = self.dropout(output)
        output = self.proj(output.contiguous().view(-1, output.shape[2])).view(
            input.shape
        )
        output = input + self.norm(output)

        return output, (h, c)

class SkiM(nn.Module):
    """Skipping Memory Net

    args:
        input_size: int, dimension of the input feature.
            Input shape shoud be (batch, length, input_size)
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_blocks: number of basic SkiM blocks
        segment_size: segmentation size for splitting long features
        bidirectional: bool, whether the RNN layers are bidirectional.
        mem_type: 'hc', 'h', 'c', 'id' or None.
            It controls whether the hidden (or cell) state of SegLSTM
            will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states will
            be identically returned.
            When mem_type is None, the MemLSTM will be removed.
        norm_type: gLN, cLN. cLN is for causal implementation.
        seg_overlap: Bool, whether the segmentation will reserve 50%
            overlap for adjacent segments.Default is False.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
        num_blocks=2,
        segment_size=20,
        bidirectional=True,
        mem_type="hc",
        norm_type="gLN",
        seg_overlap=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.norm_type = norm_type
        self.seg_overlap = seg_overlap

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
            None,
        ], f"only support 'hc', 'h', 'c', 'id', and None, current type: {mem_type}"

        self.seg_lstms = nn.ModuleList([])
        for i in range(num_blocks):
            self.seg_lstms.append(
                SegLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    norm_type=norm_type,
                )
            )
        if self.mem_type is not None:
            self.mem_lstms = nn.ModuleList([])
            for i in range(num_blocks - 1):
                self.mem_lstms.append(
                    MemLSTM(
                        hidden_size,
                        dropout=dropout,
                        bidirectional=bidirectional,
                        mem_type=mem_type,
                        norm_type=norm_type,
                    )
                )
        self.output_fc = nn.Sequential(
            nn.PReLU(), nn.Conv1d(input_size, output_size, 1)
        )

    def forward(self, input):
        # input shape: B, T (S*K), D
        B, T, D = input.shape

        if self.seg_overlap:
            input, rest = split_feature(
                input.transpose(1, 2), segment_size=self.segment_size
            )  # B, D, K, S
            input = input.permute(0, 3, 2, 1).contiguous()  # B, S, K, D
        else:
            input, rest = self._padfeature(input=input)
            input = input.view(B, -1, self.segment_size, D)  # B, S, K, D
        B, S, K, D = input.shape

        assert K == self.segment_size

        output = input.view(B * S, K, D).contiguous()  # BS, K, D
        hc = None
        for i in range(self.num_blocks):
            output, hc = self.seg_lstms[i](output, hc)  # BS, K, D
            if self.mem_type and i < self.num_blocks - 1:
                hc = self.mem_lstms[i](hc, S)
                pass

        if self.seg_overlap:
            output = output.view(B, S, K, D).permute(0, 3, 2, 1)  # B, D, K, S
            output = merge_feature(output, rest)  # B, D, T
            output = self.output_fc(output).transpose(1, 2)

        else:
            output = output.view(B, S * K, D)[:, :T, :]  # B, T, D
            output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)

        return output

    def _padfeature(self, input):
        B, T, D = input.shape
        rest = self.segment_size - T % self.segment_size

        if rest > 0:
            input = torch.nn.functional.pad(input, (0, 0, 0, rest))
        return input, rest

    def forward_stream(self, input_frame, states):
        # input_frame # B, 1, N

        B, _, N = input_frame.shape

        def empty_seg_states():
            shp = (1, B, self.hidden_size)
            return (
                torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype),
                torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype),
            )

        B, _, N = input_frame.shape
        if not states:
            states = {
                "current_step": 0,
                "seg_state": [empty_seg_states() for i in range(self.num_blocks)],
                "mem_state": [[None, None] for i in range(self.num_blocks - 1)],
            }

        output = input_frame

        if states["current_step"] and (states["current_step"]) % self.segment_size == 0:
            tmp_states = [empty_seg_states() for i in range(self.num_blocks)]
            for i in range(self.num_blocks - 1):
                tmp_states[i + 1], states["mem_state"][i] = self.mem_lstms[
                    i
                ].forward_one_step(states["seg_state"][i], states["mem_state"][i])

            states["seg_state"] = tmp_states

        for i in range(self.num_blocks):
            output, states["seg_state"][i] = self.seg_lstms[i](
                output, states["seg_state"][i]
            )

        states["current_step"] += 1

        output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)

        return output, states

class SkiMSeparator(AbsSeparator):
    """Skipping Memory (SkiM) Separator

    Args:
        input_dim: input feature dimension
        causal: bool, whether the system is causal.
        num_spk: number of target speakers.
        nonlinear: the nonlinear function for mask estimation,
            select from 'relu', 'tanh', 'sigmoid'
        layer: int, number of SkiM blocks. Default is 3.
        unit: int, dimension of the hidden state.
        segment_size: segmentation size for splitting long features
        dropout: float, dropout ratio. Default is 0.
        mem_type: 'hc', 'h', 'c', 'id' or None.
            It controls whether the hidden (or cell) state of
            SegLSTM will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states
            will be identically returned.
            When mem_type is None, the MemLSTM will be removed.
        seg_overlap: Bool, whether the segmentation will reserve 50%
            overlap for adjacent segments. Default is False.
    """

    def __init__(
        self,
        input_dim: int,
        causal: bool = True,
        num_spk: int = 2,
        predict_noise: bool = False,
        nonlinear: str = "relu",
        layer: int = 3,
        unit: int = 512,
        segment_size: int = 20,
        dropout: float = 0.0,
        mem_type: str = "hc",
        seg_overlap: bool = False,
    ):
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        self.segment_size = segment_size

        if mem_type not in ("hc", "h", "c", "id", None):
            raise ValueError("Not supporting mem_type={}".format(mem_type))

        self.num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.skim = SkiM(
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * self.num_outputs,
            dropout=dropout,
            num_blocks=layer,
            bidirectional=(not causal),
            norm_type="cLN" if causal else "gLN",
            segment_size=segment_size,
            seg_overlap=seg_overlap,
            mem_type=mem_type,
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        B, T, N = feature.shape

        processed = self.skim(feature)  # B,T, N

        processed = processed.view(B, T, N, self.num_outputs)
        masks = self.nonlinear(processed).unbind(dim=3)
        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, ilens, others

    def forward_streaming(self, input_frame: torch.Tensor, states=None):
        if is_complex(input_frame):
            feature = abs(input_frame)
        else:
            feature = input_frame

        B, _, N = feature.shape

        processed, states = self.skim.forward_stream(feature, states=states)

        processed = processed.view(B, 1, N, self.num_outputs)
        masks = self.nonlinear(processed).unbind(dim=3)
        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input_frame * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input_frame * mask_noise

        return masked, states, others

    @property
    def num_spk(self):
        return self._num_spk

class SKiMReprod(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = ConvEncoder(
            channel=64,
            kernel_size=2,
            stride=1,
        )
        self.separator = SkiMSeparator(
            input_dim,
            causal=False,
            num_spk=2,
            predict_noise=False,
            nonlinear="relu",
            layer=6,
            unit=128,
            segment_size=250,
            dropout=0.1,
            mem_type="hc",
            seg_overlap=True
        )
        self.decoder = ConvDecoder(
            channel=64,
            kernel_size=2,
            stride=1,
        )

    def forward(self, x, ilens):
        enc = self.encoder(x, ilens)
        enc = nn.functional.pad(enc[0], (0, 0, 0, 1))
        h = self.separator(enc, ilens)
        h = h[0]
        decoded = []
        for si in h:
            decoded.append(self.decoder(si[:, 0:-1, :], ilens)[0])
        return decoded


if __name__ == '__main__':
    import time
    from thop import profile
    from torchinfo import summary
    from ptflops import get_model_complexity_info
    from look2hear.datas.wsj02mixdatamodule import WSJ0DataModule
    import torchaudio
    import os
    from tqdm import tqdm
    from look2hear.metrics import MetricsTracker
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )
    from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    batch_size = 2
    audio_len = 24000
    input_dim = 64
    ckpt_path = r"H:\exp\model\SKiM_nocausual\146epoch.pth"
    test_set_path = r"D:\Projects\pyprog\TDANet\DataPreProcess\WSJ02mix\tt_eval"
    ilens = torch.tensor([audio_len]*batch_size, device=device)
    model = SKiMReprod(input_dim=input_dim)
    x = torch.randn(1, audio_len, dtype=torch.float32, device=device)
    # 载入模型参数测试
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    # model.cuda()
    # 构建测试集loader
    wsj0_loader = WSJ0DataModule(
        train_dir = test_set_path,
        valid_dir = test_set_path,
        test_dir = test_set_path,
        n_src = 2,
        sample_rate = 8000,
        segment = 3.0,
        normalize_audio = False,
        batch_size = batch_size,
        num_workers = 2,
        pin_memory = True,
        persistent_workers = False
    )
    wsj0_loader.setup()
    _, _ , test_set = wsj0_loader.make_sets
    # 统计指标
    save_result = True
    save_dir = "./output"
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
    metrics = MetricsTracker(
        save_file=os.path.join("./", "metrics.csv"))
    with progress:
        start_time = time.time()
        for idx in tqdm(progress.track(range(len(test_set)))):
            # Forward the network on the mixture.
            mix, sources, key = tensors_to_device(test_set[idx],
                                                    device=device)
            est_sources = model(mix[None], ilens)
            est_sources = torch.cat(est_sources, dim=0)
            est_sources = est_sources.unsqueeze(0)
            mix_np = mix
            sources_np = sources
            est_sources_np = est_sources.squeeze(0)
            metrics(mix=mix_np,
                    clean=sources_np,
                    estimate=est_sources_np,
                    key=key)
            if save_result is True:
                s1_path = os.path.join(save_dir, "skim", "s1")
                s2_path = os.path.join(save_dir, "skim", "s2")
                os.makedirs(s1_path, exist_ok=True)
                os.makedirs(s2_path, exist_ok=True)
                est_sources = est_sources.detach().cpu()
                torchaudio.save(os.path.join(s1_path, os.path.basename(test_set[idx][2])), est_sources[:, 0, :], 8000)
                torchaudio.save(os.path.join(s2_path, os.path.basename(test_set[idx][2])), est_sources[:, 1, :], 8000)
            # save_dir = "./TDANet"
            # # est_sources_np = normalize_tensor_wav(est_sources_np)
            # for i in range(est_sources_np.shape[0]):
            #     os.makedirs(os.path.join(save_dir, "s{}/".format(i + 1)), exist_ok=True)
                # torchaudio.save(os.path.join(save_dir, "s{}/".format(i + 1)) + key, est_sources_np[i].unsqueeze(0).cpu(), 16000)
            if idx % 50 == 0:
                metricscolumn.update(metrics.update())
            metricscolumn.update(metrics.update())
        print(f"Deal time: [{time.time() - start_time}] seconds for [{idx+1}] items.")
    metrics.final()
    # 模型计算量
    # macs, params = profile(model, inputs=(x, ilens))
    # mb = 1000*1000
    # print(f"MACs: [{macs/mb/1000}] Gb \nParams: [{params/mb}] Mb")
    # # 详细模型计算量
    # mb = 1000 * 1000
    # input_res = ((1, audio_len), (input_dim, ))
    # macs, params = get_model_complexity_info(
    #     model, (1, audio_len), as_strings=False, print_per_layer_stat=True, input_constructor=lambda _: {"x": x, "ilens": ilens}
    # )
    # print(f'Computational complexity: {macs/mb}')
    # print(f'Number of parameters: {params/mb/1000}')
    # # 模型参数量
    # print("模型参数量详情：\n", summary(model, input_data=(x, ilens), mode="train"))
    # # 前向计算耗时
    # start_time = time.time()
    # y = model(x, ilens)
    # print("batch耗时：{:.4f}".format(time.time() - start_time))
    # _ = [print(si.shape) for si in y]
