from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_complex.tensor import ComplexTensor

EPS = torch.finfo(torch.get_default_dtype()).eps

class SingleRNN(nn.Module):
    """Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False
    ):
        super().__init__()

        rnn_type = rnn_type.upper()

        assert rnn_type in [
            "RNN",
            "LSTM",
            "GRU",
        ], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input, state=None):
        # input shape: batch, seq, dim
        # input = input.to(device)
        output = input
        rnn_output, state = self.rnn(output, state)
        rnn_output = self.dropout(rnn_output)
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(output.shape)
        return rnn_output, state

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)."""

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            gLN_y: [M, N, K]
        """
        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = y.mean(dim=(1, 2), keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=(1, 2), keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            gLN_y = gLN_y.transpose(1, 2).contiguous()
        return gLN_y

class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)."""

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            cLN_y: [M, N, K]
        """

        assert y.dim() == 3

        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            cLN_y = cLN_y.transpose(1, 2).contiguous()

        return cLN_y

class AbsSeparator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        raise NotImplementedError

    def forward_streaming(
        self,
        input_frame: torch.Tensor,
        buffer=None,
    ):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_spk(self):
        raise NotImplementedError

class AbsEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        fs: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    def forward_streaming(self, input: torch.Tensor):
        raise NotImplementedError

    def streaming_frame(self, audio: torch.Tensor):
        """Stream frame.

        It splits the continuous audio into frame-level
        audio chunks in the streaming *simulation*. It is noted that this
        function takes the entire long audio as input for a streaming simulation.
        You may refer to this function to manage your streaming input
        buffer in a real streaming application.

        Args:
            audio: (B, T)
        Returns:
            chunked: List [(B, frame_size),]
        """
        raise NotImplementedError

class ConvEncoder(AbsEncoder):
    """Convolutional encoder for speech enhancement and separation"""

    def __init__(
        self,
        channel: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            1, channel, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.stride = stride
        self.kernel_size = kernel_size

        self._output_dim = channel

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz (Not used)
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
        """
        assert input.dim() == 2, "Currently only support single channel input"

        input = torch.unsqueeze(input, 1)

        feature = self.conv1d(input)
        feature = torch.nn.functional.relu(feature)
        feature = feature.transpose(1, 2)

        flens = (
            torch.div(ilens - self.kernel_size, self.stride, rounding_mode="trunc") + 1
        )

        return feature, flens

    def forward_streaming(self, input: torch.Tensor):
        output, _ = self.forward(input, 0)
        return output

    def streaming_frame(self, audio: torch.Tensor):
        """Stream frame.

        It splits the continuous audio into frame-level
        audio chunks in the streaming *simulation*. It is noted that this
        function takes the entire long audio as input for a streaming simulation.
        You may refer to this function to manage your streaming input
        buffer in a real streaming application.

        Args:
            audio: (B, T)
        Returns:
            chunked: List [(B, frame_size),]
        """
        batch_size, audio_len = audio.shape

        hop_size = self.stride
        frame_size = self.kernel_size

        audio = [
            audio[:, i * hop_size : i * hop_size + frame_size]
            for i in range((audio_len - frame_size) // hop_size + 1)
        ]

        return audio

class AbsDecoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        fs: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward_streaming(self, input_frame: torch.Tensor):
        raise NotImplementedError

    def streaming_merge(self, chunks: torch.Tensor, ilens: torch.tensor = None):
        """Stream merge.

        It merges the frame-level processed audio chunks
        in the streaming *simulation*. It is noted that, in real applications,
        the processed audio should be sent to the output channel frame by frame.
        You may refer to this function to manage your streaming output buffer.

        Args:
            chunks: List [(B, frame_size),]
            ilens: [B]
        Returns:
            merge_audio: [B, T]
        """

        raise NotImplementedError

class ConvDecoder(AbsDecoder):
    """Transposed Convolutional decoder for speech enhancement and separation"""

    def __init__(
        self,
        channel: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.convtrans1d = torch.nn.ConvTranspose1d(
            channel, 1, kernel_size, bias=False, stride=stride
        )

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): spectrum [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz (Not used)
        """
        input = input.transpose(1, 2)
        batch_size = input.shape[0]
        wav = self.convtrans1d(input, output_size=(batch_size, 1, ilens.max()))
        wav = wav.squeeze(1)

        return wav, ilens

    def forward_streaming(self, input_frame: torch.Tensor):
        return self.forward(input_frame, ilens=torch.LongTensor([self.kernel_size]))[0]

    def streaming_merge(self, chunks: torch.Tensor, ilens: torch.tensor = None):
        """Stream Merge.

        It merges the frame-level processed audio chunks
        in the streaming *simulation*. It is noted that, in real applications,
        the processed audio should be sent to the output channel frame by frame.
        You may refer to this function to manage your streaming output buffer.

        Args:
            chunks: List [(B, frame_size),]
            ilens: [B]
        Returns:
            merge_audio: [B, T]
        """
        hop_size = self.stride
        frame_size = self.kernel_size

        num_chunks = len(chunks)
        batch_size = chunks[0].shape[0]
        audio_len = (
            int(hop_size * num_chunks + frame_size - hop_size)
            if not ilens
            else ilens.max()
        )

        output = torch.zeros((batch_size, audio_len), dtype=chunks[0].dtype).to(
            chunks[0].device
        )

        for i, chunk in enumerate(chunks):
            output[:, i * hop_size : i * hop_size + frame_size] += chunk

        return output

def _pad_segment(input, segment_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest

def split_feature(input, segment_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = _pad_segment(input, segment_size)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    segments1 = (
        input[:, :, :-segment_stride]
        .contiguous()
        .view(batch_size, dim, -1, segment_size)
    )
    segments2 = (
        input[:, :, segment_stride:]
        .contiguous()
        .view(batch_size, dim, -1, segment_size)
    )
    segments = (
        torch.cat([segments1, segments2], 3)
        .view(batch_size, dim, -1, segment_size)
        .transpose(2, 3)
    )

    return segments.contiguous(), rest

def merge_feature(input, rest):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, segment_size, _ = input.shape
    segment_stride = segment_size // 2
    input = (
        input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)
    )  # B, N, K, L

    input1 = (
        input[:, :, :, :segment_size]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, segment_stride:]
    )
    input2 = (
        input[:, :, :, segment_size:]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, :-segment_stride]
    )

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]

    return output.contiguous()  # B, N, T

def is_torch_complex_tensor(c):
    return not isinstance(c, ComplexTensor) and torch.is_complex(c)

def is_complex(c):
    return isinstance(c, ComplexTensor) or is_torch_complex_tensor(c)


def choose_norm(norm_type, channel_size, shape="BDT"):
    """The input of normalization will be (M, C, K), where M is batch size.

    C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size, shape=shape)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size, shape=shape)
    elif norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)
    elif norm_type == "GN":
        return nn.GroupNorm(1, channel_size, eps=1e-8)
    else:
        raise ValueError("Unsupported normalization type")


if __name__ == '__main__':
    # ConvDecoder测试
    decoder = ConvDecoder(
        channel=64,
        kernel_size=2,
        stride=1,
    )
    enc = torch.rand(1, 23999, 64)
    print(decoder(enc, torch.LongTensor([24000])))