import os
import torch
import openvino as ov


core = ov.Core()

from openvoice.api import ToneColorConverter, OpenVoiceBaseClass
from openvoice.api import spectrogram_torch
import openvoice.se_extractor as se_extractor


class model(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, y):
        n_fft = 1024
        hop_size = 256
        win_size = 1024
        hann_window = torch.hann_window(win_size).to(
                dtype=torch.float32, device='cpu'
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        return spec
obj = model()
pt_device = "cpu"
inputs = {
    'y': torch.randn(1, 274412).float().to(pt_device),
    'n_fft':  torch.tensor([1024],dtype=torch.int64).to(pt_device),
    'sampling_rate': torch.tensor([22050],dtype=torch.int64).to(pt_device),
    'hop_size': torch.tensor([256],dtype=torch.int64).to(pt_device),
    'win_size': torch.tensor([1024],dtype=torch.int64).to(pt_device),
    #'center':  torch.tensor([False],dtype=torch.bool).to(pt_device),
}
example = (
     torch.randn(1, 274412).float().to(pt_device),
     torch.tensor([1024],dtype=torch.int64).to(pt_device),
     torch.tensor([22050],dtype=torch.int64).to(pt_device),
    torch.tensor([256],dtype=torch.int64).to(pt_device),
     torch.tensor([1024],dtype=torch.int64).to(pt_device),
    #'center':  torch.tensor([False],dtype=torch.bool).to(pt_device),
)



#ov_model = ov.convert_model(obj, example)
ov_model = ov.convert_model(
    obj,
     example_input = (torch.randn(1, 2048).float().to(pt_device))
)
ov.save_model(ov_model, './openvino_irs/spectrogram.xml')