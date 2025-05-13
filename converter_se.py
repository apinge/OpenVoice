import os
import torch
import openvino as ov
from pathlib import Path

core = ov.Core()

from openvoice.api import ToneColorConverter, OpenVoiceBaseClass
from openvoice.api import spectrogram_torch
import openvoice.se_extractor as se_extractor

pt_device = "cpu"

converter_suffix = Path("/home/gta/qiu/openvino_notebooks/notebooks/openvoice/converter")

tone_color_converter = ToneColorConverter(converter_suffix / "config.json", device=pt_device)
tone_color_converter.load_ckpt(converter_suffix / "checkpoint.pth")
print(f"ToneColorConverter version: {tone_color_converter.version}")

class OVOpenVoiceBase(torch.nn.Module):
    """
    Base class for both TTS and voice tone conversion model: constructor is same for both of them.
    """

    def __init__(self, voice_model: OpenVoiceBaseClass):
        super().__init__()
        self.voice_model = voice_model
        for par in voice_model.model.parameters():
            par.requires_grad = False

class OVOpenVoiceConverter(OVOpenVoiceBase):
    """
    Constructor of this class accepts ToneColorConverter object for voice tone conversion and wraps it's 'voice_conversion' method with forward.
    """

    def forward(self, y, y_lengths, sid_src, sid_tgt, tau):
        return self.voice_model.model.voice_conversion(y, y_lengths, sid_src, sid_tgt, tau)



def get_patched_voice_conversion(ov_model: ov.Model, device: str) -> callable:
    compiled_model = core.compile_model(ov_model, device)

    def voice_conversion_impl(y, y_lengths, sid_src, sid_tgt, tau):
        ov_output = compiled_model((y, y_lengths, sid_src, sid_tgt, tau))
        return (torch.tensor(ov_output[0]),)

    return voice_conversion_impl

ref_speaker_path = "./resources/demo_speaker1.mp3"
OUTPUT_DIR = Path("outputs/")
OUTPUT_DIR.mkdir(exist_ok=True)
en_resulting_voice_path = OUTPUT_DIR / "output_ov_en-newest_cloned.wav"
target_se  = tone_color_converter.extract_se([ref_speaker_path], se_save_path=None)
exit()

class OVFeatureExtractor(torch.nn.Module):
    def __init__(self, ref_enc):
        super().__init__()
        self.ref_enc = ref_enc
       

    def forward(self, y):
        """
        wraps the 'voice_conversion' method with forward.
        """
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
        g = self.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
        return g.detach()

obj = OVFeatureExtractor(tone_color_converter.model.ref_enc)

ov_model = ov.convert_model(
    obj,
     example_input = (torch.randn(1, 2048).float().to(pt_device))
)
ov.save_model(ov_model, './openvino_irs/extract_se.xml')