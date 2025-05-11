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


ov_voice_convertion = Path("./openvino_irs/openvoice2_tone_conversion.xml")
tone_color_converter.model.voice_conversion = get_patched_voice_conversion(ov_voice_convertion , "CPU")
zh_source_se = torch.load("/home/gta/qiu/openvino_notebooks/notebooks/openvoice/base_speakers/ses/zh.pth")

OUTPUT_DIR = Path("outputs/")
OUTPUT_DIR.mkdir(exist_ok=True)
SOURCE_DIR = Path("source/")
ref_speaker_path = "./resources/demo_speaker1.mp3"
en_resulting_voice_path = OUTPUT_DIR / "output_ov_en-newest_cloned.wav"
zh_resulting_voice_path = OUTPUT_DIR / "output_ov_zh_cloned.wav"

zh_orig_voice_path = SOURCE_DIR / "output_ov_zh.wav"
en_orig_voice_path = SOURCE_DIR / "output_ov_en-newest.wav"
target_se = torch.load("./source/target_se.pt")
tone_color_converter.convert(
        audio_src_path=zh_orig_voice_path,
        src_se=zh_source_se,
        tgt_se=target_se,
        output_path=zh_resulting_voice_path,
        tau=0.3,
        message="@MyShell",
    )