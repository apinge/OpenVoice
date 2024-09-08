import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

ckpt_converter = '/home/qiu/TTS_OV/OpenVoice/converter'
device = "cpu"
output_dir = 'outputs_v2'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

reference_speaker = 'ttsmaker-file-2024-9-7-10-11-35.mp3'
#reference_speaker = 'resources/example_reference.mp3' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

from melo.api import TTS

texts = {
    #'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。", 
    #'ZH': "爱奇艺智能助手专注于解答用户问题，如果您有任何问题，我们随时为您提供帮助"
    "ZH": "有些打击乐器不仅仅能产生节奏，还能作出旋律和合声的效果 "
}


src_path = f'{output_dir}/tmp.wav'

# Speed is adjustable
speed = 1.0

for language, text in texts.items():
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        
        source_se = torch.load(f'base_speakers/{speaker_key}.pth', map_location=device)
        model.tts_to_file(text, speaker_id, src_path, speed=speed*0.88)
        save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

        # Run the tone color converter
        
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
        