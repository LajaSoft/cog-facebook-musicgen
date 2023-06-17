
# GPT HINTS:
# cog documentation: https://github.com/replicate/cog/blob/main/docs/python.md
# audiocraft source: https://github.com/facebookresearch/audiocraft/tree/main/audiocraft


from cog import BasePredictor, Input, Path

import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
import os
import random
from typing import Iterator, Tuple, Union
import typing as tp
import calendar
import time
import json
from utils import convert_to_mp3

def load_config():
    if os.path.exists('musicgen_config.json'):
        with open('musicgen_config.json', 'r') as f:
            config = json.load(f)
        return config
    return {}
    
MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]

config=load_config()

MODEL_SIZE = 'small' if config.get('MODEL_SIZE') is None else config['MODEL_SIZE']
# reduce to get less vram (may be), and may be to get better results
PART_LEN = 30 if config.get('PART_LEN') is None else config['PART_LEN']
# how much of last part to use for continuation (less this number - less glue quality, I guess. More - longer render, for sure) shall be less than PART_LEN!
CUT_LEN = 10 if config.get('CUT_LEN') is None else config['CUT_LEN']
# how much of prev track ending will be cut (to prevent fading in end of frame)
TAIL_CUT_LEN = 3 if config.get('TAIL_CUT_LEN') is None else config['TAIL_CUT_LEN']
print ('MODEL_SIZE', MODEL_SIZE, 'PART_LEN', PART_LEN, 'CUT_LEN', CUT_LEN, 'TAIL_CUT_LEN', TAIL_CUT_LEN)

os.environ['LD_LIBRARY_PATH'] = '' #need to unset incompatible cuda libs
os.environ['XDG_CACHE_HOME']="/src/.container-cache" #need to chache locally

print(torch.version.cuda)

def my_progress_callback(generated_tokens, total_tokens):
    if (generated_tokens % 10) != 0:
        return
    bar_len = 20  # Length of the progress bar
    filled_len = int(bar_len * generated_tokens / total_tokens)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    progress = f"[{bar}] {generated_tokens}/{total_tokens}"
    print(f"r{progress}", end="\r")

# TODO make class here, reduce args calls
def continue_melody(model, prompt, rate: int, descriptions = [], duration = PART_LEN, seed = random.randint(0, 1000000000), melody_wavs: MelodyType = None) -> torch.Tensor:
    #torch.manual_seed(seed)
    # print current torch seed
    print("seed:", torch.seed())
    model.set_generation_params(duration=duration)
    if prompt is None:
        return model.generate(descriptions=descriptions)
    part = generate_continuation_with_chroma(model, prompt=prompt, prompt_sample_rate = rate,
                              descriptions = descriptions, melody_wavs=melody_wavs,
                              progress = False)
    return part


#def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
#                             melody_sample_rate: int, progress: bool = False) -> torch.Tensor:
def chroma_burn(model, prompt, rate: int, descriptions = [], seed = random.randint(0, 1000000000), steps = 1) -> torch.Tensor:
    whole_melody = prompt
    whole_melody_duration = whole_melody.shape[1]
    for i in range(steps):
        audio_write(
                'out/_preview_tmp',
                whole_melody.cpu(), 
                model.sample_rate, 
                strategy="loudness",
                loudness_compressor=True
        )
        print ("burning step", i, 'with seed', seed)
        # if whole melody longer than part steps slice it and burn each part
        for seek in range(0, whole_melody_duration, PART_LEN * rate):
            print(seek/rate)
            torch.manual_seed(seed)
            planned_duration = min(PART_LEN, int((whole_melody_duration - seek) / rate))
            model.set_generation_params(duration=planned_duration)
            current_melody = whole_melody[:, seek: seek + planned_duration * rate]
            print("whole_melody", whole_melody.shape)
            print("current_melody", current_melody.shape)
            current_melody = model.generate_with_chroma(
                descriptions=descriptions,
                melody_wavs=current_melody,
                melody_sample_rate=rate,
            )
            # replace part of whole melody with current melody
            whole_melody[0, seek: seek + PART_LEN * rate] = current_melody.cpu()[0]
    print("whole_melody", whole_melody.shape, whole_melody.shape[1]/rate)
    return whole_melody

def generate_continuation_with_chroma(model, prompt: torch.Tensor, prompt_sample_rate: int,
                            descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None, melody_wavs: MelodyType = None,
                            progress: bool = False) -> torch.Tensor:
    """Generate samples conditioned on audio prompts.

    Args:
        prompt (torch.Tensor): A batch of waveforms used for continuation.
            Prompt should be [B, C, T], or [C, T] if only one sample is generated.
        prompt_sample_rate (int): Sampling rate of the given audio waveforms.
        descriptions (tp.List[str], optional): A list of strings used as text conditioning. Defaults to None.
        progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
    """
    if prompt.dim() == 2:
        prompt = prompt[None]
    if prompt.dim() != 3:
        raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
    prompt = convert_audio(prompt, prompt_sample_rate, model.sample_rate, model.audio_channels)
    if descriptions is None:
        descriptions = [None] * len(prompt)

    if melody_wavs is not None:
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, model.sample_rate, model.sample_rate, model.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
    print (descriptions, prompt.shape, melody_wavs)
    attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions = descriptions, prompt = prompt, melody_wavs = melody_wavs)
    assert prompt_tokens is not None
    return model._generate_tokens(attributes, prompt_tokens, progress)

class Predictor(BasePredictor):
    def setup(self):
        self.model_size = MODEL_SIZE
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = MusicGen.get_pretrained(self.model_size)
        self.model.set_custom_progress_callback(my_progress_callback)

    def predict(
        self,
		prompt:str = 'happy birthday polka',
        duration:int = 10,
        seed:int = random.randint(0, 1000000000),
        burn_times:int = 0,
        temperature:float = 1,
        top_k:int = 250,
        top_p:float = 0,
        cfg_coef:float = 3.0,
        use_sampling: bool = True
    ) ->Iterator[Path]:
        """Run a single prediction on the model"""
        total_duration = 0
        if (duration > PART_LEN):
            total_duration = duration
            duration = PART_LEN

        descriptions = [prompt]
        torch.manual_seed(seed)

        configuration_data = {
            'model_size': self.model_size,
            'seed': seed,
            'duration': total_duration,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'burn_times': burn_times,
            'description': prompt,
            'use_sampling': use_sampling
        }
        self.model.set_generation_params(duration=duration, temperature=temperature, top_k=top_k, top_p=top_p, cfg_coef=cfg_coef, use_sampling=use_sampling)
        collected_parts = None
        if (total_duration == 0):
            all_parts = self.model.generate(descriptions, progress = True)  # generates .
            current_duration = duration
            collected_parts = all_parts[0]
            chroma = chroma_burn( 
                model = self.model,
                prompt= collected_parts.cpu(),
                rate= self.model.sample_rate,
                descriptions= descriptions,
                seed= seed, 
                steps = burn_times
                )
            collected_parts = chroma.cpu()
        if total_duration:
            prev_layer = None
            for burn_step in range(1, burn_times + 1):
                current_duration = 0
                print ("burn_step", burn_step)
                collected_parts = torch.empty(1,0)
                seeds = {}
                if (prev_layer is not None):
                        audio_write(
                                'out/_preview_tmp',
                                prev_layer['wav'], 
                                self.model.sample_rate, 
                                strategy="loudness",
                                loudness_compressor=False
                        )
                while round(current_duration) < total_duration:
                    chroma_part = None
                    if (prev_layer is not None):
                        chroma_part = prev_layer['wav'][..., int(current_duration) * self.model.sample_rate :int(current_duration + PART_LEN) * self.model.sample_rate :].to(self.model.device)
                        torch.manual_seed(prev_layer['seeds'][current_duration])

                    seeds[current_duration] =  torch.seed()
                    print ("seeds", seeds)
                    current_duration = collected_parts.shape[1] / self.model.sample_rate
                    print ("current_duration", current_duration , '/', total_duration)
                    part_to_continue = collected_parts[..., -int(PART_LEN * self.model.sample_rate):].to(self.model.device)
                    part_to_continue = part_to_continue[..., -int(CUT_LEN * self.model.sample_rate):]
                    part_to_continue_length = part_to_continue.shape[1]
                    planned_duration = min(PART_LEN, total_duration - int(current_duration - part_to_continue_length))
                    if chroma_part is not None:
                        chroma_part = chroma_part[..., :int(planned_duration * self.model.sample_rate)]
                    print ("part_to_continue duration", part_to_continue.shape[1] / self.model.sample_rate)
                    if chroma_part is not None:
                        print ("chroma_part", chroma_part.shape, chroma_part.shape[1]/self.model.sample_rate)
                    print ("planned_duration", planned_duration)
                    print ("part_to_continue", part_to_continue.shape)
                    if part_to_continue_length == 0:
                        part_to_continue = None
                    last_wav = continue_melody(
                        model = self.model,
                        prompt = part_to_continue,
                        rate=self.model.sample_rate,
                        descriptions=descriptions,
                        duration = planned_duration,
                        seed = seed,
                        melody_wavs=chroma_part
                    )
                    last_wav = last_wav[0]
                    orig_last_wav = last_wav
                    print ("last_wav before", last_wav.shape)
                    print ("CUT LEN, rate", CUT_LEN, self.model.sample_rate)
                    last_wav = last_wav[...,max(part_to_continue_length, 0):]
                    if (planned_duration > PART_LEN and TAIL_CUT_LEN > 0):
                        last_wav = last_wav[..., : -TAIL_CUT_LEN * self.model.sample_rate]
                    print ("last_wav", last_wav.shape)
                    collected_parts = torch.cat([collected_parts, last_wav.cpu()], dim = 1)
                    current_duration = collected_parts.shape[1] / self.model.sample_rate
                    audio_write(
                        'out/_preview_part_tmp',
                        collected_parts.cpu(), 
                        self.model.sample_rate, 
                        strategy="loudness",
                        loudness_compressor=False
                    )
                prev_layer = {
                    'seeds': seeds,
                    'wav': collected_parts.cpu(),
                }
        else:
            total_duration = duration
        file_name = f'out/{prompt.strip().replace(" ", "_").replace(":","=")}-{seed}-{int(collected_parts.shape[1]/self.model.sample_rate)}s'
    
        if (burn_times > 0):
            file_name = f'{file_name}_burned={burn_times}'
        # add timestamp to filename
        file_name = f'{file_name}_{int(time.time())}'
        audio_write(
                file_name,
                collected_parts.cpu(), 
                self.model.sample_rate, 
                strategy="loudness",
                loudness_compressor=True
        )
        print ("written", file_name)
        convert_to_mp3(f'{file_name}.wav', f'{file_name}.mp3', configuration_data)
        #remove wav
        os.remove(f'{file_name}.wav')
        yield Path(f'{file_name}.mp3')