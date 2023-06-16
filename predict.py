
# GPT HINTS:
# cog documentation: https://github.com/replicate/cog/blob/main/docs/python.md
# audiocraft source: https://github.com/facebookresearch/audiocraft/tree/main/audiocraft


from cog import BasePredictor, Input, Path

import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import random
from typing import Iterator
import calendar
import time
import json

def load_config():
    if os.path.exists('musicgen_config.json'):
        with open('musicgen_config.json', 'r') as f:
            config = json.load(f)
        return config
    return {}
    

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
    bar_len = 20  # Length of the progress bar
    filled_len = int(bar_len * generated_tokens / total_tokens)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    progress = f"[{bar}] {generated_tokens}/{total_tokens}"
    print(f"r{progress}", end="\r")

# TODO make class here, reduce args calls
def continue_melody(model, prompt, rate: int, descriptions = [], duration = PART_LEN, seed = random.randint(0, 1000000000)):
    torch.manual_seed(seed)
    # print current torch seed
    print("seed:", torch.seed())
    model.set_generation_params(duration=duration)
    part = model.generate_continuation(prompt=prompt, prompt_sample_rate = rate,
                              descriptions = descriptions,
                              progress = False)
    return part


#def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
#                             melody_sample_rate: int, progress: bool = False) -> torch.Tensor:
def chroma_burn(model, prompt, rate: int, descriptions = [], duration = PART_LEN, seed = random.randint(0, 1000000000), steps = 1) -> torch.Tensor:
    current_melody = prompt
    for i in range(steps):
        print ("burning step", i)
        torch.manual_seed(seed)
        model.set_generation_params(duration=duration)
        current_melody = model.generate_with_chroma(
            descriptions=descriptions,
            melody_wavs=current_melody,
            melody_sample_rate=rate,
        ) 
    return current_melody


class Predictor(BasePredictor):
    def setup(self):
        self.modelSize = MODEL_SIZE
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = MusicGen.get_pretrained(self.modelSize)
        self.model.set_custom_progress_callback(my_progress_callback)

    def predict(
        self,
		prompt:str = 'happy birthday polka',
        duration:int = 10,
        seed:int = random.randint(0, 1000000000),
        burn_times:int = 0
    ) -> Path:
        """Run a single prediction on the model"""
        total_duration = 0
        if (duration > PART_LEN):
            total_duration = duration
            duration = PART_LEN

        descriptions = [prompt]
        torch.manual_seed(seed)
        self.model.set_generation_params(duration=duration)  
        all_parts = self.model.generate(descriptions, progress = True)  # generates .
        if (burn_times > 0):
            all_parts = chroma_burn(self.model, all_parts, self.model.sample_rate, descriptions, duration, seed, steps = burn_times)
        current_duration = duration
        first_wav = all_parts[0]
        collected_parts = first_wav
        if total_duration:
            if TAIL_CUT_LEN > 0:
                print ("cutting tails", TAIL_CUT_LEN)
                collected_parts = collected_parts[..., : -TAIL_CUT_LEN * self.model.sample_rate]
            current_duration = current_duration = collected_parts.shape[1] / self.model.sample_rate
            last_wav = first_wav
            while round(current_duration) < total_duration:
                current_duration = collected_parts.shape[1] / self.model.sample_rate
                print ("current_duration", current_duration , '/', total_duration)
                part_to_continue = collected_parts[..., -int(PART_LEN * self.model.sample_rate):]
                part_to_continue = part_to_continue[..., -int(CUT_LEN * self.model.sample_rate):]
                part_to_continue_length = part_to_continue.shape[1]
                planned_duration = min(PART_LEN, total_duration - int(current_duration - part_to_continue_length))

                print ("part_to_continue duration", part_to_continue.shape[1] / self.model.sample_rate)
                print ("planned_duration", planned_duration)
                print ("part_to_continue", part_to_continue.shape)
                #all_parts[0] = part_to_continue
                last_wav = continue_melody(
                    model = self.model,
                    prompt = part_to_continue,
                    rate=self.model.sample_rate,
                    descriptions=descriptions,
                    duration = planned_duration,
                    seed = seed
                )[0]
                if (burn_times > 0):
                    last_wav = chroma_burn(self.model, last_wav, self.model.sample_rate, descriptions, planned_duration, seed, steps = burn_times)

                print ("last_wav before", last_wav.shape)
                print ("CUT LEN, rate", CUT_LEN, self.model.sample_rate)
                last_wav = last_wav[...,max(part_to_continue_length, 0):]
                if (planned_duration > PART_LEN and TAIL_CUT_LEN > 0):
                    last_wav = last_wav[..., : -TAIL_CUT_LEN * self.model.sample_rate]
                print ("last_wav", last_wav.shape)
                collected_parts = torch.cat([collected_parts, last_wav], dim = 1)
                current_duration = collected_parts.shape[1] / self.model.sample_rate
        else:
            total_duration = duration
        file_name = f'out/{prompt.strip().replace(" ", "_").replace(":","=")}-{self.modelSize}_{seed}-{int(collected_parts.shape[1]/self.model.sample_rate)}s';
        audio_write(
                file_name,
                collected_parts.cpu(), 
                self.model.sample_rate, 
                strategy="loudness",
                loudness_compressor=True
        )
        print ("written", file_name)
        return Path(f'{file_name}.wav') # not sure why Ineed this, it writes to current directory output.wav, but without this I taking segfaults and unicorn errors :)