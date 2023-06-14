# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

#MODELSIZE = 'large'
#MODELSIZE = 'medium'
#MODELSIZE = 'small'
MODELSIZE = 'melody'


from cog import BasePredictor, Input, Path

import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import random

import calendar;
import time;
PART_LEN = 25 # reduce to get less vram (may be), and may be to get better results
CUT_LEN = 20 # how much of last part to use for continuation (less this number - less glue quality, I guess. More - longer render, for sure) shall be less than PART_LEN!

os.environ['LD_LIBRARY_PATH'] = '' #need to unset incompatible cuda libs
os.environ['XDG_CACHE_HOME']="/src/.container-cache" #need to chache locally

print(torch.version.cuda)

def continue_melody(model, prompt, rate: int, descriptions = [], duration = PART_LEN, seed = random.randint(0, 1000000000)):
    # print current torch seed
    # torch.manual_seed(seed)
    print("seed:", torch.seed())
    model.set_generation_params(duration=duration)
    part = model.generate_continuation(prompt=prompt, prompt_sample_rate = rate,
                              descriptions = descriptions,
                              progress = True)
    return part


class Predictor(BasePredictor):
    def setup(self):
        self.modelSize = MODELSIZE
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = MusicGen.get_pretrained(self.modelSize)
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
		prompt:str = 'happy birthday polka',
        duration:int = 10,
        seed:int = random.randint(0, 1000000000)
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
        current_duration = duration
        first_wav = all_parts[0]
        collected_parts = first_wav
        if total_duration:
            current_duration = 0
            last_wav = first_wav
            while current_duration < total_duration:
                part_to_continue = collected_parts[..., -int(PART_LEN * self.model.sample_rate):]
                part_to_continue = part_to_continue[..., -int(CUT_LEN * self.model.sample_rate):]
                part_to_continue_length = part_to_continue.shape[1]

                print ("part_to_continue", part_to_continue.shape)
                #all_parts[0] = part_to_continue
                last_wav = continue_melody(
                    model = self.model,
                    prompt = part_to_continue,
                    rate=self.model.sample_rate,
                    descriptions=descriptions,
                    duration = min(PART_LEN, total_duration - current_duration + CUT_LEN),
                    seed = seed
                )[0]
                print ("last_wav before", last_wav.shape)
                print ("CUT LEN, rate", CUT_LEN, self.model.sample_rate)
                last_wav = last_wav[...,part_to_continue_length:]
                print ("last_wav", last_wav.shape)
                collected_parts = torch.cat([collected_parts, last_wav], dim = 1)
                current_duration = collected_parts.shape[1] / self.model.sample_rate
                print ("current_duration", current_duration)
        else:
            total_duration = duration
        audio_write(
                f'out/{prompt.strip().replace(" ", "_").replace(":","=")}-{self.modelSize}_{seed}-{int(collected_parts.shape[1]/self.model.sample_rate)}s',
                collected_parts.cpu(), 
                self.model.sample_rate, 
                strategy="loudness",
                loudness_compressor=True
        )

        # for i,part in all_parts:
        #     audio_write(
        #                 f'out/{prompt.strip().replace(" ", "_")}-{self.modelSize}_{seed}-{duration}s-{i}',
        #                 part.cpu(), 
        #                 self.model.sample_rate, 
        #                 strategy="loudness",
        #                 loudness_compressor=True
        #                 )
                                                                                      
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
