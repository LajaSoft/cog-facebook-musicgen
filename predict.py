# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

#MODELSIZE = 'large'
MODELSIZE = 'medium'
#MODELSIZE = 'small'


from cog import BasePredictor, Input, Path

import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import random


import calendar;
import time;

os.environ['LD_LIBRARY_PATH'] = '' #need to unset incompatible cuda libs
os.environ['XDG_CACHE_HOME']="/src/.container-cache" #need to chache locally

print(torch.version.cuda)
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
        descriptions = [prompt]
        torch.manual_seed(seed)
        self.model.set_generation_params(duration=duration)  
        wav = self.model.generate(descriptions, progress = True)  # generates .
        one_wav = wav[0]
        ts = calendar.timegm(time.gmtime())
        audio_write(f'out/{prompt.strip().replace(" ", "_")}-{self.modelSize}_{seed}-{duration}s', one_wav.cpu(), self.model.sample_rate, strategy="loudness")
                                                                                      
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
