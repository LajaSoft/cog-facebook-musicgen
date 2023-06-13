# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
os.environ['LD_LIBRARY_PATH'] = '' #need to unset incompatible cuda libs
os.environ['XDG_CACHE_HOME']="/src/.container-cache" #need to chache locally

print(torch.version.cuda)
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = MusicGen.get_pretrained('medium')
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
		prompt:str = 'happy birthday polka',
        duration:int = 10
    ) -> Path:
        """Run a single prediction on the model"""
        descriptions = [prompt]

        self.model.set_generation_params(duration=duration)  # generate 8 seconds.
        wav = self.model.generate(descriptions)  # generates 3 samples.
        for idx, one_wav in enumerate(wav):
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write(f'{idx}', one_wav.cpu(), self.model.sample_rate, strategy="loudness")
                                                                                      
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
