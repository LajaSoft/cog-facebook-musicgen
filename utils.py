import ffmpeg
import json


# Fill received object to mp3 metadata
def convert_to_mp3(wav_path, mp3_path, data={}):
    metadata = {f'comment={json.dumps(data)}'}
    (
        ffmpeg
        .input(wav_path)
        .output(mp3_path, codec='libmp3lame', metadata=metadata)
        .run()
    )
