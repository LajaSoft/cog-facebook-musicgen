# demo for [facebook musicgen](https://github.com/facebookresearch/audiocraft)

## features
* run in docker
* can continue itself as much as vram you have
* almost zero conf

## known bugs 
* output duplicates to output.wav
* lame code
* need to loadout generated parts to cpu, or may be even disk, for really long generations
* there seems problem in audiocraft library last version, progress shows  wrong total, 

## Installation:

* clone this repo
* install nvidia/stuff
* install docker
* install [cog](https://github.com/replicate/cog)
    ```
    sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
    sudo chmod +x /usr/local/bin/cog
    ```
*I think it will work in ~/bin/ too if PATH configured, but **I not tried***

## usage 
(all run in repo directory)

default prompt, random seed, 10s duration
```
cog predict
```

set prompt seed and duration:

# **RUN**
```
 cog predict  -i prompt='C major, Roland TR-808, Korg KR-33,  celtic dance, violin, moderate, jazz, disco, rythmic, complex melody' -i duration=90 -i seed=1234
```

run with `-i burn_times=n` (I tried with 4, it's amazing somehow) to burn out using chroma from audiocraf (works only with melody model, need almost twice VRAM, however just tested it on 8g VRAM 10s, seems we can)

---
## config
to change model size and other params, edit musicgen_config.json (i have .example)

## currently working model sizes:
 * small
 * medium
 * melody
 * large



output wavs in `./out/` directory


**tested on windows+wsl+docker rtx3070 and linux+docker rtx4090**
