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
 cog predict  -i prompt='C major, Roland TR-808, Korg KR-33,  celtic dance, violin, moderate, jazz, disco, rythmic, complex melody' -i duration=90 -i seed=1234 -i cfg_coef=3 -i temperature=1 
```

run with `-i burn_times=n` to burn first part with chroma n times, and use chroma of result 1st part  as chroma to all subsequent "continues"

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
