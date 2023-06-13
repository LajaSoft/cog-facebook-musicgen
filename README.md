# demo for [facebook musicgen](https://github.com/facebookresearch/audiocraft)

## features
* run in docker
* zero conf

## known bugs 
* output segfault in the end of execution
* lame code

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


```
cog predict -i prompt="Celtic dance, rythmic" -i duration=10  -i seed=10
```

to change model size, edit predict.py (sorry), default is "medium"

output wavs in `./out/` directory


**tested on windows+wsl+docker rtx3070 and linux+docker rtx4090**
