# Keyframe extraction

## Installing Important Libraries
Git clone this repository to `extra` directory.
```
git clone https://github.com/soCzech/TransNetV2.git
```

```
|- extra 
   |- TransNetV2
```
Set up `ffmpeg` in this link for window https://www.gyan.dev/ffmpeg/builds/.
- Run [transnetv2.ipynb](./transnetv2.ipynb) to extract shots from videos.
- Run [cutframe.ipynb](./cutframe.ipynb) to extract keyframes from shots.

## Input directory:
```
|- AIC_video 
   |- Videos_L01
   |- Videos_L02
   |- ...
```

## Output directory:
```
|- SceneJSON
   |- L01
   |- L02
   |- ...
```