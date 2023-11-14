# Basketball Shooting Map Generator




Basketball Shooting Map Generator can be used to generate a shooting map for basketball shooting fixed-camera video based on PyTorch YOLOv4 object  detection.

## 1. Environment

Install packages in requirement.txt

## 2. Weights Download

## darknet2pytorch

- google(https://drive.google.com/file/d/15waE6I1odd_cR3hKKpm1uXXE41s5q1ax)
- `mkdir pytorch_YOLOv4/weights/`
- download file `yolov4-basketball.weights` in the directory `pytorch_YOLOv4/weights/`

# 3. Use Basketball Shooting Map Generator

## 3.1 Prepare your basketball video

- put your basketball video in the directory `dataset/``

## 3.2 Run the demo

Change the name of the video input in python video_editor.py, and run

```sh
python video_editor.py
```
