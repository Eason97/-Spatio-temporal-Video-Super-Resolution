# Spatio-temporal-Video-Super-Resolution
This is the implementation for space-time video super-resolution. We use[Vimeo 90K](http://toflow.csail.mit.edu/)  for training and testing. You can load the pre-trained weights to initialize the network parameters.

## Environment:

- Ubuntu: 18.04

- CUDA Version: 11.0 
- Python 3.8

## Dependencies:

Note that our proposed model used Deformable Convolution module. You need to compile DCN module first. You can follow the instruction of [DCNv2](https://github.com/CharlesShang/DCNv2) .

- torch==1.6.0
- torchvision==0.7.0
- NVIDIA GPU and CUDA

## Pretrained Weights 

Download [optical flow weights](https://drive.google.com/file/d/1RPzbEkU3rtvMxEW4xPXMyXiSlxJteEYL/view?usp=sharing) and [space-time video super-resolution weights](https://drive.google.com/file/d/1tE3nWiOKCoAluWDmMJSdHUSZqQrkqnrA/view?usp=sharing).
