#!/bin/bash
today=`date +%Y_%m_%d_%H_%M_%S`
for i in {1..1000}
do
  gst-launch-1.0 nvarguscamerasrc num-buffers=9000 ! 'video/x-raw(memory:NVMM), width=(int)3840, height=(int)2160, framerate=30/1, format=(string)NV12' ! nvtee ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)I420_10LE' ! omxh265enc preset-level=3 bitrate=8000000 ! matroskamux ! filesink location=/mnt/video$i-$today.mkv -e 
done
