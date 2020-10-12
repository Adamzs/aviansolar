#!/bin/bash

gst-launch-1.0 nvarguscamerasrc num-buffers=300 ! \
 'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, \
 format=(string)NV12, framerate=(fraction)30/1' ! nvv4l2h265enc \
 control-rate=1 bitrate=8000000 ! 'video/x-h265, \
 stream-format=(string)byte-stream' ! h265parse ! qtmux ! filesink \
 location=/data/test.mp4 -e
