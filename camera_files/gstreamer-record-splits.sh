#!/bin/sh
### BEGIN INIT INFO
# Provides:       gstreamer camera recording
# Required-Start:    $local_fs $remote_fs $network $syslog $named
# Required-Stop:     $local_fs $remote_fs $network $syslog $named
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: starts gstreamer-record-splits
# Description:       starts gstreamer-record-splits using start-stop-daemon
### END INIT INFO

today=`date +%Y_%m_%d_%H_%M_%S`
sleep 1m
mount 10.0.0.86:/media/usb /media && gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=(int)3840, height=(int)2160, framerate=30/1, format=(string)NV12' ! nvtee ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)I420_10LE' ! omxh265enc preset-level=3 bitrate=8000000 ! splitmuxsink location=/media/video-%05d-$today.mkv max-size-time=300000000000 > /home/boulderai/log.txt 2> /home/boulderai/error-log.txt
