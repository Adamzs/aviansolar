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

VAR=`cat /data/bcount.txt`

echo $(($VAR + 1)) > /data/bcount.txt

bootnum=$VAR
today=`date +%Y_%m_%d_%H_%M_%S`
docker run --net=host --runtime nvidia --ipc=host -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /tmp/argus_socket:/tmp/argus_socket -v /dev/video0:/dev/video0 -v /data:/data -v mynfs:/data/mount/nfs --privileged=true -e DISPLAY=$DISPLAY -t -d drive-mount && gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=(int)3840, height=(int)2160, framerate=30/1, format=(string)NV12' ! nvtee ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)I420_10LE' ! omxh265enc preset-level=3 bitrate=28000000 ! splitmuxsink location=/data/var/lib/docker/volumes/mynfs/_data/video-$bootnum-%05d-$today.mkv max-size-time=300000000000 > /data/log.txt 2> /data/error-log.txt

#gst-launch-1.0 videotestsrc is-live=1 num-buffers=600 ! video/x-raw,width=3840,height=2160,framerate=60/1 ! nvvidconv ! 'video/x-raw(memory:NVMM), width=(int)3840, height=(int)2160, format=(string)NV12' ! nvv4l2h264enc maxperf-enable=1 bitrate=8000000 ! h264parse ! qtmux ! filesink location=/data/var/lib/docker/volumes/mynfs/_data/filename_h264.mp4 -e

