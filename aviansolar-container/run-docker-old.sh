#!/bin/bash
mkdir -p /data
docker run --net=host --runtime nvidia --ipc=host -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /tmp/argus_socket:/tmp/argus_socket -v /dev/video0:/dev/video0 -v /data:/data -v mynfs:/media --cap-add SYS_PTRACE --cap-add CAP_SYS_ADMIN -e DISPLAY=$DISPLAY -t -i as-camera-capture
