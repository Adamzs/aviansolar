#!/bin/bash
mkdir -p /data
docker run --net=host --runtime nvidia --ipc=host -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /tmp/argus_socket:/tmp/argus_socket -v /dev/video0:/dev/video0 -v /data:/data -v mynfs:/data/mount/nfs --privileged=true -e DISPLAY=$DISPLAY -t -d drive-mount
