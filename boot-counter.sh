#!/bin/sh
VAR=`cat /data/bai-edge-sdk/aviansolar-container/docker/bcount.txt`

echo $(($VAR + 1)) > /data/bai-edge-sdk/aviansolar-container/docker/bcount.txt
