#!/bin/sh
VAR=`cat ~/bcount.txt`

echo $(($VAR + 1)) > ~/bcount.txt
