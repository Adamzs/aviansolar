# aviansolar

install packages:
<br>
sudo apt-get install exfat-fuse exfat-utils
<br>
Copy gstreamer-record-splits.sh to /etc/init.d/
<br>
make executable: sudo chmod +x gstreamer-record-splits.sh
<br>
add init.d simlinks: sudo update-rc.d gstreamer-record-splits.sh defaults 99 

