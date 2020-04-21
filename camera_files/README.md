# aviansolar

First set correct time zone:
<br>
sudo timedatectl set-timezone America/Chicago
<br><br>
Then install packages for exfat file system support (for SD card)
<br>
sudo apt-get install exfat-fuse exfat-utils
<br>
<br>
Copy gstreamer-record-splits.sh to /etc/init.d/
<br>
make script executable: 
<br>
sudo chmod +x /etc/init.d/gstreamer-record-splits.sh
<br>
add init.d simlinks: 
<br>
cd /etc/init.d
<br>
sudo update-rc.d gstreamer-record-splits.sh defaults 99 

