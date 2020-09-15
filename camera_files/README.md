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
Copy gstreamer-record-splits.sh to /usr/local/bin/
sudo cp gstreamer-record-splits.sh /usr/local/bin/.
sudo chmod 777 /usr/local/bin/gstreamer-record-splits.sh
<br>
add gstreamer to crontab
@reboot root /usr/local/bin/gstreamer-record-splits.sh
<br>
Copy bootcout.sh to /usr/local/bin
sudo cp bootcount.sh /usr/local/bin/.
sudo chmod 777 /usr/local/bin/bootcount.sh
<br>
Create the initial count file
nano ~/bcount.txt
Enter 0 and save
<br>
@reboot root /usr/local/bin/bootcount.sh
<br>
Install fake hardware clock
sudo apt-get install fake-hwclock













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



Copy gstreamer-record-splits.sh to /usr/local/bin/
sudo cp gstreamer-record-splits.sh /usr/local/bin/.
sudo chmod 744 /usr/local/bin/gstreamer-record-splits.sh

Copy gstreamer-record-splits.service to /etc/systemd/system/
sudo cp gstreamer-record-splits.service /etc/systemd/system/.
sudo chmod 664 /etc/systemd/system/gstreamer-record-splits.service

sudo systemctl daemon-reload
sudo systemctl enable gstreamer-record-splits.service



https://linuxconfig.org/how-to-run-script-on-startup-on-ubuntu-20-04-focal-fossa-server-desktop