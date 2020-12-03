ARG L4TBASE_VERSION="r32.4.3"
FROM nvcr.io/nvidia/l4t-base:${L4TBASE_VERSION}

# Needed for tzdata related installs
ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Denver"
# Install gstreamer dependencies from
# https://developer.nvidia.com/embedded/dlc/l4t-accelerated-gstreamer-guide-32-1
RUN  apt-get update && \
     apt-get install -y \
     gstreamer1.0-tools \
     gstreamer1.0-alsa \
     gstreamer1.0-plugins-base \
     gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-bad \
     gstreamer1.0-plugins-ugly \
     gstreamer1.0-libav \
     libgstreamer1.0-dev \
     libgstreamer-plugins-base1.0-dev \
     libgstreamer-plugins-good1.0-dev \
     libgstreamer-plugins-bad1.0-dev \
     systemd \
     nfs-common \
     vim

# Create a directory to hold our app
RUN mkdir -p /usr/src/app
#RUN mkdir -p /media

# Add our ap, in this case a simple script
# To the /usr/src/app directory
#ADD gstreamer-record-splits.sh /usr/src/app
#ADD bcount.txt /usr/src/app

WORKDIR /usr/src/app

#CMD [ "./gstreamer-record-splits.sh" ]
#CMD /bin/bash
