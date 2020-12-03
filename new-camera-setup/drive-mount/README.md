# camera-capture

This sample builds a container which captures a video using an NVIDIA nvarguscamerasrc gstreamer pipeline.

# Building the container

Run [build-docker.sh](build-docker.sh) to build the container.

# Running the container

Run [run-docker.sh](run-docker.sh) to run the container.  
A 10 second video capture named `test.mp4` will be placed in the `/data/` directory by the running container.