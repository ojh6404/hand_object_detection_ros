# hand_object_detection_ros 

ROS1 wrapper package for [hand_object_detector](https://github.com/ddshan/hand_object_detector.git).

https://github.com/ojh6404/tracking_ros/assets/54985442/f8a49814-2645-4b71-887e-1c8f02da5c38

## Setup

### Prerequisite
This package is build upon
- ROS1 (Noetic)
- docker and nvidia-container-toolkit (for environment safety and cuda build)

### Build package

#### on your workspace
It is better to use docker environment cause it needs specific cuda version and build environment. But you can build it directly if you want provided that you use `cuda-11.3` and `torch==1.12`. Instruction's are below.
```bash
mkdir -p ~/ros/catkin_ws/src && cd ~/ros/catkin_ws/src
git clone https://github.com/ojh6404/hand_object_detection_ros.git
cd ~/ros/catkin_ws/src/hand_object_detection_ros
./prepare.sh # install torch and build python submodules
cd ~/ros/catkin_ws && catkin b
```

#### using docker (Recommended)
Otherwise, you can build this package on docker environment.
```bash
git clone https://github.com/ojh6404/hand_object_detection_ros.git
cd hand_object_detection_ros
docker build -t hand_object_detection_ros .
```

## Usage
### 1. run directly
```bash
roslaunch hand_object_detection_ros sample.launch \
    input_image:=/kinect_head/rgb/image_rect_color \
    device:=cuda:0
```
### 2. using docker
You can run on docker by
```bash
./run_docker -host pr1040 -mount ./launch -name sample.launch \
    input_image:=/kinect_head/rgb/image_rect_color \
    device:=cuda:0
```
where
- `-host` : hostname like `pr1040` or `localhost`
- `-mount` : mount launch file directory for launch inside docker.
- `-name` : launch file name to run

### TODO
add rostest and docker build test.
