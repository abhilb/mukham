![example workflow](https://github.com/abhilb/mukham/actions/workflows/cmake.yml/badge.svg)


    __  ___            __      __                      
   /  |/  /  __  __   / /__   / /_   ____ _   ____ ___ 
  / /|_/ /  / / / /  / //_/  / __ \ / __ `/  / __ `__ \
 / /  / /  / /_/ /  / ,<    / / / // /_/ /  / / / / / /
/_/  /_/   \__,_/  /_/|_|  /_/ /_/ \__,_/  /_/ /_/ /_/ 
                                                       


A framework to test various face detetion and face landmarks models. 
Supports the following models now:

### Face detection

- Blazeface
- Tinyface
- Ultraface
- Yunet
- Faceboxes

### Face Landmarks

- Facemesh
- Dlib
![](assets/screenshot.gif)

## Build Instructions
### Windows
* OpenCV
    * Install OpenCV Version 4.5
    * Set the environmental variable OPENCV_DIR
    * Add the OPENCV_DIR/bin path to the PATH env variable
* SDL2
    * Install SDL2 library
    * Set the SDL2_DIR environmental variable
    * Add SDL2_DIR/bin to the PATH env variable
