[![Raspberry PI 4 Build](https://github.com/abhilb/mukham/actions/workflows/pi4_build.yml/badge.svg)](https://github.com/abhilb/mukham/actions/workflows/pi4_build.yml)
<br/>
[![Linux Build](https://github.com/abhilb/mukham/actions/workflows/linux_build.yml/badge.svg)](https://github.com/abhilb/mukham/actions/workflows/linux_build.yml)
<br/>
[![Windows Build](https://github.com/abhilb/mukham/actions/workflows/windows_build.yml/badge.svg)](https://github.com/abhilb/mukham/actions/workflows/windows_build.yml)

```
    __  ___            __      __
   /  |/  /  __  __   / /__   / /_   ____ _   ____ ___
  / /|_/ /  / / / /  / //_/  / __ \ / __ `/  / __ `__ \
 / /  / /  / /_/ /  / ,<    / / / // /_/ /  / / / / / /
/_/  /_/   \__,_/  /_/|_|  /_/ /_/ \__,_/  /_/ /_/ /_/

```

A framework to test various face detetion and face landmarks models.
Supports the following models now:

### Face detection

| Model     | Status   |
| -----     | :------: |
| Blazeface | Working  |
| Tinyface  | WIP      |
| Ultraface | WIP      |
| Yunet     | WIP      |
| Faceboxes | WIP      |

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
    * Create a file sdl2-config.cmake in the SDL2_DIR

[1] Blog post on using libsdl2 with cmake. [url](https://trenki2.github.io/blog/2017/06/02/using-sdl2-with-cmake/)
