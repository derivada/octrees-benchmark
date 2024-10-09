
[![pipeline status Master](https://gitlab.citius.usc.es/lidar/rule-based-classifier-cpp/badges/master/pipeline.svg)](https://gitlab.citius.usc.es/lidar/rule-based-classifier-cpp/commits/master)
[![pipeline status Dev](https://gitlab.citius.usc.es/lidar/rule-based-classifier-cpp/badges/dev/pipeline.svg)](https://gitlab.citius.usc.es/lidar/rule-based-classifier-cpp/commits/dev)

# rule-based-classifier: C++ version.

## Background

LiDAR (Light and Ranging Detection) technology has now become the quintessential technique for collecting geospatial 
data from the earth's surface. This code implements a method for automatic classification of objects with LiDAR data, 
with the aim of detecting ground, vegetation and buildings in the cloud of points.


Original project: https://gitlab.citius.usc.es/lidar/rule-based-classifier.
		
## Install

#### Cloning the project
```bash
git clone https://gitlab.citius.usc.es/lidar/rule-based-classifier-cpp.git
cd rule-based-classifier-cpp
```

#### Dependencies
- Eigen and Armadillo
  - Ubuntu
      ```bash
      sudo apt install libeigen3-dev libarmadillo-dev
      ```
  - ArchLinux
      ```bash
      sudo pacman -S eigen
      git clone https://aur.archlinux.org/armadillo.git lib/armadillo
      (cd armadillo && makepkg -si --noconfirm)
      ```
 
The following commands must be executed in the root folder of the project.

- LASTools:
    ```bash
    wget https://lastools.github.io/download/LAStools.zip && unzip LAStools.zip -d ./lib && rm LAStools.zip
    ```
- LASlib:
    ```bash
    (cd lib/LAStools && cmake . && make)
    ```

## Usage

There are two ways to compile the project:

##### Make

In the project directory, just execute

    make

An executable main is created in the same directory

##### CMakeLists

In the project directory, just execute
  ```bash
  mkdir build && cd build
  cmake -B build -DCMAKE_BUILD_TYPE=Release ..
  make
  ```

This creates the executable build/rule-based-classifier-cpp.

/!\ WARNING: If cmake is executed in the project directory, the already existing Makefile will be overwrited.


#### Execution
    ${path_to_binary_executable} -i data/ptR_18C.las [-o output_dir]

## Authorship
Grupo de Arquitectura de Computadores (GAC)  
Centro Singular de Investigación en Tecnologías Inteligentes (CiTIUS)  
Universidad de Santiago de Compostela (USC)  

Maintainers: 
- Miguel Yermo García ([miguel.yermo@usc.es](mailto:miguel.yermo@usc.es))
- Silvia Rodríguez Alcaraz ([silvia.alcaraz@usc.es](mailto:silvia.alcaraz@usc.es))