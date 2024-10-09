# rule-based-classifier: C++ version.

## Background

LiDAR (Light and Ranging Detection) technology has now become the quintessential technique for collecting geospatial 
data from the earth's surface. This code implements a method for automatic classification of objects with LiDAR data, 
with the aim of detecting ground, vegetation and buildings in the cloud of points.


Original project: https://gitlab.citius.usc.es/lidar/rule-based-classifier.
		
## Installation

### Dependencies
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
    git clone --depth 1 https://github.com/LAStools/LAStools lib/LAStools
    (cd lib/LAStools && cmake . && make)
    ```

### Compilation

In the project directory, just execute
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release .
  cmake --build build .
  ```

This creates the executable build/rule-based-classifier-cpp.

### Execution
    ./build/rule-based-classifier-cpp -i data/ptR_18C.las [-o output_dir]

## Authorship
Grupo de Arquitectura de Computadores (GAC)  
Centro Singular de Investigación en Tecnologías Inteligentes (CiTIUS)  
Universidad de Santiago de Compostela (USC)  

Maintainers: 
- Miguel Yermo García ([miguel.yermo@usc.es](mailto:miguel.yermo@usc.es))
- Silvia Rodríguez Alcaraz ([silvia.alcaraz@usc.es](mailto:silvia.alcaraz@usc.es))