# octrees-benchmark

## Background

LiDAR (Light and Ranging Detection) technology has now become the quintessential technique for collecting geospatial 
data from the earth's surface. This code implements a linearized octree based on ideas from Keller et al. and Behley et al.
for fast fixed-radius neighbourhood searches.


Original project: https://gitlab.citius.usc.es/lidar/rule-based-classifier.
		
## Installation

### Dependencies
- LASTools:
    First we need the dependencies, listed at https://github.com/LAStools/LAStools:
    ```bash
    sudo apt-get install libjpeg62 libpng-dev libtiff-dev libjpeg-dev libz-dev libproj-dev liblzma-dev libjbig-dev libzstd-dev libgeotiff-dev libwebp-dev liblzma-dev libsqlite3-dev
    ```

    Now we clone the repo and build:
    ```bash
    git clone --depth 1 https://github.com/LAStools/LAStools lib/LAStools
    (cd lib/LAStools && cmake . && make)
- PCL version 1.15 (Optional) 
    Get 1.15 source code from  `https://github.com/PointCloudLibrary/pcl/releases` and build it. The folder were I installed it is `~/local/pcl`, but that can be changed to any other folder, with an appropiate change in `CMakeLibraries.cmake`. Can also change the version to look for in that file.
    ```bash
    wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.15.0/source.tar.gz
    tar xvf source.tar.gz
    cd pcl && mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local/pcl -DCMAKE_BUILD_TYPE=Release
    make -j2
    make -j2 install
    ```
    If PCL is not found during compilation, code will compile just fine, but without support por _neighborPCLKD_ or _neighborsPCLOct_ benchmarks.

### Compilation

In the project directory, just execute
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release .
  cmake --build build
  ```

This creates the executable at `build/octrees-benchmark`.

### Execution
## Main Options

| Option | Alias | Description |
|--------|-------|-------------|
| `-h` | `--help` | Show help message. |
| `-i` | – | Path to input file. |
| `-o` | – | Path to output file. |
| `-r` | `--radii` | Benchmark radii (comma-separated, e.g., `2.5,5.0,7.5`). |
| `-s` | `--searches` | Number of searches (random centers, unless `--sequential` is set), type `all` to search over the whole cloud (with sequential indexing).  |
| `-t` | `--repeats` | Number of repeats to do for each benchmark. |
| `-k` | `--kernels` | Specify which kernels to use (comma-separated or `all`). Possible values: `sphere`, `cube`, `square`, `circle`. |
| `-a` | `--search-algo` | Specify which search algorithms to run (comma-separated or `all`). Default: `neighborsPtr,neighbors,neighborsPrune,neighborsStruct`. <br> Possible values: <br> &nbsp;&nbsp;&bull; `neighborsPtr` – basic algorithm on pointer-based octree <br> &nbsp;&nbsp;&bull; `neighbors` – basic on linear octree <br> &nbsp;&nbsp;&bull; `neighborsPrune` – optimized linear octree search with octant pruning <br> &nbsp;&nbsp;&bull; `neighborsStruct` – optimized using index ranges <br> &nbsp;&nbsp;&bull; `neighborsApprox` – approximate search (will do both upper and lower bounds), specify `--approx-tol` with it <br> &nbsp;&nbsp;&bull; `neighborsUnibn` – unibnOctree search <br> &nbsp;&nbsp;&bull; `neighborsPCLKD` – PCL KD-tree search (if available) <br> &nbsp;&nbsp;&bull; `neighborsPCLOct` – PCL Octree search  (if available) |
| `-e` | `--encodings` | Select SFC encodings to reorder the cloud before the searches (comma-separated or `all`). Default: `all`. <br> Possible values: <br> &nbsp;&nbsp;&bull; `none` – no encoding, Linear Octree won't be built with it <br> &nbsp;&nbsp;&bull; `mort` – Morton SFC Reordering <br> &nbsp;&nbsp;&bull; `hilb` – Hilbert SFC Reordering |

## Other Options

| Option | Description |
|--------|-------------|
| `--debug` | Enable debug mode (measures octree build and encoding times). |
| `--check` | Enable result checking (old option, may not work, `avg_result_size` can be used to check everything is going fine). |
| `--no-warmup` | Disable warmup phase. |
| `--approx-tol` | Set tolerance for approximate searches (comma-separated e.g., `10.0,50.0,100.0`). |
| `--num-threads` | List of thread counts for scalability test (comma-separated e.g., `1,2,4,8,16,32`). By default, OMP chooses max threads only so we don't do an scalability test.|
| `--sequential` | Make the search set sequential instead of random, which runs way faster normally. If `-s all` is choosen, `--sequential` is automatically applied. |
| `--max-leaf` | Max number of points per octree leaf (default is `128`). Does not apply to PCL Octree. |
| `--pcl-oct-resolution` | Min octant size for subdivision in PCL Octree. |

## Authorship
Grupo de Arquitectura de Computadores (GAC)  
Centro Singular de Investigación en Tecnologías Inteligentes (CiTIUS)  
Universidad de Santiago de Compostela (USC)  

Linear octree and SFCs and benchmarks from:
- Pablo Díaz Viñambres ([pablo.diaz.vinambres@rai.usc.es](mailto:pablo.diaz.vinambres@rai.usc.es))

Original pointer-based implementation, readers and program structure from: 
- Miguel Yermo García ([miguel.yermo@usc.es](mailto:miguel.yermo@usc.es))
- Silvia Rodríguez Alcaraz ([silvia.alcaraz@usc.es](mailto:silvia.alcaraz@usc.es))
