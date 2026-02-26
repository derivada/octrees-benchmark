# octrees-benchmark

## Background

LiDAR (Light and Ranging Detection) technology has now become the quintessential technique for collecting geospatial data from the earth's surface. This code implements a linearized octree based on ideas from Keller et al. and Behley et al. for fast fixed-radius neighbourhood searches, achieving better performance than other Octrees and KD-trees tested, such as nanoflann KD-tree, picoTree, PCL Octree and KD-Tree, and unibnOctree. We also analyze the performance of Morton and Hilbert Space Filling Curves (SFCs). SFC Reordering allows for faster searches and is essential for the construction of the linear Octree. Extensive benchmarking and result plotting code and scripts are also provided.
		
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
    (cd lib/LAStools && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/../LASlib . && cmake --build build -- -j && cmake --install build)
    rm -rf lib/LAStools
    ```
    
- PCL version 1.15 (Optional) 
    Get 1.15 source code from  `https://github.com/PointCloudLibrary/pcl/releases` and build it. The folder were we installed it is `~/local/pcl`, but that can be changed to any other folder, with an appropiate change in `CMakeLibraries.cmake`. Can also change the version to look for in that file.
    ```bash
    wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.15.0/source.tar.gz
    tar xvf source.tar.gz
    cd pcl && mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local/pcl -DCMAKE_BUILD_TYPE=Release
    make -j2
    make -j2 install
    ```
    If PCL is not found during compilation, code will compile just fine, but without support for PCL Octree and KD-Tree related benchmarks.

- PAPI (Optional, for cache profiling)
  ```bash
    wget https://github.com/icl-utk-edu/papi/releases/download/papi-7-2-0-t/papi-7.2.0.tar.gz -P lib
    tar xvf lib/papi-7.2.0.tar.gz -C lib && rm lib/papi-7.2.0.tar.gz && mv lib/papi-7.2.0 lib/papi/
    (cd lib/papi/src && ./configure --prefix=$(pwd)/.. && make -j && make install)
  ```

### Compilation

In the project directory, just execute
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release .
  cmake --build build
  ```

This creates the executable at `build/octrees-benchmark`.

### Execution
We provide scripts for replicating our results, check `bench_neighbors.bash` for kNN and fixed-radius neighbor searches, `bench_memory.bash` for the measurements of Octree and KD-tree sizes, and `bench_locality.bash` for benchmarks analyzing how SFCs improve locality.

Under the folder `plots` we include all the python scripts for generating the figures seen in the publication.

### Tests / Examples
A small usage example of the library features is provided under `tests/example.cpp`. It is automatically built alongside the main project. To run it, simply execute from the `build/` directory:
```bash
make test_library
./tests/test_library
```

## Main Options

| Option | Alias | Description |
| :--- | :--- | :--- |
| `-h` | `--help` | Show help message. |
| `-i` | `--input` | Path to input file. |
| `-c` | `--container-type` | Container type to use. Default: `AoS`. <br> Possible values: `SoA`, `AoS`. |
| `-o` | `--output` | Path to output file. |
| `-r` | `--radii` | Benchmark radii (comma-separated, e.g., `2.5,5.0,7.5`). |
| `-v` | `--kvalues` | kNN benchmark k's (comma-separated, e.g., `10,50,250,1000`). |
| `-s` | `--searches` | Number of searches (random centers, unless `--sequential` is set), type `all` to search over the whole cloud (with sequential indexing). |
| `-t` | `--repeats` | Number of repeats to do for each benchmark. |
| `-k` | `--kernels` | Specify which kernels to use (comma-separated or `all`). Possible values: `sphere`, `cube`, `square`, `circle`. |
| `-a` | `--search-algo` | Specify which search algorithms to run (comma-separated or `all`). Default: `neighborsPtr,neighbors,neighborsPrune,neighborsStruct`. <br> Possible values: <br> **Radius Search:** <br> &nbsp;&nbsp;&bull; `neighborsPtr` – basic search on pointer-based octree <br> &nbsp;&nbsp;&bull; `neighbors` – basic search on linear octree <br> &nbsp;&nbsp;&bull; `neighborsPrune` – optimized linear octree search with octant pruning <br> &nbsp;&nbsp;&bull; `neighborsStruct` – optimized linear search using index ranges <br> &nbsp;&nbsp;&bull; `neighborsApprox` – approximate search (upper/lower bounds), requires `--approx-tol` <br> &nbsp;&nbsp;&bull; `neighborsUnibn` – unibnOctree search <br> &nbsp;&nbsp;&bull; `neighborsPCLKD` – PCL KD-tree search (if available) <br> &nbsp;&nbsp;&bull; `neighborsPCLOct` – PCL Octree search (if available) <br> &nbsp;&nbsp;&bull; `neighborsPico` – PicoTree search <br> **KNN Search:** <br> &nbsp;&nbsp;&bull; `KNNV2` – linear octree KNN searches <br> &nbsp;&nbsp;&bull; `KNNNanoflann` – nanoflann KNN searches <br> &nbsp;&nbsp;&bull; `KNNPCLKD` – PCL KD-tree KNN search (if available) <br> &nbsp;&nbsp;&bull; `KNNPCLOCT` – PCL Octree KNN search (if available) <br> &nbsp;&nbsp;&bull; `KNNPico` – PicoTree KNN search |
| `-e` | `--encodings` | Select SFC encodings to reorder the cloud before the searches (comma-separated or `all`). Default: `all`. <br> Possible values: <br> &nbsp;&nbsp;&bull; `none` – no encoding, Linear Octree won't be built with it <br> &nbsp;&nbsp;&bull; `mort` – Morton SFC Reordering <br> &nbsp;&nbsp;&bull; `hilb` – Hilbert SFC Reordering |
| – | `--debug` | Enable debug mode (measures octree build and encoding times). |
| – | `--build-enc` | Run benchmarks for the encoding and build of selected structures (the ones with a representative on `-a` / `--search-algo`). |
| – | `--memory` | Run a simple benchmark for measuring the memory consumed by a structure for heap profiling. <br> Possible values: `ptrOct`, `linOct`, `unibnOct`, `nanoKD`, `pclOct`, `pclKD`, `picoTree`. |
| – | `--locality` | Run benchmarks for analyzing the locality of the point cloud after given reorderings. |
| – | `--cache-profiling` | Enable cache profiling during search algo executions using PAPI. |
| – | `--check` | Enable result checking (legacy option; use `avg_result_size` to verify correctness). |
| – | `--no-warmup` | Disable warmup phase. |
| – | `--approx-tol` | Tolerance values for approximate search (comma-separated e.g., `10.0,50.0,100.0`). |
| – | `--num-threads` | List of thread counts for scalability test (comma-separated e.g., `1,2,4,8,16,32`). If not specified, OpenMP defaults to maximum threads and no scalability test is run. |
| – | `--sequential` | Make the search set sequential instead of random (usually faster). Automatically set when `-s all` is used. |
| – | `--max-leaf` | Max number of points per octree leaf (default = 128). Does not apply to PCL Octree. |
| – | `--pcl-oct-resolution`| Min octant size for subdivision in PCL Octree. |

## Authorship
Grupo de Arquitectura de Computadores (GAC)  
Centro Singular de Investigación en Tecnologías Inteligentes (CiTIUS)  
Universidad de Santiago de Compostela (USC)  

Linear octree implementation, SFCs, benchmarking and plotting code from:
- Pablo Díaz Viñambres ([pablo.diaz.vinambres@rai.usc.es](mailto:pablo.diaz.vinambres@rai.usc.es))

Optimized search algorithm and vectorization of SFC encoders:
- Abel Rodríguez Calleja ([GitHub profile](https://github.com/Abel-Breaker))

Original pointer-based Octree, readers and program structure from: 
- Miguel Yermo García ([miguel.yermo@usc.es](mailto:miguel.yermo@usc.es))
- Silvia Rodríguez Alcaraz ([silvia.alcaraz@usc.es](mailto:silvia.alcaraz@usc.es))
