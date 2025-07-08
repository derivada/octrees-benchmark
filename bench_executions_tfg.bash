# Datasets utilizados
datasets_low_density=(
    "data/paris_lille/Lille_0.las"
    "data/dales_las/test/5080_54400.las"
    "data/paris_lille/Paris_Luxembourg_6.las"
)
datasets_high_density=(
    "data/semantic3d/bildstein_station1_xyz_intensity_rgb.las" 
    "data/semantic3d/sg27_station8_intensity_rgb.las"
    "data/speulderbos/Speulderbos_2017_TLS.las"
)
datasets_build=(
    "data/paris_lille/Lille_0.las"
    "data/semantic3d/station1_xyz_intensity_rgb.las" 
    "data/speulderbos/Speulderbos_2017_TLS.las"
)

# Variables comúnes
FOLDER="out_tfg_v3"
THREADS="1,4,8,12,16,20,24,28,32,36,40"
KERNELS_3D="cube,sphere"
N_SEARCHES="10000"
TOLERANCES="5.0,10.0,25.0,50.0,75.0"
NUM_THREADS="1,4,8,12,16,20,24,28,32,36,40"
FULL_NO_PCLKD="neighbors,neighborsPrune,neighborsPtr,neighborsStruct"
FULL_ALGOS="neighbors,neighborsPrune,neighborsPtr,neighborsStruct,neighborsPCLKD,neighborsUnibn,neighborsPCLOct"
set -e
mkdir -p "$FOLDER"

# Búsquedas aleatorias
for data in "${datasets_low_density[@]}"; do
  ./build/octrees-benchmark --kernels "all" -i "$data" -o "$FOLDER/subset" -r "0.5,1.0,2.0,3.0" -s "$N_SEARCHES" --repeats 5 -a "neighborsNanoflann"
done
for data in "${datasets_high_density[@]}"; do
  ./build/octrees-benchmark --kernels "all" -i "$data" -o "$FOLDER/subset" -r "0.01,0.05,0.1,0.2" -s "$N_SEARCHES" --repeats 5 -a "neighborsNanoflann"
done

# Búsquedas completas
# ./build/octrees-benchmark --kernels "sphere" -i "data/semantic3d/bildstein_station1_xyz_intensity_rgb.las" -o "$FOLDER/full" -r "0.01,0.05,0.1" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPCLOct"
# ./build/octrees-benchmark --kernels "sphere" -i "data/dales_las/test/5080_54400.las" -o "$FOLDER/full" -r "15.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPCLOct"
# ./build/octrees-benchmark --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/full" -r "3.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPCLOct"
# ./build/octrees-benchmark --kernels "sphere" -i "data/paris_lille/Paris_Luxembourg_6.las" -o "$FOLDER/full" -r "3.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPCLOct"
# ./build/octrees-benchmark --kernels "sphere" -i "data/semantic3d/sg27_station8_intensity_rgb.las" -o "$FOLDER/full" -r "0.01,0.025,0.05" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPCLOct"
# ./build/octrees-benchmark --kernels "sphere" -i "data/speulderbos/Speulderbos_2017_TLS.las" -o "$FOLDER/full" -r "0.25" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPCLOct"

# # Búsquedas aproximadas
# ./build/octrees-benchmark --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/approx_full" -r "3.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsStruct,neighborsApprox" --approx-tol "$TOLERANCES"

# # Paralelismo
# ./build/octrees-benchmark --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/parallel_subset" -r "0.25,0.5,1.0,2.0" -s "$N_SEARCHES" --repeats 5 -a "neighborsPtr,neighborsStruct" --num-threads "$NUM_THREADS"
# ./build/octrees-benchmark --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/parallel_full" -r "0.25,0.5,1.0,2.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPtr,neighborsStruct" --num-threads "$NUM_THREADS"

# # Tiempo de codificación, reordenamiento y construcción de Octrees
# for data in "${datasets_build[@]}"; do
#     ./build/octrees-benchmark -i "$data" -o "$FOLDER/build_128" --debug --max-leaf 128 --repeats 3
#     ./build/octrees-benchmark -i "$data" -o "$FOLDER/build_64" --debug --max-leaf 64 --repeats 3
# done