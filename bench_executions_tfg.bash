# Some variables
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
datasets_build_enc=(
    "data/paris_lille/Lille_0.las"
    "data/dales_las/test/5080_54400.las"
    "data/semantic3d/sg27_station8_intensity_rgb.las"
    "data/speulderbos/Speulderbos_2017_TLS.las"
)

FOLDER="out_tfg"
THREADS="1,4,8,12,16,20,24,28,32,36,40"
KERNELS_3D="cube,sphere"
N_SEARCHES="5000"
TOLERANCES="5.0,10.0,25.0,50.0,75.0"
NUM_THREADS="1,4,8,12,16,20,24,28,32,36,40"

# Inicializacion
set -e
mkdir -p "$FOLDER"

# 1.1. Punteros (neighbors) vs lineal (neighborsV2)
# 1.2. lineal neighbors vs neighborsV2 vs neighborsStruct
# 1.3. punteros (neighbors) vs lineal (neighborsStruct)
# subset runs
echo "BENCHMARK 1 - SUBSET RUNS"
for data in "${datasets_low_density[@]}"; do
   ./build/rule-based-classifier-cpp --kernels "all" -i "$data" -o "$FOLDER/subset" -r "0.5,1.0,2.0,3.0" -s "$N_SEARCHES" --repeats 3 -a "neighbors,neighborsV2,neighborsPtr,neighborsStruct"
done
for data in "${datasets_high_density[@]}"; do
   ./build/rule-based-classifier-cpp --kernels "all" -i "$data" -o "$FOLDER/subset" -r "0.01,0.05,0.1,0.2" -s "$N_SEARCHES" --repeats 3 -a "neighbors,neighborsV2,neighborsPtr,neighborsStruct"
done

# full run bildstein
echo "BENCHMARK 1 - FULL RUNS"
# Full cloud searches
./build/rule-based-classifier-cpp --kernels "$KERNELS_3D" -i "data/semantic3d/bildstein_station1_xyz_intensity_rgb.las" -o "$FOLDER/full" -r "0.01,0.05" -s "all" --sequential --repeats 1 --no-warmup -a "neighbors,neighborsV2,neighborsPtr,neighborsStruct"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/dales_las/test/5080_54400.las" -o "$FOLDER/full" -r "10.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighbors,neighborsV2,neighborsPtr,neighborsStruct"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/full" -r "3.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighbors,neighborsV2,neighborsPtr,neighborsStruct"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Paris_Luxembourg_6.las" -o "$FOLDER/full" -r "3.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighbors,neighborsV2,neighborsPtr,neighborsStruct"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/semantic3d/sg27_station8_intensity_rgb.las" -o "$FOLDER/full" -r "0.01,0.05,0.1" -s "all" --sequential --repeats 1 --no-warmup -a "neighbors,neighborsV2,neighborsPtr,neighborsStruct"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/speulderbos/Speulderbos_2017_TLS.las" -o "$FOLDER/full" -r "0.05" -s "all" --sequential --repeats 1 --no-warmup -a "neighbors,neighborsV2,neighborsPtr,neighborsStruct"

# 4. approx searches
# Speulderbos subset
echo "BENCHMARK 4 - APPROX SEARCHES SPEULDERBOS"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/speulderbos/Speulderbos_2017_TLS.las" -o "$FOLDER/approx_subset" -r "0.05" -s "$N_SEARCHES" --repeats 3 -a "neighborsStruct,neighborsApprox" --approx-tol "$TOLERANCES"
# Lille full
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/approx_full" -r "3.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsStruct,neighborsApprox" --approx-tol "$TOLERANCES"

# 5. parallel eff
# Lille subset
echo "BENCHMARK 5 - PARALLEL SUBSET"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/parallel_subset" -r "0.25,0.5,1.0,2.0" -s "$N_SEARCHES" --repeats 3 -a "neighborsPtr,neighborsStruct" --num-threads "$NUM_THREADS"

# Lille full
echo "BENCHMARK 5 - PARALLEL FULL"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/parallel_full" -r "0.25,0.5,1.0,2.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPtr,neighborsStruct" --num-threads "$NUM_THREADS"

# 6. Build and encoding times
echo "BENCHMARK 6 - ENCODING AND BUILD TIMES"
for data in "${datasets_build_enc[@]}"; do
    ./build/rule-based-classifier-cpp -i "$data" -o "$FOLDER/build_64" --debug --max-leaf 64 --repeats 3
    ./build/rule-based-classifier-cpp -i "$data" -o "$FOLDER/build_128" --debug --max-leaf 128 --repeats 3
done
