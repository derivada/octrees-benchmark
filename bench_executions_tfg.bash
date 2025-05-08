# Some variables
datasets_low_density=(
    "data/paris_lille/Lille_0.las"
    "data/dales_las/test/5080_54400.las"
    "data/paris_lille/Paris_Luxembourg_6.las"
)
datasets_high_density=(
    "data/semantic3d/bildstein_station1_xyz_intensity_rgb.txt" 
    "data/semantic3d/sg27_station8_intensity_rgb.txt"
    "data/speulderbos/Speulderbos_2017_TLS.las"
)
FOLDER="out_tfg"
RADII="0.5,1.0,2.0,3.0"
LOW_RADII="0.01,0.05,0.1,0.2"
THREADS="1,4,8,12,16,20,24,28,32,36,40"
KERNELS_3D="cube,sphere"
N_SEARCHES="5000"
TOLERANCES="5.0,10.0,25.0,50.0,75.0"
NUM_THREADS="1,4,8,12,16,20,24,28,32,36,40"

# Inicializacion
set -e
mkdir -p "$FOLDER"

# 1. Punteros (neighbors) vs lineal (neighborsV2)
# subset runs
echo "BENCHMARK 1 - SUBSET RUNS"
for data in "${datasets_low_density[@]}"; do
   ./build/rule-based-classifier-cpp --kernels "all" -i "$data" -o "$FOLDER/subset_1" -r "$RADII" -s "$N_SEARCHES" --repeats 3 -a "neighborsV2,neighborsPtr,neighborsStruct,neighbors"
done
for data in "${datasets_high_density[@]}"; do
   ./build/rule-based-classifier-cpp --kernels "all" -i "$data" -o "$FOLDER/subset_1" -r "$LOW_RADII" -s "$N_SEARCHES" --repeats 3 -a "neighborsV2,neighborsPtr"
done

# full run bildstein
echo "BENCHMARK 1 - FULL RUNS"
./build/rule-based-classifier-cpp --kernels "$KERNELS_3D" -i "data/semantic3d/bildstein_station1_xyz_intensity_rgb.txt" -i "data/semantic3d/bildstein_station1_xyz_intensity_rgb.txt" -o "$FOLDER/full_1" -r "0.01,0.05" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsV2,neighborsPtr"

# Other full searches
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/dales_las/test/5080_54400.las" -o "$FOLDER/full_1" -r "10.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsV2,neighborsPtr,neighborsStruct"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/full_1" -r "3.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsV2,neighborsPtr,neighborsStruct,neighborsApprox" --tolerances "$TOLERANCES"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Paris_Luxembourg_6.las" -o "$FOLDER/full_1" -r "3.0" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsV2,neighborsPtr,neighborsStruct"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/semantic3d/sg27_station8_intensity_rgb.txt" -o "$FOLDER/full_1" -r "0.05" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsV2,neighborsPtr,neighborsStruct"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/speulderbos/Speulderbos_2017_TLS.las" -o "$FOLDER/full_1" -r "0.05" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsV2,neighborsPtr,neighborsStruct"


# 2. lineal neighbors vs neighborsV2 vs neighborsStruct
# subset runs already done at 1.
# full run sg27
echo "BENCHMARK 2 - FULL RUN SG27"
./build/rule-based-classifier-cpp --kernels "$KERNELS_3D" -i "data/semantic3d/sg27_station8_intensity_rgb.txt" -o "$FOLDER/full_2" -r "0.01,0.05" -s "all" --sequential --repeats 1 --no-warmup  "data/semantic3d/sg27_station8_intensity_rgb.txt"


# 3. punteros (neighbors) vs lineal (neighborsStruct)
# already did all

# 4. approx searches
# Speulderbos subset
echo "BENCHMARK 4 - APPROX SEARCHES SPEULDERBOS"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/speulderbos/Speulderbos_2017_TLS.las" -o "$FOLDER/subset_approx" -r "0.05" -s "$N_SEARCHES" --repeats 3 -a "neighborsStruct,neighborsApprox" --approx-tol "$TOLERANCES"
# Lille full already done at 1.

# 5. eff paralelismo
# subset
echo "BENCHMARK 5 - PARALLEL SUBSET"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/subset_parallel" -r "$RADII" -s "$N_SEARCHES" --repeats 3 -a "neighborsPtr,neighborsV2" --num-threads "$NUM_THREADS"

# full
echo "BENCHMARK 5 - PARALLEL FULL"
./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/subset_full" -r "$RADII" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPtr,neighborsV2" --num-threads "$NUM_THREADS"

# 6. Build and encoding times
# datasets_build_enc=(
#     "data/paris_lille/Lille_0.las"
#     "data/dales_las/test/5080_54400.las"
#     "data/semantic3d/sg27_station8_intensity_rgb.txt"
#     "data/speulderbos/Speulderbos_2017_TLS.las"
# )
# for data in "${datasets_build_enc[@]}"; do
#     ./build/rule-based-classifier-cpp -i "$data" -o "enc_build_times_64" -b "log" --max-leaf 64 --repeats 3
#     ./build/rule-based-classifier-cpp -i "$data" -o "enc_build_times_128" -b "log" --max-leaf 128 --repeats 3
# done