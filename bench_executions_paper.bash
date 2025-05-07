#### Benchmark executions for whole cloud neighbor finding, instead of a random subset
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
FOLDER="out_paper_v2"
RADII="0.5,1.0,2.0,3.0"
LOW_RADII="0.01,0.05,0.1"
THREADS="1,4,8,12,16,20,24,28,32,36,40"
KERNELS_3D="cube,sphere"
N_SEARCHES="5000"

# Subset searches
#for data in "${datasets_low_density[@]}"; do
#    # Random subset of centers executions, parallel only for this datasets. use repeats here for more robust measurements
#    ./build/rule-based-classifier-cpp --kernels "all" -i "$data" -o "$FOLDER/subset" -r "$RADII" -b "srch" -s "$N_SEARCHES" --repeats 2
#done
#for data in "${datasets_high_density[@]}"; do
#    # Random subsets of centers executions
#    ./build/rule-based-classifier-cpp --kernels "all" -i "$data" -o "$FOLDER/subset" -r "$LOW_RADII" -b "srch" -s "$N_SEARCHES" --repeats 1 --no-warmup
#done 

# Bildstein full searches
#./build/rule-based-classifier-cpp --kernels "$KERNELS_3D" -i "data/semantic3d/bildstein_station1_xyz_intensity_rgb.txt" -o "$FOLDER/full" -r "0.01,0.05,0.1" -b "srch" -s "all" --sequential --repeats 1 --no-warmup

# Other full searches
#./build/rule-based-classifier-cpp --kernels "sphere" -i "data/dales_las/test/5080_54400.las" -o "$FOLDER/full" -r "10.0" -b "srch" -s "all" --sequential --repeats 1 --no-warmup
#./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/full" -r "3.0" -b "srch" -s "all" --sequential --repeats 1 --no-warmup
#./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Paris_Luxembourg_6.las" -o "$FOLDER/full" -r "3.0" -b "srch" -s "all" --sequential --repeats 1 --no-warmup
#./build/rule-based-classifier-cpp --kernels "sphere" -i "data/semantic3d/sg27_station8_intensity_rgb.txt" -o "$FOLDER/full" -r "0.05" -b "srch" -s "all" --sequential --repeats 1 --no-warmup
#./build/rule-based-classifier-cpp --kernels "sphere" -i "data/speulderbos/Speulderbos_2017_TLS.las" -o "$FOLDER/full" -r "0.05" -b "srch" -s "all" --sequential --repeats 1 --no-warmup


# Parallel only for Lille_0, both random subset and full (with less radii/numthreads in the full executions)
#./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/parallel_subset" -r "0.25,0.5,1.0,2.0" -b "parallel" -s "$N_SEARCHES" --repeats 1 --no-warmup --num-threads "$THREADS"
# ./build/rule-based-classifier-cpp --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/parallel_full" -r "0.25,0.5,1.0,2.0" -b "parallel" -s "all" --repeats 1 --no-warmup --num-threads "$THREADS"

# Build and encoding times
datasets_build_enc=(
    "data/paris_lille/Lille_0.las"
    "data/dales_las/test/5080_54400.las"
    "data/semantic3d/sg27_station8_intensity_rgb.txt"
    "data/speulderbos/Speulderbos_2017_TLS.las"
)
for data in "${datasets_build_enc[@]}"; do
    ./build/rule-based-classifier-cpp -i "$data" -o "enc_build_times_64" -b "log" --max-leaf 64 --repeats 3
    ./build/rule-based-classifier-cpp -i "$data" -o "enc_build_times_128" -b "log" --max-leaf 128 --repeats 3
done
