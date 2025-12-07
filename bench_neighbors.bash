# setup
FOLDER="out_paper_2"
set -e
mkdir -p "$FOLDER"

# datasets
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

# Random searches
N_SEARCHES="5000"
for data in "${datasets_low_density[@]}"; do
  ./build/octrees-benchmark -i "$data" -o "$FOLDER/subset" -r "0.5,1.0,2.0,3.0" -s "$N_SEARCHES" --repeats 5 -a "neighborsPico" -e "hilb"
done
for data in "${datasets_high_density[@]}"; do
  ./build/octrees-benchmark -i "$data" -o "$FOLDER/subset" -r "0.01,0.05,0.1,0.2" -s "$N_SEARCHES" --repeats 5 -a "neighborsPico" -e "hilb"
done

# Full searches
./build/octrees-benchmark -i "data/semantic3d/bildstein_station1_xyz_intensity_rgb.las" -o "$FOLDER/full" -r "0.01,0.05,0.1" -s "all" --sequential --repeats 1  -a "neighborsPico" -e "hilb"
./build/octrees-benchmark -i "data/dales_las/test/5080_54400.las" -o "$FOLDER/full" -r "5.0,10.0,15.0" -s "all" --sequential --repeats 1  -a "neighborsPico" -e "hilb"
./build/octrees-benchmark -i "data/paris_lille/Lille_0.las" -o "$FOLDER/full" -r "0.5,1.5,3.0" -s "all" --sequential --repeats 1  -a "neighborsPico" -e "hilb"
./build/octrees-benchmark -i "data/paris_lille/Paris_Luxembourg_6.las" -o "$FOLDER/full" -r "0.5,1.5,3.0" -s "all" --sequential --repeats 1  -a "neighborsPico" -e "hilb"
./build/octrees-benchmark -i "data/semantic3d/sg27_station8_intensity_rgb.las" -o "$FOLDER/full" -r "0.01,0.025,0.05" -s "all" --sequential --repeats 1  -a "neighborsPico" -e "hilb"
./build/octrees-benchmark -i "data/speulderbos/Speulderbos_2017_TLS.las" -o "$FOLDER/full" -r "0.05,0.10,0.25" -s "all" --sequential --repeats 1  -a "neighborsPico" -e "hilb"
# ./build/octrees-benchmark --kernels "cube,sphere" -i "data/semantic3d/sg27_station8_intensity_rgb.las"" -o "$FOLDER/full" -r "0.01,0.02,0.03,0.05" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPtr,neighbors" -e "all"


# knn full searches
datasets_knn=(
    "data/paris_lille/Paris_Luxembourg_6.las"
    "data/semantic3d/sg27_station8_intensity_rgb.las"
    "data/dales_las/test/5080_54400.las"
)
K_VALUES="5,10,25,50,75,100,200,300,400,500"

for data in  "${datasets_knn[@]}"; do
    ./build/octrees-benchmark -e "hilb" --kernels "all" -i "$data" -o "$FOLDER/knn_full" -v "$K_VALUES" -s "all" --repeats 1 --no-warmup --sequential -a "KNNPico"
done



# parallel subsets
for data in "${datasets_low_density[@]}"; do
    numactl --interleave=all ./build/octrees-benchmark -e "hilb" --kernels "sphere" -i "$data" -o "$FOLDER/parallel_subset" -s "$N_SEARCHES" --repeats 3 -a "neighborsPico" --num-threads "1,2,4,8,16,24,32,40" -r "0.1,0.25,0.5,1.0,2.0"
done
for data in "${datasets_high_density[@]}"; do
   numactl --interleave=all ./build/octrees-benchmark -e "hilb" --kernels "sphere" -i "$data" -o "$FOLDER/parallel_subset" -s "$N_SEARCHES" --repeats 3 -a "neighborsPico" --num-threads "1,2,4,8,16,24,32,40" -r "0.01,0.05,0.1,0.2"
done

# parallel full (slow)
numactl --interleave=all ./build/octrees-benchmark -e "none" --kernels "sphere" -i "data/paris_lille/Lille_0.las" -o "$FOLDER/parallel_full" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPico" --num-threads 1,2,4,8,16,24,32,40 -r "0.1,0.25,0.5,1.0,2.0"
numactl --interleave=all ./build/octrees-benchmark -e "none" --kernels "sphere" -i "data/dales_las/test/5080_54400.las" -o "$FOLDER/parallel_full" -s "all" --sequential --repeats 1 --no-warmup -a "neighborsPico" --num-threads 1,2,4,8,16,24,32,40 -r "0.1,0.25,0.5,1.0,2.0"
