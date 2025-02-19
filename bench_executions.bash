
# 0.5, 1.0, 2.5, 5.0 radii
datasets_search_1=(
  "data/paris_lille/Lille_0.las"
  "data/paris_lille/Paris_Luxembourg_6.las"
  "data/paris_lille/Lille_11.las"
  "data/speulderbos/Speulderbos_2017_TLS.las"
)

# 0.1, 0.2, 0.5, 1.0 radii
datasets_search_2=(
    "data/semantic3d/bildstein_station1_xyz_intensity_rgb.txt" 
    "data/semantic3d/station1_xyz_intensity_rgb.txt"
    "data/semantic3d/sg27_station8_intensity_rgb.txt"
)

# 1.0, 2.5, 5.0, 7.5, 10.0 radii
datasets_search_3=(
    "data/dales_las/test/5135_54435.las"
    "data/dales_las/train/5110_54320.las"
)

datasets_algo_comp=(
    "data/paris_lille/Lille_0.las"
    "data/paris_lille/Paris_Luxembourg_6.las"
    "data/dales_las/test/5135_54435.las"
)

# Some quick benchmarks with a single radius
# ./build/rule-based-classifier-cpp -i "data/speulderbos/Speulderbos_2017_TLS.las"                    -o "out" -r "1.0"    -b "srch" -t 3 
# ./build/rule-based-classifier-cpp -i "data/paris_lille/Lille_0.las"                    -o "out" -r "2.5"    -b "srch" -t 3 
# ./build/rule-based-classifier-cpp -i "data/dales_las/test/5135_54435.las"              -o "out" -r "10.0"   -b "srch" -t 3 
# ./build/rule-based-classifier-cpp -i "data/semantic3d/sg27_station8_intensity_rgb.txt" -o "out" -r "0.5"    -b "srch" -t 3 

##### REGULAR BENCHMARK
# for dataset in "${datasets_search_1[@]}"; do
#     if [[ ! -f "$dataset" ]]; then
#         echo "Error: File not found - $dataset"
#         exit 1
#     fi
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out" -r "0.5,1.0,2.5,5.0"
# done

# for dataset in "${datasets_search_2[@]}"; do
#     if [[ ! -f "$dataset" ]]; then
#         echo "Error: File not found - $dataset"
#         exit 1
#     fi
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out" -r "0.1,0.2,0.5,1.0"
# done

# for dataset in "${datasets_search_3[@]}"; do
#     if [[ ! -f "$dataset" ]]; then
#         echo "Error: File not found - $dataset"
#         exit 1
#     fi
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out" -r "1.0,2.5,5.0,7.5,10.0"
# done

#### IMPLEMENTATION COMPARISONS
# for dataset in "${datasets_algo_comp[@]}"; do
#     if [[ ! -f "$dataset" ]]; then
#         echo "Error: File not found - $dataset"
#         exit 1
#     fi
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out/algo_comp" -r "0.5,1.0,2.5,5.0" -b "comp"
# done


###### TIME PER POINT COMPARISONS
# For time per point comparisons, trying to get a closer avg_result_size to the other datasets
# for dataset in "${datasets_search_1[@]}"; do
#     if [[ ! -f "$dataset" ]]; then
#         echo "Error: File not found - $dataset"
#         exit 1
#     fi
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out/tpp_comp" -r "0.5,0.75,1.0"
# done

# for dataset in "${datasets_search_2[@]}"; do
#     if [[ ! -f "$dataset" ]]; then
#         echo "Error: File not found - $dataset"
#         exit 1
#     fi
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out/tpp_comp" -r "0.05, 0.1"
# done

# for dataset in "${datasets_search_3[@]}"; do
#     if [[ ! -f "$dataset" ]]; then
#         echo "Error: File not found - $dataset"
#         exit 1
#     fi
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out/tpp_comp" -r "10.0, 12.0"
# done

#### SEQ VS SHUFFLE COMPARISONS
# for dataset in "${datasets_seq[@]}"; do
#     if [[ ! -f "$dataset" ]]; then
#         echo "Error: File not found - $dataset"
#         exit 1
#     fi
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out/tpp_comp" -r "10.0,15.0" -s 1000 -t 1
# done

# for dataset in "${datasets_seq[@]}"; do
#     if [[ ! -f "$dataset" ]]; then
#         echo "Error: File not found - $dataset"
#         exit 1
#     fi
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out/seq_vs_shuffle" -b "seq" -r "0.5,1.0,2.5,5.0" -t 1 --no-warmup
# done

#### POINT TYPE COMPARISONS
# for dataset in "${datasets_search_1[@]}"; do
#     if [[ ! -f "$dataset" ]]; then
#         echo "Error: File not found - $dataset"
#         exit 1
#     fi
#     ./build/rule-based-classifier-cpp -i "$dataset" -b "pt" -o "out/point_comp" -r "0.5,1.0,2.5,5.0"
# done
