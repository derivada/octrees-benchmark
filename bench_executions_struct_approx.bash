datasets=(
    "data/paris_lille/Lille_0.las"
    "data/paris_lille/Paris_Luxembourg_6.las"
    "data/paris_lille/Lille_11.las"
    "data/dales_las/test/5135_54435.las"
    "data/dales_las/train/5110_54320.las"
)
radii="0.5,1.0,2.0,3.0"

datasets_high_density=(
    "data/speulderbos/Speulderbos_2017_TLS.las"
    "data/semantic3d/bildstein_station1_xyz_intensity_rgb.txt" 
    "data/semantic3d/sg27_station8_intensity_rgb.txt"
)

radii_high_density="0.05,0.1,0.2,0.5"

N_SEARCHES=5000

# Pointer vs Linear benchmark
for dataset in "${datasets[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/octree_comp" -r "$radii" -b "srch" -s $N_SEARCHES
done

for dataset in "${datasets_high_density[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/octree_comp" -r "$radii_high_density" -b "srch" -s $N_SEARCHES
done

# Point struct comparison benchmark
for dataset in "${datasets[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/point_comp" -r "$radii" -b "pt" -s $N_SEARCHES
done

for dataset in "${datasets_high_density[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/point_comp" -r "$radii_high_density" -b "pt" -s $N_SEARCHES
done

# Algorithm comparison benchmark
for dataset in "${datasets[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/algo_comp" -r "$radii" -b "comp" -s $N_SEARCHES
done

for dataset in "${datasets_high_density[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/algo_comp" -r "$radii_high_density" -b "comp" -s $N_SEARCHES
done


# Approximate searches benchmark
for dataset in "${datasets[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/approx_search" -r "$radii" -b "approx" -s $N_SEARCHES --approx-tol "5.0,10.0,25.0,50.0,100.0" 
done

for dataset in "${datasets_high_density[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/approx_search" -r "$radii_high_density" -b "approx" -s $N_SEARCHES --approx-tol "5.0,10.0,25.0,50.0,100.0" 
done
