datasets=(
  "data/paris_lille/Lille_0.las"
  "data/dales_las/test/5135_54435.las"
)
radii="0.5,1.0,2.0,3.0"

datasets_high_density=(
  "data/speulderbos/Speulderbos_2017_TLS.las"
  "data/semantic3d/bildstein_station1_xyz_intensity_rgb.txt" 
  "data/semantic3d/sg27_station8_intensity_rgb.txt"
)
radii_high_density="0.05,0.1,0.2,0.5"

# approx benchmark
for dataset in "${datasets[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/approx_search" -r "$radii" -b "approx" --approx-tol "5.0,10.0,25.0,50.0,100.0" -s 5000
done

for dataset in "${datasets_high_density[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/approx_search" -r "$radii_high_density" -b "approx" --approx-tol "5.0,10.0,25.0,50.0,100.0" -s 5000
done

# struct vs original benchmark
for dataset in "${datasets[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/struct_vs_vect" -r "$radii" -b "comp" -s 5000
done

for dataset in "${datasets_high_density[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/struct_vs_vect" -r "$radii_high_density" -b "comp" -s 5000
done

