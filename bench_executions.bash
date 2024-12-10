# In order of magnitude

# config 0.5, 1.0, 2.5, 5.0 radii
datasets_1=(
  "data/paris_lille/Lille_0.las"
  "data/paris_lille/Paris_Luxembourg_6.las"
  "data/paris_lille/Lille_11.las"
  "data/speulderbos/Speulderbos_2017_TLS.las"
)

# config 0.1, 0.2, 0.5, 1.0 radii
datasets_2=(
    "data/semantic3d/bildstein_station1_xyz_intensity_rgb.txt" 
    "data/semantic3d/station1_xyz_intensity_rgb.txt"
    "data/semantic3d/sg27_station8_intensity_rgb.txt"
)

# config 2.5, 5.0, 7.5, 10.0 radii
datasets_3=(
    "data/dales_las/test/5135_54435.las"
    "data/dales_las/train/5110_54320.las"
)

for dataset in "${datasets_1[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset"
done
