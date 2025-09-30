datasets=(
    "data/alcoy/alcoy.las"
    "data/paris_lille/Lille.las"
    "data/paris_lille/Lille_0.las"
    "data/paris_lille/Paris_Luxembourg_6.las"
    "data/paris_lille/Paris_Luxembourg.las"
    "data/semantic3d/bildstein_station1_xyz_intensity_rgb.las"
    "data/semantic3d/sg27_station8_intensity_rgb.las"
    "data/semantic3d/station1_xyz_intensity_rgb.las"
    "data/speulderbos/Speulderbos_2017_TLS.las"
    "data/dales_las/test/5080_54400.las"
    "data/dales_las/test/5140_54390.las"
    "data/dales_las/test/5150_54325.las"
)

for file in "${datasets[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Skipping file not found: $file"
        continue
    fi
    ./build/octrees-benchmark -i "$file" -o out/locality_knn/ -e none,mort,hilb --locality
done