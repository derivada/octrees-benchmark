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

structures=(ptrOct linOct nanoKD pclOct pclKD uniOct)

if ! command -v heaptrack &> /dev/null; then
    echo "Heaptrack not found. Please install it first."
    exit 1
fi

# setup main dir
BIN="./build/octrees-benchmark"
OUTDIR="heaptrack"
mkdir -p "$OUTDIR"

# iterate over datasets and structures
echo "running memory benchmarks"
for dataset in "${datasets[@]}"; do
    # extract dataset name and dir
    cloud_name=$(basename "$dataset" .las)
    cloud_dir=$(dirname "$dataset")

    echo "  running dataset: $cloud_name"
    # setup dataset dir
    dataset_outdir="$OUTDIR/$cloud_name"
    mkdir -p "$dataset_outdir"

    for structure in "${structures[@]}"; do
        echo "    running structure: $structure"

        # output prefix for heaptrack
        out_prefix="${dataset_outdir}/${structure}"

        # run heaptrack with the correct parameters
        heaptrack --output "${out_prefix}.heaptrack" \
            "$BIN" -i "$dataset" --memory "$structure"

        echo "    done: output saved to ${out_prefix}.heaptrack"
        echo
    done
    echo "  done with dataset $cloud_name"
done
echo "done"