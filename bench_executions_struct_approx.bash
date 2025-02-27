datasets=(
  "data/paris_lille/Lille_0.las"
  "data/dales_las/test/5135_54435.las"
)

# approx benchmark
# for dataset in "${datasets[@]}"; do
#     if [[ ! -f "$dataset" ]]; then
#         echo "Error: File not found - $dataset"
#         exit 1
#     fi
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out/approx_search" -r "0.5,1.0,2.5,5.0" -b "approx" --approx-tol "5.0,10.0,25.0,50.0,100.0"
# done


# struct vs original benchmark
for dataset in "${datasets[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Error: File not found - $dataset"
        exit 1
    fi
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/struct_vs_vect" -r "0.5,1.0,2.5,5.0" -b "struct"
done

