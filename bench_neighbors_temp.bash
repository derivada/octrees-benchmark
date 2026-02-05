# setup
FOLDER="out/results"
set -e
mkdir -p "$FOLDER"

NANO_PICO="neighborsNanoflann,neighborsPico"

# nano,pico runs on none,mort
./build/octrees-benchmark -i "data/dales_las/test/5080_54400.las" -o "$FOLDER/full" -r "5.0,10.0,15.0" -s "all" --sequential --repeats 1  -a "$NANO_PICO" -e "none,mort"
./build/octrees-benchmark -i "data/paris_lille/Lille_0.las" -o "$FOLDER/full" -r "0.5,1.5,3.0" -s "all" --sequential --repeats 1  -a "$NANO_PICO" -e "none,mort"
./build/octrees-benchmark -i "data/paris_lille/Paris_Luxembourg_6.las" -o "$FOLDER/full" -r "0.5,1.5,3.0" -s "all" --sequential --repeats 1  -a "$NANO_PICO" -e "none,mort"
./build/octrees-benchmark -i "data/speulderbos/Speulderbos_2017_TLS.las" -o "$FOLDER/full" -r "0.05,0.10,0.25" -s "all" --sequential --repeats 1  -a "$NANO_PICO" -e "none,mort"
./build/octrees-benchmark -i "data/semantic3d/sg27_station8_intensity_rgb.las" -o "$FOLDER/full" -r "0.025" -s "all" --sequential --repeats 1  -a "neighborsNanoflann" -e "all"
./build/octrees-benchmark -i "data/semantic3d/sg27_station8_intensity_rgb.las" -o "$FOLDER/full" -r "0.025" -s "all" --sequential --repeats 1  -a "neighborsPico" -e "none,mort"
./build/octrees-benchmark -i "data/semantic3d/bildstein_station1_xyz_intensity_rgb.las" -o "$FOLDER/full" -r "0.01,0.05,0.10" -s "all" --sequential --repeats 1  -a "$NANO_PICO" -e "none,mort"


./build/octrees-benchmark -i "data/paris_lille/Paris_Luxembourg_6.las" -o "$FOLDER/full" -r "0.5" -s "all" --sequential --repeats 1  -a "neighborsPCLKD" -e "all"
./build/octrees-benchmark -i "data/paris_lille/Paris_Luxembourg_6.las" -o "$FOLDER/full" -r "0.5,1.5" -s "all" --sequential --repeats 1  -a "neighbors,neighborsPrune,neighborsPtr,neighborsStruct,neighborsUnibn,neighborsPCLOct" -e "none,hilb"
./build/octrees-benchmark -i "data/speulderbos/Speulderbos_2017_TLS.las" -o "$FOLDER/full" -r "0.05" -s "all" --sequential --repeats 1  -a "neighborsUnibn,neighborsPCLOct" -e "all"
./build/octrees-benchmark -i "data/semantic3d/sg27_station8_intensity_rgb.las" -o "$FOLDER/full" -r "0.01" -s "all" --sequential --repeats 1  -a "neighborsPCLKD,neighborsUnibn,neighborsPCLOct,neighborsNanoflann,neighborsPico" -e "all"
