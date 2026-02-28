#!/bin/bash

set -e

LIB_PREFIX="$(dirname "$(readlink -f "$0")")/../lib"

# Get absolute path of LIB_PREFIX
LIB_PREFIX="$(readlink -f "${LIB_PREFIX}")"

git clone -b v1.0.0 https://github.com/Jaybro/pico_tree.git picotree_source
cd picotree_source
cmake -B build -DBUILD_TESTING=0 -DBUILD_EXAMPLES=0 -DCMAKE_INSTALL_PREFIX=${LIB_PREFIX}/picotree
cmake --build build -- -j4
cmake --install build
cd ..
rm -rf picotree_source