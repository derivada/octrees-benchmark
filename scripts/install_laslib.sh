#!/bin/bash

set -e

LIB_PREFIX="$(dirname "$(readlink -f "$0")")/../lib"

# Get absolute path of LIB_PREFIX
LIB_PREFIX="$(readlink -f "${LIB_PREFIX}")"

git clone --depth 1 https://github.com/LAStools/LAStools ${LIB_PREFIX}/LAStools
(cd "${LIB_PREFIX}"/LAStools && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${LIB_PREFIX}"/LASlib . && cmake --build build -- -j && cmake --install build)
rm -rf "${LIB_PREFIX}"/LAStools