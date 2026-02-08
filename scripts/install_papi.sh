#!/bin/bash

set -e

LIB_PREFIX="$(dirname "$(readlink -f "$0")")/../lib"

# Get absolute path of LIB_PREFIX
LIB_PREFIX="$(readlink -f "${LIB_PREFIX}")"

wget https://github.com/icl-utk-edu/papi/releases/download/papi-7-2-0-t/papi-7.2.0.tar.gz -P lib
tar xvf lib/papi-7.2.0.tar.gz -C lib && rm lib/papi-7.2.0.tar.gz && mv lib/papi-7.2.0 lib/papi/
(cd lib/papi/src && ./configure --prefix="${LIB_PREFIX}"/papi && make -j && make install)