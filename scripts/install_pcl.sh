#!/bin/bash

# Exit on error
set -e

# Get the path of the script

# Libs prefix
LIB_PREFIX="$(dirname "$(readlink -f "$0")")/../lib"

# Get absolute path of LIB_PREFIX
LIB_PREFIX="$(readlink -f "${LIB_PREFIX}")"

BOOST_PREFIX="${LIB_PREFIX}/boost"
EIGEN3_PREFIX="${LIB_PREFIX}/eigen3"
FLANN_PREFIX="${LIB_PREFIX}/flann"
PCL_PREFIX="${LIB_PREFIX}/pcl"

# Build Boost from source 

BOOST_VERSION=1_82_0
BOOST_ARCHIVE="boost_${BOOST_VERSION}.tar.gz"
BOOST_DIR="boost_${BOOST_VERSION}"
wget -c "https://archives.boost.io/release/1.82.0/source/${BOOST_ARCHIVE}"
tar -xvzf "${BOOST_ARCHIVE}"
cd "${BOOST_DIR}"
./bootstrap.sh --prefix="${BOOST_PREFIX}"
./b2 install -j 4
cd ..

#  Build Eigen3 from source
wget -c https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip
mv eigen-3.4.0 eigen3_source

# Install Eigen3
mkdir -p eigen3_source/build
cd eigen3_source/build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${EIGEN3_PREFIX} ..
make install -j
cd ../..
rm -rf eigen3_source
rm eigen-3.4.0.zip

# Build FLANN from source
git clone https://github.com/flann-lib/flann ./flann_source

# Install FLANN
mkdir flann_source/build
cd flann_source/build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_MATLAB_BINDINGS=OFF -DCMAKE_INSTALL_PREFIX=${FLANN_PREFIX} ..
make install -j
cd ../..
rm -rf flann_source

# Build PCL 1.15 from source
wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.15.0/source.tar.gz
tar xvf source.tar.gz && rm source.tar.gz

mv pcl pcl_source

mkdir -p pcl_source/build
cd pcl_source/build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DBoost_ROOT="${BOOST_PREFIX}" \
      -DBoost_NO_SYSTEM_PATHS=ON \
      -DEigen3_DIR=${EIGEN3_PREFIX} \
      -DFLANN_ROOT=${FLANN_PREFIX} \
      -DBUILD_apps=OFF \
      -DBUILD_examples=OFF \
      -DBUILD_simulation=OFF \
      -DBUILD_tools=OFF \
      -DWITH_VTK=OFF \
      -DWITH_QT=OFF \
      -DWITH_PCAP=OFF \
      -DCMAKE_INSTALL_PREFIX=${PCL_PREFIX} ..
make -j 4
make install
cd ../..
rm -rf pcl_source