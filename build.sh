#!/usr/bin/env bash

MYPWD=$(pwd)

GCC=gcc
GXX=g++

BUILD_TYPE=RelWithDebInfo

if [ -n "$1" ]; then
BUILD_TYPE=$1
fi
CXX_MARCH=native

# https://stackoverflow.com/a/45181694
NUM_CORES=`getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu || echo 1`

# Don't use all (virtual) cores in an attempt to not freeze the system.
# Some students reported issues when running with all cores
# (might have rather been RAM issue though).
NUM_PARALLEL_BUILDS=$((NUM_CORES - 2 < 1 ? 1 : NUM_CORES - 2))

EIGEN_DIR="$MYPWD/Thirdparty/eigen"
COMMON_CMAKE_ARGS=(
    -DCMAKE_C_COMPILER=${GCC}
    -DCMAKE_CXX_COMPILER=${GXX}
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
    -DCMAKE_CXX_FLAGS="-march=$CXX_MARCH -O3 -Wno-deprecated-declarations -Wno-null-pointer-arithmetic -Wno-unknown-warning-option -Wno-unused-function" #  -Wno-int-in-bool-context
)

BUILD_PANGOLIN=Thirdparty/build-Pangolin
# Build Pangolin
rm -rf "$BUILD_PANGOLIN"
mkdir -p "$BUILD_PANGOLIN"
pushd "$BUILD_PANGOLIN"

cmake ../Pangolin "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_FIND_FRAMEWORK=LAST \
    -DEXPORT_Pangolin=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF\
    "-DEIGEN_INCLUDE_DIR=$EIGEN_DIR"

make -j$NUM_PARALLEL_BUILDS pangolin


popd




echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release\
    "-DEIGEN_INCLUDE_DIR=$EIGEN_DIR"

make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release\
    "-DEIGEN_INCLUDE_DIR=$EIGEN_DIR"
make -j

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..



# echo "Configuring and building ORB_SLAM2 ..."


# mkdir build
# cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j8
