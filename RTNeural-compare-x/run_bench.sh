#!/bin/bash

set -e

build_bench()
{
    rm -Rf build
    sed -i '' "5s/.*/set($1 ON CACHE BOOL \"Use RTNeural with this backend\" FORCE)/" CMakeLists.txt
    cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DRTNEURAL_ONLY=ON "$2"
    cmake --build build --parallel --config Release
}

run_bench()
{
    rm -f $1
    touch $1

    bench=./build/rtneural_compare_bench

    layers=("dense" "gru" "lstm" "conv1d")
    sizes=(4 8 16 32 64 128 256)

    for layer in "${layers[@]}"; do
        for size in "${sizes[@]}"; do
            length_seconds=4
            if [ $layer == "conv1d" ] && [ $size -gt 16 ]; then
              length_seconds=1
            fi
            $bench $layer $length_seconds $size | tee -a $1
        done
    done

    activations=("tanh" "relu" "sigmoid")
    for activation in "${activations[@]}"; do
        for size in "${sizes[@]}"; do
            length_seconds=32
            $bench $activation $length_seconds $size | tee -a $1
        done
    done
}

build_bench "RTNEURAL_EIGEN" "-DRTNEURAL_USE_APPROX=ON"
run_bench "results/bench_eigen_approx.txt"

build_bench "RTNEURAL_EIGEN" "-DRTNEURAL_USE_APPROX=OFF"
run_bench "results/bench_eigen.txt"

build_bench "RTNEURAL_STL" "-DRTNEURAL_USE_APPROX=ON"
run_bench "results/bench_stl_approx.txt"

build_bench "RTNEURAL_STL" "-DRTNEURAL_USE_APPROX=OFF"
run_bench "results/bench_stl.txt"

build_bench "RTNEURAL_XSIMD" "-DRTNEURAL_USE_APPROX=OFF"
run_bench "results/bench_xsimd.txt"
