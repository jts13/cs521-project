# cs521-project

This repository contains the code used in our paper 'Improving Real Time Neural Network Inference Through Approximation'.

The repository contains:
* RTNeural-compare-x: a modified version of RTNeural-compare used to generate comparable charts for performance analysis
* RTNeural-harness-x: a simple 'runner' for generating audio samples by applying a given model (JSON file) to a given input signal (WAV file)
* RTNeural-train-x: experimental model training
* RTNeural-x: a modified version of RTNeural that includes various approximation techniques (enabled via CMake)
* rural: a Rust crate for neural-network-related experiments - includes implementations and measurements of the algorithms used in our paper

## Useful commands

```shell
cmake -B build
cmake --build build --config Release
```

```shell
cd RTNeural-compare-x
./run_bench.sh
python results/analyze_results.py
```
