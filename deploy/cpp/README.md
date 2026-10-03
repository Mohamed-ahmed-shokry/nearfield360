# NearField360 Embedded C++ Deployment Architecture

High-performance, lightweight C++ runtime wrapper for deploying NearField360 multi-camera surround perception on embedded automotive computing platforms (NVIDIA DRIVE AGX, Jetson Orin, and industrial x86_64/aarch64 compute units).

## Key Features

1. **TensorRT RAII Engine (`TensorRTEngine`):**
   - High-throughput asynchronous execution with CUDA stream management.
   - Support for FP32, FP16, and INT8 precision modes with zero-copy host pinned memory buffers.
   - Standard 256-byte alignment matching NVIDIA GPU DMA transfer requirements.
2. **SIMD Fisheye Preprocessor (`FisheyePreprocessor`):**
   - Multi-camera surround batching (4 cameras: Front, Mirror Left, Mirror Right, Rear).
   - Fast $HWC \to CHW$ planar transposition and per-channel mean/std normalization matching `InferenceConfig`.
3. **Perception Pipeline & BEV Projection (`PerceptionPipeline`):**
   - Direct unprojection from 2D fisheye pixel coordinates $(u, v)$ to the vehicle Bird's-Eye-View (BEV) metric plane $(x, y)$.

## Build Instructions

### Native / Host Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Cross-Compilation for NVIDIA Jetson Orin (aarch64 Linux)

```bash
mkdir -p build_jetson && cd build_jetson
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DTENSORRT_PATH=/usr/lib/aarch64-linux-gnu \
    -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

## Running the Benchmark

```bash
./nearfield360_benchmark --model models/detection.onnx --batch-size 4 --iterations 100 --precision fp16
```

To output machine-readable JSON:

```bash
./nearfield360_benchmark --model models/detection.onnx --batch-size 4 --json > report.json
```
