// Copyright 2026 NearField360 Contributors
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "nearfield360/fisheye_preprocessor.hpp"
#include "nearfield360/perception_pipeline.hpp"
#include "nearfield360/tensorrt_engine.hpp"

using namespace nearfield360;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "Options:\n"
              << "  --model <path>       Path to ONNX model or TensorRT engine plan (required)\n"
              << "  --batch-size <N>     Evaluation batch size (default: 4)\n"
              << "  --iterations <N>     Number of benchmark iterations (default: 100)\n"
              << "  --precision <mode>   Precision mode: fp32, fp16, int8 (default: fp16)\n"
              << "  --json               Output machine-readable JSON metrics\n"
              << "  --help               Display this help message\n";
}

int main(int argc, char* argv[]) {
    std::string model_path = "";
    int batch_size = 4;
    int iterations = 100;
    PrecisionMode precision = PrecisionMode::FP16;
    bool json_output = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (arg == "--precision" && i + 1 < argc) {
            std::string p = argv[++i];
            if (p == "fp32") precision = PrecisionMode::FP32;
            else if (p == "int8") precision = PrecisionMode::INT8;
            else precision = PrecisionMode::FP16;
        } else if (arg == "--json") {
            json_output = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model argument is required.\n";
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::cout << "[INFO] Initializing TensorRT Engine from: " << model_path << "\n";
        auto engine = std::make_shared<TensorRTEngine>(model_path, precision, batch_size);

        std::cout << "[INFO] Running benchmark (" << iterations << " iterations, batch size "
                  << batch_size << ")...\n";
        EngineProfileStats stats = engine->benchmark(iterations, batch_size);

        if (json_output) {
            std::cout << "{\n"
                      << "  \"model\": \"" << model_path << "\",\n"
                      << "  \"batch_size\": " << batch_size << ",\n"
                      << "  \"iterations\": " << stats.total_iterations << ",\n"
                      << "  \"mean_latency_ms\": " << stats.mean_latency_ms << ",\n"
                      << "  \"median_latency_ms\": " << stats.median_latency_ms << ",\n"
                      << "  \"p95_latency_ms\": " << stats.p95_latency_ms << ",\n"
                      << "  \"p99_latency_ms\": " << stats.p99_latency_ms << ",\n"
                      << "  \"fps\": " << stats.fps << "\n"
                      << "}\n";
        } else {
            std::cout << "\n=========================================\n"
                      << " NearField360 TensorRT Embedded Benchmark\n"
                      << "=========================================\n"
                      << " Batch Size:         " << batch_size << "\n"
                      << " Timed Iterations:   " << stats.total_iterations << "\n"
                      << " Mean Latency:       " << std::fixed << std::setprecision(2)
                      << stats.mean_latency_ms << " ms\n"
                      << " Median Latency:     " << stats.median_latency_ms << " ms\n"
                      << " 95th Percentile:    " << stats.p95_latency_ms << " ms\n"
                      << " 99th Percentile:    " << stats.p99_latency_ms << " ms\n"
                      << " Effective FPS:      " << stats.fps << " frames/sec\n"
                      << "=========================================\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "[ERROR] Execution failed: " << exc.what() << "\n";
        return 1;
    }

    return 0;
}
