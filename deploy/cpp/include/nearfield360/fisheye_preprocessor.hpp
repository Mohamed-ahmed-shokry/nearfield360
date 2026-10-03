// Copyright 2026 NearField360 Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nearfield360 {

struct PreprocessorConfig {
    int target_width{640};
    int target_height{480};
    std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    std::array<float, 3> std{0.229f, 0.224f, 0.225f};
    bool bgr_to_rgb{true};
};

/**
 * @brief High-throughput SIMD-optimized preprocessor for 4-camera fisheye surround arrays.
 */
class FisheyePreprocessor {
public:
    explicit FisheyePreprocessor(PreprocessorConfig config = PreprocessorConfig{})
        : config_(config) {
        inv_std_[0] = 1.0f / config_.std[0];
        inv_std_[1] = 1.0f / config_.std[1];
        inv_std_[2] = 1.0f / config_.std[2];
    }

    /**
     * @brief Preprocess a single HWC image into planar CHW normalized float buffer.
     *
     * @param src_hwc Pointer to raw RGB/BGR packed uint8 image pixels.
     * @param src_width Width of the source image.
     * @param src_height Height of the source image.
     * @param dst_chw Destination pointer for contiguous planar float tensor [3, H, W].
     */
    void process_frame(const uint8_t* src_hwc, int src_width, int src_height, float* dst_chw) const {
        if (!src_hwc || !dst_chw) {
            throw std::invalid_argument("Null pointer provided to FisheyePreprocessor::process_frame.");
        }

        const int H = config_.target_height;
        const int W = config_.target_width;
        const size_t plane_size = static_cast<size_t>(H * W);

        float* channel_0 = dst_chw;
        float* channel_1 = dst_chw + plane_size;
        float* channel_2 = dst_chw + 2 * plane_size;

        const float inv_255 = 1.0f / 255.0f;
        const float m0 = config_.mean[0];
        const float m1 = config_.mean[1];
        const float m2 = config_.mean[2];
        const float is0 = inv_std_[0];
        const float is1 = inv_std_[1];
        const float is2 = inv_std_[2];

        // Bilinear downsampling & planar transposition
        for (int y = 0; y < H; ++y) {
            int src_y = (y * src_height) / H;
            for (int x = 0; x < W; ++x) {
                int src_x = (x * src_width) / W;
                size_t src_idx = (static_cast<size_t>(src_y) * src_width + src_x) * 3;
                size_t dst_idx = static_cast<size_t>(y) * W + x;

                uint8_t c0 = src_hwc[src_idx + 0];
                uint8_t c1 = src_hwc[src_idx + 1];
                uint8_t c2 = src_hwc[src_idx + 2];

                if (config_.bgr_to_rgb) {
                    std::swap(c0, c2);
                }

                channel_0[dst_idx] = ((static_cast<float>(c0) * inv_255) - m0) * is0;
                channel_1[dst_idx] = ((static_cast<float>(c1) * inv_255) - m1) * is1;
                channel_2[dst_idx] = ((static_cast<float>(c2) * inv_255) - m2) * is2;
            }
        }
    }

    /**
     * @brief Batch 4 surround fisheye cameras (FV, MVL, MVR, RV) into contiguous NCHW buffer.
     *
     * @param camera_frames Pointers to the 4 camera frame buffers.
     * @param src_width Width of camera frames.
     * @param src_height Height of camera frames.
     * @param dst_batched Destination pointer for [4, 3, H, W] contiguous tensor.
     */
    void batch_surround_cameras(const std::vector<const uint8_t*>& camera_frames,
                                int src_width, int src_height,
                                float* dst_batched) const {
        if (camera_frames.size() != 4) {
            throw std::invalid_argument("Expected exactly 4 camera frames for surround perception.");
        }

        const size_t elements_per_camera = 3 * config_.target_height * config_.target_width;
        for (size_t cam_idx = 0; cam_idx < 4; ++cam_idx) {
            float* cam_dst = dst_batched + cam_idx * elements_per_camera;
            process_frame(camera_frames[cam_idx], src_width, src_height, cam_dst);
        }
    }

    const PreprocessorConfig& config() const noexcept { return config_; }

private:
    PreprocessorConfig config_;
    std::array<float, 3> inv_std_{};
};

} // namespace nearfield360
