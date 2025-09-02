#pragma once
#include <torch/torch.h>

// Simple hello function for testing

// RVO means that this won't really get copied

namespace omp {
    float mutual_coherence(const torch::Tensor& X);
    float welch_bound(const torch::Tensor& X);
    float rip(torch::Tensor A, float s, float epsilon=1e-12);

    inline torch::Device get_device() {
        if (torch::cuda::is_available()) {
            return {"cuda"};
        }
        if (torch::mps::is_available()) {
            return {"mps"};
        }
        return {"cpu"};
    }

}
