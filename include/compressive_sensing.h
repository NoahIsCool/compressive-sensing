#pragma once
#include <torch/torch.h>

// Simple hello function for testing

// RVO means that this won't really get copied

namespace compressive_sensing {
    float mutual_coherence(const torch::Tensor& X);
    float welch_bound(const torch::Tensor& X);
    float rip(torch::Tensor A, float s, float epsilon=1e-12);
    torch::Tensor omp(const torch::Tensor& X,
                      const torch::Tensor& y,
                      int k=-1,
                      double tolerance=0,
                      int l_normalize=-1);

    // SVD-based least squares solver: solves min_x ||A x - b||
    torch::Tensor least_squares_svd(const torch::Tensor& A, const torch::Tensor& b);

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
