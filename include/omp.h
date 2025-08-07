#pragma once
#include <torch/torch.h>

// Simple hello function for testing

// RVO means that this won't really get coppied
inline torch::Device get_device() {
    // Try with CUDA if available
    if (torch::cuda::is_available()) {
        return torch::Device("cuda");
    } else if (torch::mps::is_available()){
        return torch::Device("mps");
    } else {
        return torch::Device("cpu");
    }
}
// device can be: "cpu", "cuda", "mps", etc.
void hello(const torch::Device& device = torch::kCPU);

// Placeholder for future compressive sensing functionality
// Example function declarations to be implemented:
// torch::Tensor compress(const torch::Tensor& signal, const torch::Tensor& sensing_matrix);
// torch::Tensor reconstruct(const torch::Tensor& measurements, const torch::Tensor& sensing_matrix);