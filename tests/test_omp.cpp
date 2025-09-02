#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <torch/torch.h>
#include "../include/omp.h"

// Test the hello function with CPU device
TEST(LibraryTest, mu) {
    
    auto X = torch::tensor({
            {0, 0},
            {1, 2},
            {3, 4},
        {5, 6}
                           }, torch::kFloat).transpose(0, 1);
    std::cout << X << std::endl;
    omp::mutual_coherence(X);
}
