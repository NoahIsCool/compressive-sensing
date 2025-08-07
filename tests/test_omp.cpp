#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <torch/torch.h>
#include "../include/omp.h"

// Test the hello function with CPU device
TEST(LibraryTest, HelloFunctionCPU) {
    // Test with CPU device (always available)

    ASSERT_NO_THROW(hello());

    // We can't easily test stdout output, but we can test that PyTorch is working properly
    // by performing similar operations here
    torch::Device cpu_device("cpu");
    torch::Tensor tensor = torch::rand({2, 3}, cpu_device);
    torch::Tensor matrix = torch::ones({3, 2}, cpu_device);
    torch::Tensor result = torch::matmul(tensor, matrix);

    // Check the dimensions of the result
    EXPECT_EQ(result.dim(), 2);
    EXPECT_EQ(result.size(0), 2);
    EXPECT_EQ(result.size(1), 2);
    EXPECT_EQ(result.device().type(), torch::kCPU);
}

// Test the hello function with CUDA if available
TEST(LibraryTest, HelloFunctionCUDA) {
    if (torch::cuda::is_available()) {
        ASSERT_NO_THROW(hello(torch::kCUDA));

        torch::Device cuda_device("cuda");
        torch::Tensor tensor = torch::rand({2, 3}, cuda_device);
        torch::Tensor matrix = torch::ones({3, 2}, cuda_device);
        torch::Tensor result = torch::matmul(tensor, matrix);

        EXPECT_EQ(result.dim(), 2);
        EXPECT_EQ(result.size(0), 2);
        EXPECT_EQ(result.size(1), 2);
        EXPECT_EQ(result.device().type(), torch::kCUDA);
    } else {
        GTEST_SKIP() << "CUDA is not available, skipping CUDA test";
    }
}

// Test the hello function with MPS if available (Apple Silicon)
TEST(LibraryTest, HelloFunctionMPS) {
    #ifdef USE_MPS
    if (torch::mps::is_available()) {
        ASSERT_NO_THROW(hello("mps"));

        torch::Device mps_device("mps");
        torch::Tensor tensor = torch::rand({2, 3}, mps_device);
        torch::Tensor matrix = torch::ones({3, 2}, mps_device);
        torch::Tensor result = torch::matmul(tensor, matrix);

        EXPECT_EQ(result.dim(), 2);
        EXPECT_EQ(result.size(0), 2);
        EXPECT_EQ(result.size(1), 2);
        EXPECT_EQ(result.device().type(), torch::kMPS);
    } else {
        GTEST_SKIP() << "MPS is not available, skipping MPS test";
    }
    #else
    GTEST_SKIP() << "MPS support not enabled, skipping MPS test";
    #endif
}

// Test basic PyTorch functionality
TEST(TorchTest, TensorCreation) {
    // Create a PyTorch tensor
    torch::Tensor tensor = torch::ones({2, 3});

    // Check tensor dimensions
    EXPECT_EQ(tensor.dim(), 2);
    EXPECT_EQ(tensor.size(0), 2);
    EXPECT_EQ(tensor.size(1), 3);

    // Check tensor values
    for (int i = 0; i < tensor.size(0); ++i) {
        for (int j = 0; j < tensor.size(1); ++j) {
            EXPECT_EQ(tensor[i][j].item<float>(), 1.0f);
        }
    }
}

// Example of testing a compressive sensing function (to be implemented)
TEST(CompressiveSensingTest, BasicFunctionality) {
    // This is a placeholder for future tests of compressive sensing functionality
    SUCCEED();
}
