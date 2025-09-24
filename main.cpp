
#include "include/compressive_sensing.h"
#include <iostream>
#include <torch/torch.h>

int main() {
    
    auto X = torch::tensor({
                {0, 0},
                {1, 2},
                {3, 4},
            {5, 6}
        }, torch::kFloat).transpose(0, 1);

    std::cout << X << std::endl;
    std::cout << compressive_sensing::mutual_coherence(X) << std::endl;
    return 0;


}
