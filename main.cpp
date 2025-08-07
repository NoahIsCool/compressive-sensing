
#include "include/library.h"
#include <iostream>

int main() {
    // Try with CPU (always available)
    std::cout << "\n--- Testing with CPU ---\n" << std::endl;
    const auto dev = get_device();

    hello();



    // Try with MPS if available (for Apple Silicon)
    #ifdef USE_MPS
    if (torch::mps::is_available()) {
        std::cout << "\n--- Testing with MPS ---\n" << std::endl;
        hello("mps");
    } else {
        std::cout << "\n--- MPS not available ---\n" << std::endl;
    }
    #endif

    return 0;
}
