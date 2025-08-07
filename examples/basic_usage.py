import torch
import compressive_sensing

print("\n--- Basic usage example ---\n")

# Call the C++ hello function directly with specific device
print("Calling C++ hello function with CPU:")
compressive_sensing.hello("cpu")

# Use the Python wrapper that auto-detects the best device
print("\nCalling Python wrapper:")
used_device = compressive_sensing.hello_world()

print(f"\nExample completed using device: {used_device}")
