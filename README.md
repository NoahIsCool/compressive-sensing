# Compressive Sensing

A C++/PyTorch implementation of compressive sensing algorithms with Python bindings.

## Installation

### Prerequisites

- CMake 3.31 or higher
- C++26 compatible compiler
- PyTorch
- Python 3.9 or higher

### Install from source

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install --no-build-isolation -e .
```

## Usage

```python
import torch
import compressive_sensing

# Call the C++ function directly
compressive_sensing.hello("cpu")  # specify device

# Or use the Python wrapper that detects the best device
compressive_sensing.hello_world()
```

## Testing

Run the C++ tests:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```
