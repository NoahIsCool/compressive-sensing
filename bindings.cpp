// bindings.cpp
#include <pybind11/pybind11.h>
#include "include/library.h"

namespace py = pybind11;

PYBIND11_MODULE(_compressive_sensing, m)
{
    m.doc() = "Python bindings for the compressive_sensing C++ library";

    m.def(
        "hello",
        [](const std::string &device) {
            hello(torch::Device(device));
        },
        py::arg("device") = "cpu",
        R"pbdoc(
            Demonstration function.

            Parameters
            ----------
            device : str, optional
                "cpu", "cuda", "mps", …  (default: "cpu")
        )pbdoc");
}

