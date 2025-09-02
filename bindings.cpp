// bindings.cpp
#include <pybind11/pybind11.h>
#include "include/omp.h"

namespace py = pybind11;
 // bindings.cpp
 #include <pybind11/pybind11.h>
#include <torch/extension.h>
 #include "include/omp.h"
 
 namespace py = pybind11;
 
 PYBIND11_MODULE(_compressive_sensing, m)
 {
     m.doc() = "Python bindings for the compressive_sensing C++ library";
 
     m.def(
         "get_device",
         []() {
             return omp::get_device().str();
         },
         R"pbdoc(
             Return the best available device as a string: "cuda", "mps", or "cpu".
         )pbdoc");
            // omp::mutual_coherence(X);
        // inline torch::Device get_device() {
    m.def(
        "mutual_coherence",
        &omp::mutual_coherence,
        py::arg("X"),
        R"pbdoc(
            Compute the mutual coherence of the column-normalized dictionary X.
            Parameters
            ----------
            X : torch.Tensor (shape [m, N])
                Dictionary matrix (m < N).
            Returns
            -------
            float
        )pbdoc");

    m.def(
        "welch_bound",
        &omp::welch_bound,
        py::arg("X"),
        R"pbdoc(
            Compute the Welch bound for matrix X.
            Parameters
            ----------
            X : torch.Tensor (shape [m, N])
            Returns
            -------
            float
        )pbdoc");
}

