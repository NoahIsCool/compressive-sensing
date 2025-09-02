cmake_minimum_required(VERSION 3.31)
project(compressive_sensing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)

# ------------------------------------------------------------------
#  Prefer CMake packages that live in the active Python environment
# ------------------------------------------------------------------
# 1) Ask Python for <venv>/lib/pythonX.Y/site-packages
# 2) Recursively collect every *Config.cmake that lives under it
# 3) Push the parent directory of each hit to the front of
#    CMAKE_PREFIX_PATH so that find_package() will pick those copies
#    before it searches the system.
#
# The block is a no-op when no virtual-env is active or when the env
# contains no CMake-aware wheels, so it is safe to leave it in place
# for all builds.

# ------------------------------------------------------------
# Ensure CMake searches the venv's libtorch, not a system copy
# ------------------------------------------------------------
message("python path: ${Python_EXECUTABLE}")
message("torch dir: ${Torch_DIR_FROM_VENV}")
if (NOT ${Python_EXECUTABLE})
    set(Python_EXECUTABLE python3)
endif()

execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch, pathlib, sys; print(pathlib.Path(torch.__file__).parent / \"share\" / \"cmake\" / \"Torch\")"
        OUTPUT_VARIABLE Torch_DIR_FROM_VENV
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Add only if the user did not override Torch_DIR on the command line
if(NOT Torch_DIR)
    set(Torch_DIR "${Torch_DIR_FROM_VENV}" CACHE PATH "TorchConfig.cmake location")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${Torch_DIR}")

message("Using this directory for pytorch: ${CMAKE_PREFIX_PATH}")

# ---------- Dependencies ----------
find_package(Torch REQUIRED)

# pybind11 (header-only) ----------
include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# GoogleTest for C++ unit tests ----------
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ---------- Core static library ----------
add_library(compressive_sensing STATIC lib/omp.cpp)
target_link_libraries(compressive_sensing PRIVATE ${TORCH_LIBRARIES})
target_include_directories(compressive_sensing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---------- Python extension module ----------
pybind11_add_module(_compressive_sensing bindings.cpp)
target_link_libraries(_compressive_sensing PRIVATE compressive_sensing ${TORCH_LIBRARIES})
# On Windows leave the filename as _compressive_sensing.pyd, on *nix -> .so
set_target_properties(_compressive_sensing PROPERTIES
        OUTPUT_NAME "_compressive_sensing")

# Install the Python extension _inside_ the package directory
install(
        TARGETS _compressive_sensing
        DESTINATION compressive_sensing
)

# ---------- CLI executable ----------
add_executable(main main.cpp)
target_link_libraries(main PRIVATE compressive_sensing ${TORCH_LIBRARIES})

# ---------- C++ tests ----------
add_executable(compressive_sensing_test tests/test_omp.cpp)
target_link_libraries(compressive_sensing_test PRIVATE compressive_sensing GTest::gtest_main GTest::gmock ${TORCH_LIBRARIES})
include(GoogleTest)
gtest_discover_tests(compressive_sensing_test)

# ---------- Bash script test (runs via CTest) ----------

if(NOT EXISTS "${BASH_TEST_SCRIPT}")
    # You can override BASH_TEST_SCRIPT on the CMake command line:
    #   -DBASH_TEST_SCRIPT=/full/path/to/your_script.sh
    set(BASH_TEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_python_tests.sh" CACHE FILEPATH "Path to bash test script to run with CTest")
endif()
# Use bash explicitly so the script doesn't need the +x bit set
add_test(
        NAME bash_script_tests
        COMMAND /usr/bin/env bash "${BASH_TEST_SCRIPT}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
