# setup.py for CMake integration with pip install -e .
import os
import pathlib
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: Extension):
        # Build the extension module directly in the compressive_sensing package directory
        extdir = os.path.join(os.path.abspath(self.build_lib), 'compressive_sensing')
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        os.makedirs(extdir, exist_ok=True)

        debug = int(os.getenv("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # ------------------------------------------------------------------
        # Locate the TorchConfig.cmake that belongs to *this* Python wheel
        # ------------------------------------------------------------------
        torch_dir_arg = []
        try:
            import torch  # noqa: F401 (import inside function on purpose)

            torch_dir = pathlib.Path(torch.__file__).parent / "share" / "cmake" / "Torch"
            torch_dir_arg = [f"-DTorch_DIR={torch_dir}"]
        except ModuleNotFoundError:
            # Torch isn't installed – raise a clearer error early
            raise RuntimeError(
                "PyTorch (torch) must be installed in the current environment "
                "before building compressive_sensing"
            ) from None

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            *torch_dir_arg,
        ]

        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "-j", "--config", cfg], cwd=build_temp)


setup(
    name="compressive_sensing",
    ext_modules=[CMakeExtension("_compressive_sensing")],
    cmdclass={"build_ext": CMakeBuild},
    packages=['compressive_sensing'],
    package_dir={'compressive_sensing': 'src/compressive_sensing'},
    zip_safe=False,
)