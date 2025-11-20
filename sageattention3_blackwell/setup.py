import warnings
import re
import os
from pathlib import Path
from packaging.version import parse, Version
from setuptools import setup, find_packages
import subprocess
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "sageattn3"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FAHOPPER_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FAHOPPER_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FAHOPPER_FORCE_CXX11_ABI", "FALSE") == "TRUE"

# Supported NVIDIA GPU architectures; keep in sync with workflows.
SUPPORTED_ARCHS = {"10.0", "12.0"}

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "1"]


cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    
    compute_capabilities = set()
    # Prefer TORCH_CUDA_ARCH_LIST if explicitly specified (works without GPUs).
    PARSE_CUDA_ARCH_RE = re.compile(
        r"(?P<major>[0-9]+)\.(?P<minor>[0-9])(?P<suffix>[a-zA-Z]{0,1})(?P<ptx>\+PTX){0,1}"
    )
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list_env is not None:
        # Parse the CUDA architecture list from the environment variable.
        for arch in arch_list_env.replace(" ", ";").split(";"):
            match = PARSE_CUDA_ARCH_RE.match(arch)
            if match is None:
                warnings.warn(f"Invalid CUDA architecture {arch}. Supported architectures: {SUPPORTED_ARCHS}.")
                continue
            major = match.group("major")
            minor = match.group("minor")
            suffix = match.group("suffix") or ""
            ptx = match.group("ptx") or ""
            if f"{major}.{minor}" not in SUPPORTED_ARCHS:
                warnings.warn(f"Unsupported CUDA architecture {arch}. Supported architectures: {SUPPORTED_ARCHS}.")
                continue
            compute_capabilities.add(f"{major}.{minor}{suffix}{ptx}")

    if not compute_capabilities:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(
                "Unable to detect a CUDA device, and no TORCH_CUDA_ARCH_LIST was set."
            )
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            capability = f"{major}.{minor}"
            if capability not in SUPPORTED_ARCHS:
                warnings.warn(
                    f"skipping GPU {i} with compute capability {capability}; supported: {SUPPORTED_ARCHS}"
                )
                continue
            if capability not in compute_capabilities:
                compute_capabilities.add(capability)

    if not compute_capabilities:
        raise RuntimeError(
            "No target compute capabilities. Set TORCH_CUDA_ARCH_LIST or build on a machine with GPUs."
        )
    else:
        print(f"Target compute capabilities: {compute_capabilities}")


    def has_capability(target):
            return any(cc.startswith(target) for cc in compute_capabilities)

    def get_nvcc_flags(allowed_capabilities):
        flags = []
        # Add target compute capabilities to NVCC flags.
        for capability in compute_capabilities:
            if not any(capability.startswith(prefix) for prefix in allowed_capabilities):
                continue
            num = capability.split("+")[0].replace(".", "")
            if num in {"100", "120"}:
                num += "a"
            flags += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
            if capability.endswith("+PTX"):
                flags += ["-gencode", f"arch=compute_{num},code=compute_{num}"]
        return flags

    if has_capability(("10.0",)):
        cc_flag += get_nvcc_flags(["10.0"])
    if has_capability(("12.0",)):
        cc_flag += get_nvcc_flags(["12.0"])

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    repo_dir = Path(this_dir)
    cutlass_dir = repo_dir / "csrc" / "cutlass"
    (repo_dir / "csrc").mkdir(parents=True, exist_ok=True)
    if not cutlass_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", str(cutlass_dir)],
            check=True
        )
    nvcc_flags = [
        "-O3",
        # "-O0",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        # "--ptxas-options=-v",  # printing out number of registers
        "--ptxas-options=--verbose,--warn-on-local-memory-usage",  # printing out number of registers
        "-lineinfo",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-DQBLKSIZE=128",
        "-DKBLKSIZE=128",
        "-DCTA256",
        "-DDQINRMEM",
    ]
    include_dirs = [
        repo_dir / "sageattn3",
        cutlass_dir / "include",
        cutlass_dir / "tools" / "util" / "include",
    ]

    ext_modules.append(
        CUDAExtension(
            name="fp4attn_cuda",
            sources=["sageattn3/blackwell/api.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="fp4quant_cuda",
            sources=["sageattn3/quantization/fp4_quantization_4d.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]
        )
    )



class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        super().run()

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="FP4FlashAttention",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
