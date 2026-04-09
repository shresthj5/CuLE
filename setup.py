import argparse
import os
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

DEFAULT_CUDA_ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST")
if DEFAULT_CUDA_ARCH_LIST is None:
    DEFAULT_CUDA_ARCH_LIST = "8.0 8.6"
    os.environ["TORCH_CUDA_ARCH_LIST"] = DEFAULT_CUDA_ARCH_LIST


parser = argparse.ArgumentParser("CuLE", add_help=False)
parser.add_argument(
    "--fastbuild",
    action="store_true",
    default=False,
    help="Build CuLE supporting only 2K roms",
)
parser.add_argument(
    "--compiler",
    type=str,
    default=None,
    help="Host compiler passed to nvcc via -ccbin",
)
parser.add_argument(
    "--atari-env-block-size",
    type=int,
    default=int(os.environ.get("CULE_ATARI_ENV_BLOCK_SIZE", "1")),
    help="Compile-time block size for CuLE's env/reset/preprocess CUDA kernels",
)
parser.add_argument(
    "--atari-process-block-size",
    type=int,
    default=int(os.environ.get("CULE_ATARI_PROCESS_BLOCK_SIZE", "64")),
    help="Compile-time block size for CuLE's frame preprocessing CUDA kernel",
)
args, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

supported_env_block_sizes = {1, 32, 64, 128, 256}
if args.atari_env_block_size not in supported_env_block_sizes:
    raise SystemExit(
        "Unsupported --atari-env-block-size={} (expected one of {})".format(
            args.atari_env_block_size,
            sorted(supported_env_block_sizes),
        )
    )
if args.atari_process_block_size not in supported_env_block_sizes:
    raise SystemExit(
        "Unsupported --atari-process-block-size={} (expected one of {})".format(
            args.atari_process_block_size,
            sorted(supported_env_block_sizes),
        )
    )


base_dir = Path(__file__).resolve().parent
third_party_dir = base_dir / "third_party"

sources = [
    str(base_dir / "torchcule" / "frontend.cpp"),
    str(base_dir / "torchcule" / "backend.cu"),
]

include_dirs = [
    str(base_dir),
    str(third_party_dir / "agency"),
]

cxx_flags = ["-O3", "-Wall", "-Wextra", "-fPIC"]
nvcc_flags = [
    "-O3",
    "-Xptxas=-v",
    "-lineinfo",
    "-Xcompiler=-Wall,-Wextra,-fPIC",
]
libraries = ["z"]
extra_link_args = []

if os.name != "nt":
    cxx_flags.append("-fopenmp")
    nvcc_flags.append("-Xcompiler=-fopenmp")
    extra_link_args.append("-fopenmp")

if args.fastbuild:
    nvcc_flags.append("-DCULE_FAST_COMPILE")

if args.compiler:
    nvcc_flags.append(f"-ccbin={args.compiler}")

block_size_macro = f"-DCULE_ATARI_ENV_BLOCK_SIZE={args.atari_env_block_size}"
cxx_flags.append(block_size_macro)
nvcc_flags.append(block_size_macro)
process_block_size_macro = f"-DCULE_ATARI_PROCESS_BLOCK_SIZE={args.atari_process_block_size}"
cxx_flags.append(process_block_size_macro)
nvcc_flags.append(process_block_size_macro)


setup(
    name="torchcule",
    version="0.1.0",
    description="A GPU RL environment package for PyTorch",
    url="https://github.com/NVlabs/cule",
    author="Steven Dalton",
    author_email="sdalton1@gmail.com",
    install_requires=["gymnasium", "ale-py"],
    ext_modules=[
        CUDAExtension(
            name="torchcule_atari",
            sources=sources,
            include_dirs=include_dirs,
            libraries=libraries,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
            extra_link_args=extra_link_args,
        )
    ],
    packages=find_packages(exclude=["build"]),
    cmdclass={"build_ext": BuildExtension},
)
