#!/bin/bash
# Build the Linux prebuilt kernel for one CUDA version inside PyTorch's manylinux_2_28
# build image, so the library runs on any Linux with glibc 2.28 or newer.
#   tools/build_kernel_linux.sh 12.8    # writes atlas/_kernel/_C_cu12.so
#   tools/build_kernel_linux.sh 13.0    # writes atlas/_kernel/_C_cu13.so
# Compiles against the oldest supported torch (2.10); the stable ABI carries it forward.
set -euo pipefail
CUDA="${1:?usage: $0 <CUDA version, e.g. 12.8 or 13.0>}"
TORCH="${TORCH_VERSION:-2.10.*}"
TAG="cu${CUDA//./}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm \
  -v "$REPO":/repo -w /repo \
  -v atlas-pip-cache:/root/.cache/pip \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  "pytorch/manylinux2_28-builder:cuda${CUDA}" bash -c "
    set -euo pipefail
    PY=/opt/python/cp311-cp311/bin/python
    \$PY -m pip install -q --root-user-action=ignore 'torch==${TORCH}' \
        --index-url https://download.pytorch.org/whl/${TAG}
    \$PY -m pip install -q --root-user-action=ignore ninja setuptools
    \$PY tools/build_kernel.py
    chown \"\$HOST_UID:\$HOST_GID\" atlas/_kernel/_C_cu*.so
  "
