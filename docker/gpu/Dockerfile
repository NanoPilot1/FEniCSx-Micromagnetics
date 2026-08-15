# syntax=docker/dockerfile:1.6

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# Main versions
# -----------------------------------------------------------------------------
ARG PETSC_VERSION=v3.25.1
ARG PETSC_ARCH=arch-linux-cuda126

ARG DOLFINX_VERSION=v0.10.0
ARG BASIX_VERSION=v0.10.0
ARG UFL_VERSION=2025.2.0
ARG FFCX_VERSION=v0.10.1

ARG ADIOS2_VERSION=v2.10.2
ARG NANOBIND_VERSION=2.12.0

# -----------------------------------------------------------------------------
# Build control
#
# BUILD_JOBS is the most important knob for RAM usage.
# Use BUILD_JOBS=1 on machines with limited RAM.
# Use BUILD_JOBS=2 only if the machine has enough RAM/swap.
#
# INSTALL_PETSC_HEAVY=false avoids MUMPS/ScaLAPACK/SuiteSparse by default.
# Enable it only if you really need those direct-solver packages.
#
# INSTALL_JAX=true because this image needs JAX/JAXFMM.
# Keep BUILD_JOBS low to avoid RAM exhaustion during PETSc/DOLFINx/ADIOS2 builds.
# -----------------------------------------------------------------------------
ARG BUILD_JOBS=2
ARG INSTALL_PETSC_HEAVY=false
ARG INSTALL_JAX=true

ENV MAKEFLAGS="-j${BUILD_JOBS}"
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV MAX_JOBS=${BUILD_JOBS}

# -----------------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------------
ENV PETSC_DIR=/opt/petsc
ENV PETSC_ARCH=${PETSC_ARCH}
ENV CUDA_HOME=/usr/local/cuda
ENV VIRTUAL_ENV=/opt/venv
ENV ADIOS2_DIR=/opt/adios2

ENV PATH=${VIRTUAL_ENV}/bin:${ADIOS2_DIR}/bin:${PETSC_DIR}/${PETSC_ARCH}/bin:/opt/fenicsx/bin:${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ADIOS2_DIR}/lib:${PETSC_DIR}/${PETSC_ARCH}/lib:/opt/fenicsx/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PKG_CONFIG_PATH=${ADIOS2_DIR}/lib/pkgconfig:${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:/opt/fenicsx/lib/pkgconfig:${PKG_CONFIG_PATH}
ENV CMAKE_PREFIX_PATH=${ADIOS2_DIR}:/opt/fenicsx:${PETSC_DIR}/${PETSC_ARCH}:${CMAKE_PREFIX_PATH}
ENV PYTHONPATH=${ADIOS2_DIR}/lib/python3.12/site-packages:${PYTHONPATH}

# Avoid CPU oversubscription when using MPI/PETSc.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV NUMBA_NUM_THREADS=2
ENV VECLIB_MAXIMUM_THREADS=1

# Useful when the MPI stack is not CUDA-aware.
# This can be overridden at runtime.
ENV PETSC_OPTIONS="-use_gpu_aware_mpi 0"

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    git \
    wget \
    curl \
    ca-certificates \
    pkg-config \
    cmake \
    ninja-build \
    flex \
    bison \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev \
    liblapack-dev \
    libblas-dev \
    libhdf5-openmpi-dev \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libboost-timer-dev \
    libboost-iostreams-dev \
    libpugixml-dev \
    libspdlog-dev \
    libeigen3-dev \
    zlib1g-dev \
    libxml2-dev \
    libffi-dev \
    libyaml-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python virtual environment
# -----------------------------------------------------------------------------
RUN python3 -m venv ${VIRTUAL_ENV} && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        cython \
        numpy \
        scipy \
        pybind11 \
        "nanobind==${NANOBIND_VERSION}" \
	    "scikit-build-core[pyproject]>=0.10" \
        mpi4py

# -----------------------------------------------------------------------------
# PETSc with CUDA and MPI
#
# Reduced default PETSc package set:
# - kept: fblaslapack, hypre, metis, parmetis, htool
# - optional: scalapack, mumps, suitesparse
#
# The optional packages are useful in some workflows, but they increase build
# time, RAM pressure and image size. Enable with:
#
#   --build-arg INSTALL_PETSC_HEAVY=true
# -----------------------------------------------------------------------------
RUN git clone --branch ${PETSC_VERSION} --depth 1 https://gitlab.com/petsc/petsc.git ${PETSC_DIR}

WORKDIR ${PETSC_DIR}

RUN set -eux; \
    PETSC_EXTRA_PACKAGES=""; \
    if [ "${INSTALL_PETSC_HEAVY}" = "true" ]; then \
        PETSC_EXTRA_PACKAGES="${PETSC_EXTRA_PACKAGES} --download-scalapack=1"; \
        PETSC_EXTRA_PACKAGES="${PETSC_EXTRA_PACKAGES} --download-mumps=1"; \
        PETSC_EXTRA_PACKAGES="${PETSC_EXTRA_PACKAGES} --download-suitesparse=1"; \
    else \
        echo "Skipping heavy PETSc packages: ScaLAPACK, MUMPS, SuiteSparse"; \
    fi; \
    ./configure \
        PETSC_ARCH=${PETSC_ARCH} \
        --with-debugging=0 \
        --with-shared-libraries=1 \
        --with-mpi=1 \
        --with-cc=mpicc \
        --with-cxx=mpicxx \
        --with-fc=mpif90 \
        --with-cuda=1 \
        --with-cudac=${CUDA_HOME}/bin/nvcc \
        --with-cuda-dir=${CUDA_HOME} \
        --download-fblaslapack=1 \
        --download-hypre=1 \
        --download-metis=1 \
        --download-parmetis=1 \
        --download-htool=1 \
        ${PETSC_EXTRA_PACKAGES} \
        COPTFLAGS="-O3" \
        CXXOPTFLAGS="-O3" \
        FOPTFLAGS="-O3" \
        CUDAOPTFLAGS="-O3"

RUN make -j${BUILD_JOBS} PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} all

# -----------------------------------------------------------------------------
# petsc4py compiled against this PETSc installation
# -----------------------------------------------------------------------------
RUN python -m pip install \
    --no-build-isolation \
    --no-binary=petsc4py \
    petsc4py==3.25.1

# -----------------------------------------------------------------------------
# FEniCSx / DOLFINx v0.10.0 against this PETSc installation
# -----------------------------------------------------------------------------
WORKDIR /opt/src

RUN git clone --branch ${BASIX_VERSION} --depth 1 https://github.com/FEniCS/basix.git && \
    git clone --branch ${UFL_VERSION} --depth 1 https://github.com/FEniCS/ufl.git && \
    git clone --branch ${FFCX_VERSION} --depth 1 https://github.com/FEniCS/ffcx.git && \
    git clone --branch ${DOLFINX_VERSION} --depth 1 https://github.com/FEniCS/dolfinx.git

# -----------------------------------------------------------------------------
# Basix C++
# -----------------------------------------------------------------------------
RUN cmake -G Ninja \
    -S /opt/src/basix/cpp \
    -B /opt/src/basix/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/fenicsx \
    -DPython_EXECUTABLE=${VIRTUAL_ENV}/bin/python && \
    cmake --build /opt/src/basix/build --parallel ${BUILD_JOBS} && \
    cmake --install /opt/src/basix/build

# -----------------------------------------------------------------------------
# Basix Python, UFL and FFCx
# -----------------------------------------------------------------------------
RUN python -m pip install /opt/src/basix/python && \
    python -m pip install /opt/src/ufl && \
    python -m pip install /opt/src/ffcx

# -----------------------------------------------------------------------------
# DOLFINx C++
# -----------------------------------------------------------------------------
RUN cmake -G Ninja \
    -S /opt/src/dolfinx/cpp \
    -B /opt/src/dolfinx/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/fenicsx \
    -DPETSC_DIR=${PETSC_DIR} \
    -DPETSC_ARCH=${PETSC_ARCH} \
    -DPython_EXECUTABLE=${VIRTUAL_ENV}/bin/python && \
    cmake --build /opt/src/dolfinx/build --parallel ${BUILD_JOBS} && \
    cmake --install /opt/src/dolfinx/build

# -----------------------------------------------------------------------------
# DOLFINx Python
# -----------------------------------------------------------------------------
WORKDIR /opt/src/dolfinx/python

RUN python -m pip install -r build-requirements.txt && \
    python -m pip install \
        --check-build-dependencies \
        --no-build-isolation \
        .

# -----------------------------------------------------------------------------
# Basic scientific Python stack
#
# Important:
# - Do not install adios2 from pip.
# - Do not install adios4dolfinx yet.
# -----------------------------------------------------------------------------
RUN python -m pip install \
    numba \
    meshio \
    h5py \
    matplotlib \
    pyvista \
    bempp-cl \
    cupy-cuda12x

# -----------------------------------------------------------------------------
# ADIOS2 from source with MPI and Python bindings
#
# This replaces pip install adios2, which usually does not provide the MPI-enabled
# ADIOS2 required by adios4dolfinx.
# -----------------------------------------------------------------------------
WORKDIR /opt/src

RUN git clone --branch ${ADIOS2_VERSION} --depth 1 https://github.com/ornladios/ADIOS2.git adios2-src && \
    cmake -G Ninja \
        -S /opt/src/adios2-src \
        -B /opt/src/adios2-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${ADIOS2_DIR} \
        -DADIOS2_USE_MPI=ON \
        -DADIOS2_USE_Python=ON \
        -DADIOS2_USE_Fortran=OFF \
        -DADIOS2_BUILD_EXAMPLES=OFF \
        -DADIOS2_BUILD_TESTING=OFF \
        -DPython_EXECUTABLE=${VIRTUAL_ENV}/bin/python \
        -DPython3_EXECUTABLE=${VIRTUAL_ENV}/bin/python \
        -DMPI_C_COMPILER=mpicc \
        -DMPI_CXX_COMPILER=mpicxx && \
    cmake --build /opt/src/adios2-build --parallel ${BUILD_JOBS} && \
    cmake --install /opt/src/adios2-build

# -----------------------------------------------------------------------------
# Make the ADIOS2 Python bindings visible inside the virtual environment
#
# ADIOS2 may install its Python module under /opt/adios2/lib/... instead of the
# virtualenv site-packages. A .pth file is used to register that path.
# -----------------------------------------------------------------------------
RUN set -eux; \
    echo "Searching for ADIOS2 Python site-packages directory..."; \
    find ${ADIOS2_DIR} -maxdepth 8 -type d \( -name site-packages -o -name dist-packages \) -print; \
    ADIOS2_PYTHON_SITE="$(find ${ADIOS2_DIR} -maxdepth 8 -type d \( -name site-packages -o -name dist-packages \) | head -n 1)"; \
    if [ -z "${ADIOS2_PYTHON_SITE}" ]; then \
        echo "ERROR: ADIOS2 Python site-packages/dist-packages directory not found."; \
        echo "Full ADIOS2 installation tree:"; \
        find ${ADIOS2_DIR} -maxdepth 10 -type d -print; \
        exit 1; \
    fi; \
    echo "ADIOS2 Python path: ${ADIOS2_PYTHON_SITE}"; \
    echo "${ADIOS2_PYTHON_SITE}" > ${VIRTUAL_ENV}/lib/python3.12/site-packages/adios2-from-source.pth; \
    python - <<'PY'
import sys
print("Python executable:", sys.executable)
print("Checking ADIOS2 Python binding before installing adios4dolfinx...")

import adios2
print("adios2 imported successfully")
print("adios2 file:", getattr(adios2, "__file__", "no __file__"))
PY

# -----------------------------------------------------------------------------
# adios4dolfinx against the MPI-enabled ADIOS2 from source
#
# --no-deps is intentional. Otherwise pip may install its own adios2 wheel, which
# may not be MPI-enabled and can shadow the source-built ADIOS2 binding.
# -----------------------------------------------------------------------------
RUN python -m pip install --no-deps adios4dolfinx

# -----------------------------------------------------------------------------
# ADIOS2 build-time test
#
# Only ADIOS2 is imported here. Do not import adios4dolfinx during docker build,
# because adios4dolfinx imports dolfinx, and dolfinx may load NVIDIA driver
# libraries that are only available at runtime with docker run --gpus all.
# -----------------------------------------------------------------------------
RUN python - <<'PY'
import adios2
import importlib.metadata as metadata

print("====================================")
print("ADIOS2 build-time test")
print("====================================")
print("adios2 file:", getattr(adios2, "__file__", "no __file__"))

try:
    print("adios4dolfinx installed version:", metadata.version("adios4dolfinx"))
except Exception as e:
    print("adios4dolfinx metadata check failed:", repr(e))

print("ADIOS2 imported successfully.")
print("adios4dolfinx import must be tested at runtime with --gpus all.")
PY

# -----------------------------------------------------------------------------
# Optional JAX and JAXFMM
# -----------------------------------------------------------------------------
RUN if [ "${INSTALL_JAX}" = "true" ]; then \
        python -m pip install --upgrade "jax[cuda12]" && \
        python -m pip install "git+https://gitlab.com/jaxfmm/jaxfmm.git@strayfield" ; \
    else \
        echo "Skipping JAX/JAXFMM installation" ; \
    fi

# -----------------------------------------------------------------------------
# SpinFEMx package for the GPU image
#
# The standard pyproject.toml installs the CPU backend only.
# The full src tree is also exposed through PYTHONPATH so that the experimental
# GPU backend can be imported inside this Docker image.
# -----------------------------------------------------------------------------
WORKDIR /opt/projects/SpinFEMx

COPY . .

RUN python -m pip install -e .

ENV PYTHONPATH=/opt/projects/SpinFEMx/src:${PYTHONPATH}

# -----------------------------------------------------------------------------
# Build-level verification
#
# Do not import petsc4py, dolfinx, adios4dolfinx, or Micromagnetic here.
# PETSc with CUDA may try to load NVIDIA driver libraries that are not available
# during docker build. Full validation must be done at runtime with --gpus all.
# -----------------------------------------------------------------------------
WORKDIR /workspace

RUN python - <<'PY'
print("====================================")
print("Build-level Python verification")
print("====================================")

import sys
import importlib.metadata as metadata

print("Python:", sys.version)

import numpy
import scipy
import h5py
import meshio
import matplotlib
import numba

print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("h5py:", h5py.__version__)
print("meshio:", meshio.__version__)
print("matplotlib:", matplotlib.__version__)
print("numba:", numba.__version__)

try:
    import pyvista
    print("pyvista:", pyvista.__version__)
except Exception as e:
    print("pyvista import failed:", repr(e))

try:
    import cupy as cp
    print("cupy:", cp.__version__)
    print("cupy imported successfully")
except Exception as e:
    print("cupy import failed:", repr(e))

try:
    import adios2
    print("adios2:", getattr(adios2, "__file__", "no __file__"))
except Exception as e:
    print("adios2 import failed:", repr(e))

for pkg in ["adios4dolfinx", "dolfinx", "petsc4py"]:
    try:
        print(f"{pkg} installed version:", metadata.version(pkg))
    except Exception as e:
        print(f"{pkg} metadata check failed:", repr(e))

print("Build-level verification completed.")
print("Runtime validation must be done with docker run --gpus all.")
PY

CMD ["/bin/bash"]
