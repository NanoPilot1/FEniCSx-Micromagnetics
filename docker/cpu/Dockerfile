# syntax=docker/dockerfile:1.6

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# Main versions
# -----------------------------------------------------------------------------
ARG PETSC_VERSION=v3.25.1
ARG PETSC_ARCH=arch-linux-cpu

ARG DOLFINX_VERSION=v0.10.0
ARG BASIX_VERSION=v0.10.0
ARG UFL_VERSION=2025.2.0
ARG FFCX_VERSION=v0.10.1

ARG NANOBIND_VERSION=2.12.0

ARG ADIOS2_VERSION=v2.10.2

# -----------------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------------
ENV PETSC_DIR=/opt/petsc
ENV PETSC_ARCH=${PETSC_ARCH}
ENV VIRTUAL_ENV=/opt/venv
ENV ADIOS2_DIR=/opt/adios2

ENV PATH=${VIRTUAL_ENV}/bin:${ADIOS2_DIR}/bin:${PETSC_DIR}/${PETSC_ARCH}/bin:/opt/fenicsx/bin:${PATH}
ENV LD_LIBRARY_PATH=${ADIOS2_DIR}/lib:${PETSC_DIR}/${PETSC_ARCH}/lib:/opt/fenicsx/lib:${LD_LIBRARY_PATH}
ENV PKG_CONFIG_PATH=${ADIOS2_DIR}/lib/pkgconfig:${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:/opt/fenicsx/lib/pkgconfig:${PKG_CONFIG_PATH}
ENV CMAKE_PREFIX_PATH=${ADIOS2_DIR}:/opt/fenicsx:${PETSC_DIR}/${PETSC_ARCH}:${CMAKE_PREFIX_PATH}
ENV PYTHONPATH=${ADIOS2_DIR}/lib/python3.12/site-packages:${PYTHONPATH}

# Avoid CPU oversubscription when using MPI/PETSc.

ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMBA_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1

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
# PETSc with MPI, CPU-only
#
# CUDA, CuPy, JAX and GPU-aware MPI are intentionally disabled/removed.
# PT-Scotch is omitted because upstream download can be fragile.
# METIS and ParMETIS are kept.
# -----------------------------------------------------------------------------
RUN git clone --branch ${PETSC_VERSION} --depth 1 https://gitlab.com/petsc/petsc.git ${PETSC_DIR}

WORKDIR ${PETSC_DIR}

RUN ./configure \
    PETSC_ARCH=${PETSC_ARCH} \
    --with-debugging=0 \
    --with-shared-libraries=1 \
    --with-mpi=1 \
    --with-cc=mpicc \
    --with-cxx=mpicxx \
    --with-fc=mpif90 \
    --download-fblaslapack=1 \
    --download-hypre=1 \
    --download-metis=1 \
    --download-parmetis=1 \
    --download-scalapack=1 \
    --download-mumps=1 \
    --download-suitesparse=1 \
    --download-htool=1 \
    COPTFLAGS="-O3" \
    CXXOPTFLAGS="-O3" \
    FOPTFLAGS="-O3"

RUN make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} all

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
    cmake --build /opt/src/basix/build && \
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
    cmake --build /opt/src/dolfinx/build && \
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

ARG JAX_VERSION=0.11.0
ARG JAXFMM_VERSION=0.3.3

# -----------------------------------------------------------------------------
# Basic scientific Python stack and CPU JAXFMM
#
# JAX is pinned so a clean Docker build does not silently change the FMM
# numerical/runtime stack. CuPy and CUDA packages remain excluded.
# -----------------------------------------------------------------------------
RUN python -m pip install \
    numba \
    meshio \
    h5py \
    matplotlib \
    pyvista \
    bempp-cl

RUN python -m pip install \
        "jax==${JAX_VERSION}" && \
    python -m pip install \
        --no-deps \
        "jaxfmm==${JAXFMM_VERSION}" && \
    python -m pip check

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
    cmake --build /opt/src/adios2-build && \
    cmake --install /opt/src/adios2-build

# -----------------------------------------------------------------------------
# Make the ADIOS2 Python bindings visible inside the virtual environment
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
# adios4dolfinx is pinned to the release line compatible with DOLFINx 0.10.
# ADIOS2 itself is provided by the MPI-enabled source build above.
# -----------------------------------------------------------------------------
ARG ADIOS4DOLFINX_VERSION=0.10.0.post1

RUN python -m pip install \
        "adios4dolfinx==${ADIOS4DOLFINX_VERSION}" && \
    python -m pip check

# -----------------------------------------------------------------------------
# ADIOS2/adios4dolfinx build-time test
#
# In the CPU-only image we can import adios4dolfinx during build because DOLFINx
# does not depend on CUDA or NVIDIA driver libraries.
# -----------------------------------------------------------------------------
RUN python - <<'PY'
import adios2
import adios4dolfinx
import dolfinx
from petsc4py import PETSc

print("====================================")
print("CPU build-time verification")
print("====================================")
print("adios2 file:", getattr(adios2, "__file__", "no __file__"))
print("adios4dolfinx file:", getattr(adios4dolfinx, "__file__", "no __file__"))
print("dolfinx:", dolfinx.__version__)
print("PETSc:", PETSc.Sys.getVersion())

for pkg in [
    "hypre",
    "metis",
    "parmetis",
    "scalapack",
    "mumps",
    "suitesparse",
    "htool",
]:
    try:
        print(f"{pkg}:", PETSc.Sys.hasExternalPackage(pkg))
    except Exception as e:
        print(f"{pkg}: could not query ({repr(e)})")

print("CPU build-time verification completed.")
PY

# -----------------------------------------------------------------------------
# CPU project
# -----------------------------------------------------------------------------
WORKDIR /opt/projects/SpinFEMx

COPY . .

RUN python -m pip install .

# -----------------------------------------------------------------------------
# Optional: install GPU project only if it does not require CUDA/JAX/CuPy at import
#
# If your GPU package imports cupy, jax, or CUDA-dependent modules at import time,
# do not install it in this CPU-only image.
# -----------------------------------------------------------------------------
# WORKDIR /opt/projects/micromagnetic-GPU
# COPY micromagnetic-GPU/ ./
# RUN python -m pip install .

# -----------------------------------------------------------------------------
# Final build-level verification
# -----------------------------------------------------------------------------
WORKDIR /workspace

RUN python - <<'PY'
print("====================================")
print("Final CPU image verification")
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
import pyvista
import jax
import jaxfmm
from jaxfmm.apps.mag.strayfield import plan_strayfield
import adios2
import adios4dolfinx
import dolfinx
from petsc4py import PETSc

print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("h5py:", h5py.__version__)
print("meshio:", meshio.__version__)
print("matplotlib:", matplotlib.__version__)
print("numba:", numba.__version__)
print("pyvista:", pyvista.__version__)
print("jax:", jax.__version__)
print("jax backend:", jax.default_backend())
print("jax devices:", jax.devices())
print("jaxfmm:", metadata.version("jaxfmm"))
assert jax.default_backend() == "cpu"
print("adios2:", getattr(adios2, "__file__", "no __file__"))
print("adios4dolfinx:", getattr(adios4dolfinx, "__file__", "no __file__"))
print("dolfinx:", dolfinx.__version__)
print("PETSc:", PETSc.Sys.getVersion())

try:
    import spinfemx
    import spinfemx.cpu as fxm_cpu

    print("spinfemx:", getattr(spinfemx, "__file__", "no __file__"))
    print("spinfemx.cpu:", getattr(fxm_cpu, "__file__", "no __file__"))

except Exception as e:
    print("spinfemx import failed:", repr(e))

print("Final CPU image verification completed.")
PY

CMD ["/bin/bash"]