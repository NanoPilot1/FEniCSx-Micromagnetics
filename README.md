# SpinFEMx

This repository provides a research-oriented micromagnetic framework for solving the Landau-Lifshitz-Gilbert (LLG) equation with the finite element method in FEniCSx.

The repository contains two execution backends: a default CPU/MPI backend and GPU/CUDA backend. The CPU/MPI backend is the main implementation, while the GPU/CUDA backend has its Docker-based installation workflow.


## Main features

The CPU backend includes:

- Exchange field
- Uniaxial and cubic anisotropy
- Bulk and interfacial DMI
- Demagnetizing field using FEM/BEM approaches
- LLG and LLG-STT solvers (SOT and Zhang-Li Model)
- Energy minimization tools
- MPI-compatible workflows
- ADIOS2/adios4dolfinx checkpointing

For larger CPU/MPI simulations, the CPU Docker image provides PETSc with Htool support for hierarchical-matrix acceleration of dense or boundary-integral components.

The GPU backend includes CUDA-oriented implementations and computes the demagnetizing field using the FEM/BEM approach with either a dense GPU boundary operator for small systems or a custom compressed H-matrix backend for larger systems. Optionally, is possible to use JAXFMM, a third-party JAX-based fast multipole method library, that is useful for large system.

---


## CPU installation

The recommended CPU installation uses `conda-forge`, because DOLFINx depends on MPI, PETSc and
compiled C++ components.

Create and activate the environment:

```bash
conda create -n spinfemx -c conda-forge \
    python=3.12 \
    fenics-dolfinx=0.10.* \
    mpich \
    pyvista \
    adios4dolfinx=0.10.* \
    numpy \
    scipy \
    h5py \
    numba \
    meshio \
    pip

conda activate spinfemx
python -m pip install .
```

This installation is CPU-only and does **not** provide Htool support.

---

## CPU Docker image

For simulations requiring the optional Htool/H-matrix demagnetizing-field backend, use the CPU Docker image. This image is intended for CPU/MPI simulations that require PETSc built with Htool support.

Build the image from the repository root:

```bash
docker build -f docker/cpu/Dockerfile -t spinfemx:cpu .
```

Run a simulation with MPI:

```bash
docker run -it \
  -v "$PWD":/workspace \
  -w /workspace \
  spinfemx:cpu \
  mpiexec --allow-run-as-root -n <core_number> python script.py
```



## GPU Docker image

The GPU Docker image provides CUDA, PETSc-CUDA, CuPy, JAX, and JAXFMM. Build from the root of the repository:

```bash
docker build -f docker/gpu/Dockerfile -t spinfemx:gpu .
```


Run:

```bash
docker run --gpus all -it spinfemx:gpu bash
```

## Dependencies

The CPU backend requires a working FEniCSx/PETSc/MPI environment. The main dependencies are:

- NumPy / SciPy
- mpi4py / petsc4py
- FEniCSx / DOLFINx
- ADIOS2 with MPI support
- adios4dolfinx
- bempp-cl
- PyVista / meshio

The GPU backend additionally requires the third party libraties:

- CUDA
- PETSc built with CUDA support
- CuPy
- JAX
- JAXFMM

---


## License

This project is distributed under the license specified in the `LICENSE` file.
