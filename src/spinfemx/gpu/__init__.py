"""
Experimental CUDA/GPU backend for SpinFEMx.

This backend is intended to be used through the GPU Docker image.
It is not installed by the default CPU Python package.
"""

from __future__ import annotations
from .io_utils import load_mesh_xdmf

__all__ = [
    "LLG_GPU",
    "LLG_STT_GPU",
    "LLG_SOT_GPU",
    "EnergyMinimizerGPU",
    "ExchangeField",
    "AnisotropyField",
    "CubicAnisotropyField",
    "DMIBULK",
    "DMIInterfacial",
    "DemagFieldFMMJAXGPU",
    "DemagFieldLindholmGPU",
    "load_mesh_xdmf"
]


def __getattr__(name: str):
    if name == "LLG_GPU":
        from .solvers.llg_module_GPU import LLG_GPU
        return LLG_GPU

    if name == "LLG_STT_GPU":
        from .solvers.llg_stt_module_GPU import LLG_STT_GPU
        return LLG_STT_GPU
        
    if name == "LLG_SOT_GPU":
        from .solvers.llg_SOT_module_GPU import LLG_SOT_GPU
        return LLG_SOT_GPU   
    
    if name == "EnergyMinimizerGPU":
        from .solvers.Minimizer_GPU import EnergyMinimizerGPU
        return EnergyMinimizerGPU
    
    if name == "ExchangeField":
        from .fields.Exchange_GPU import ExchangeField
        return ExchangeField

    if name == "AnisotropyField":
        from .fields.Anisotropy_GPU import AnisotropyField
        return AnisotropyField

    if name == "CubicAnisotropyField":
        from .fields.Cubic_Anisotropy_GPU import CubicAnisotropyFieldGPU
        return CubicAnisotropyFieldGPU

    if name == "DMIBULK":
        from .fields.DMI_Bulk_GPU import DMIBULK
        return DMIBULK

    if name == "DMIInterfacial":
        from .fields.DMI_Interfacial_GPU import DMIInterfacial
        return DMIInterfacial

    if name == "DemagFieldFMMJAXGPU":
        from .fields.Demag_FMM_GPU import DemagFieldFMMJAXGPU
        return DemagFieldFMMJAXGPU

    if name == "DemagFieldLindholmGPU":
        from .fields.Demag_Lindholm_GPU import DemagFieldLindholmGPU
        return DemagFieldLindholmGPU

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
