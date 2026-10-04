import importlib
import pytest

REQUIRED = ["numpy", "torch", "gymnasium", "matplotlib", "imageio", "yaml"]


@pytest.mark.parametrize("module", REQUIRED)
def test_dependency_importable(module):
    importlib.import_module(module)


def test_numpy_torch_abi_compatible():
    """torch<2.4 was built against NumPy 1.x; importing both must not raise."""
    import numpy as np
    import torch

    t = torch.from_numpy(np.ones(3, dtype=np.float32))
    assert t.sum().item() == pytest.approx(3.0)


def test_no_cuda_assumptions():
    import torch

    assert torch.cuda.is_available() is False, "Plan assumes CPU-only execution"
