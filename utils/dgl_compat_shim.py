"""
Compatibility shim for DGL + torchdata on PyTorch builds that don't ship
torch.utils._import_utils (e.g. torch==2.2.1 on Windows).

torchdata's datapipes.iter.util.cacheholder does:
    from torch.utils._import_utils import dill_available
This module doesn't exist in some torch builds, which crashes the import
chain: import dgl -> dgl.dataloading -> dgl.distributed -> dgl.graphbolt
-> torchdata.datapipes -> torch.utils._import_utils.

injecting a minimal stand-in module providing dill_available() before
dgl (and therefore torchdata) gets imported anywhere in the process 
resolves the crash.

Usage: import this module FIRST, before "import dgl", e.g.:
    import dgl_compat_shim  # noqa: F401  (must be first)
    import dgl
"""
import sys
import types
import torch

if not hasattr(torch.utils, "_import_utils"):
    _shim = types.ModuleType("torch.utils._import_utils")

    def dill_available():
        try:
            import dill  # noqa: F401
            return True
        except ImportError:
            return False

    _shim.dill_available = dill_available
    sys.modules["torch.utils._import_utils"] = _shim
    torch.utils._import_utils = _shim
