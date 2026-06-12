"""Official NVlabs pkls embed source importing torch_utils.* / dnnlib.*;
deepVogue.legacy must alias those names to the renamed in-repo packages."""

import importlib
import io

import deepVogue.legacy as legacy
from deepVogue import neuronal_network_utils, pytorch_utils


def test_torch_utils_alias_resolves():
    mod = importlib.import_module("torch_utils")
    assert mod is pytorch_utils


def test_torch_utils_submodules_are_same_objects():
    from deepVogue.pytorch_utils import persistence
    from deepVogue.pytorch_utils.ops import bias_act

    assert importlib.import_module("torch_utils.persistence") is persistence
    assert importlib.import_module("torch_utils.ops.bias_act") is bias_act


def test_dnnlib_alias_resolves():
    mod = importlib.import_module("dnnlib")
    assert mod is neuronal_network_utils
    assert importlib.import_module("dnnlib.util") is neuronal_network_utils.util


def test_unpickler_resolves_nvidia_global_names():
    # hand-rolled GLOBAL-opcode pickles using the module paths NVIDIA pkls embed
    cls = legacy._LegacyUnpickler(io.BytesIO(b"cdnnlib.util\nEasyDict\n.")).load()
    assert cls is neuronal_network_utils.EasyDict

    from deepVogue.pytorch_utils import persistence

    fn = legacy._LegacyUnpickler(
        io.BytesIO(b"ctorch_utils.persistence\n_reconstruct_persistent_obj\n.")
    ).load()
    assert fn is persistence._reconstruct_persistent_obj
