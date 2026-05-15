import os
import copy
import numpy as np
import click
from typing import List, Optional
import torch
import pickle
from deepVogue import neuronal_network_utils
from deepVogue import legacy


def extract_conv_names(model, model_res):
    model_names = list(name for name, weight in model.named_parameters())

    return model_names


def _is_low_res_layer(name, low_res_set, backbone):
    if backbone == "sg2":
        return any(f"synthesis.b{res}" in name for res in low_res_set)
    # SG3: synthesis.L<idx>_<bandwidth>_<ch>.* — bandwidth ≈ layer resolution.
    if name.startswith("synthesis.L") and "_" in name:
        try:
            tag = name.split(".")[1]
            bw = int(tag.split("_")[1])
            return bw <= max(low_res_set)
        except (IndexError, ValueError):
            return False
    return False


def _detect_backbone(model):
    for n, _ in model.named_parameters():
        if n.startswith("synthesis.b"):
            return "sg2"
        if n.startswith("synthesis.L"):
            return "sg3"
    return "sg2"


def blend_models(low, high, model_res, resolution):

    resolutions = set(4 * 2**x for x in range(int(np.log2(resolution) - 1)))
    backbone = _detect_backbone(low)

    low_names = extract_conv_names(low, model_res)
    high_names = extract_conv_names(high, model_res)

    assert all((x == y for x, y in zip(low_names, high_names)))

    model_out = copy.deepcopy(low)
    params_src = high.named_parameters()
    dict_dest = model_out.state_dict()

    for name, param in params_src:
        if not _is_low_res_layer(name, resolutions, backbone) and "mapping" not in name:
            dict_dest[name].data.copy_(param.data)

    model_out_dict = model_out.state_dict()
    model_out_dict.update(dict_dest)
    model_out.load_state_dict(dict_dest)

    return model_out


# ----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option(
    "--lower_res_pkl",
    help="Network pickle filename for lower resolutions",
    required=True,
)
@click.option(
    "--higher_res_pkl",
    help="Network pickle filename for higher resolutions",
    required=True,
)
@click.option(
    "--output_path",
    "out",
    help="Network pickle filepath for output",
    default="./blended.pkl",
)
@click.option(
    "--model_res",
    type=int,
    help="Output resolution of model (likely 1024, 512, or 256)",
    default=1024,
    show_default=True,
)
@click.option(
    "--split_res",
    "resolution",
    type=int,
    help="Resolution to split model weights",
    default=64,
    show_default=True,
)
def create_blended_model(
    ctx: click.Context,
    lower_res_pkl: str,
    higher_res_pkl: str,
    model_res: Optional[int],
    resolution: Optional[int],
    out: Optional[str],
):

    G_kwargs = neuronal_network_utils.EasyDict()

    with neuronal_network_utils.util.open_url(lower_res_pkl) as f:
        lo = legacy.load_network_pkl(f, custom=False, **G_kwargs)  # type: ignore
        lo_G, lo_D, lo_G_ema = lo["G"], lo["D"], lo["G_ema"]

    with neuronal_network_utils.util.open_url(higher_res_pkl) as f:
        hi = legacy.load_network_pkl(f, custom=False, **G_kwargs)["G_ema"]  # type: ignore

    model_out = blend_models(lo_G_ema, hi, model_res, resolution)
    # for n in model_out.named_parameters():
    #     print(n[0])

    data = dict([("G", None), ("D", None), ("G_ema", None)])
    with open(out, "wb") as f:
        data["G"] = lo_G
        data["D"] = lo_D
        data["G_ema"] = model_out
        pickle.dump(data, f)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    create_blended_model()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
