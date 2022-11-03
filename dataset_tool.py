# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from deepVogue.dataset_tool.tools import *
import io
import json
import os
from typing import Optional
import click
import numpy as np
import PIL.Image
from tqdm import tqdm


@click.command()
@click.pass_context
# --source
@click.option(
    "--source",
    help="Directory or archive name for input dataset",
    required=True,
    metavar="PATH",
)
# --dest
@click.option(
    "--dest",
    help="Output directory or archive name for output dataset",
    required=True,
    metavar="PATH",
)
# --max-images
@click.option(
    "--max-images", help="Output only up to `max-images` images", type=int, default=None
)
# --resize-filter
@click.option(
    "--resize-filter",
    help="Filter to use when resizing images for output resolution",
    type=click.Choice(["box", "lanczos"]),
    default="lanczos",
    show_default=True,
)
# --transform
@click.option(
    "--transform",
    help="Input crop/resize mode",
    type=click.Choice(["center-crop", "center-crop-wide"]),
)
# --width
@click.option("--width", help="Output width", type=int)
# --height
@click.option("--height", help="Output height", type=int)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resize_filter: str,
    width: Optional[int],
    height: Optional[int],
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --width and --height options.  Output resolution will be either the original
    input resolution (if --width/--height was not specified) or the one specified with
    --width/height.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --width and --height options.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --width 512 --height=384
    """

    PIL.Image.init()  # type: ignore

    if dest == "":
        ctx.fail("--dest output filename or directory must not be an empty string")

    num_files, input_iter = open_dataset(source, max_images=max_images)

    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    transform_image = make_transform(transform, width, height, resize_filter)

    dataset_attrs = None

    labels = []

    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f"{idx:08d}"
        archive_fname = f"{idx_str[:5]}/img{idx_str}.png"

        # Apply crop and resize.
        img = transform_image(image["img"])

        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1

        cur_image_attrs = {
            "width": img.shape[1],
            "height": img.shape[0],
            "channels": channels,
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs["width"]
            height = dataset_attrs["height"]

            if width != height:
                error(
                    f"Image dimensions after scale and crop are required to be square.  Got {width}x{height}"
                )
            if dataset_attrs["channels"] not in [1, 3]:
                error("Input images must be stored as RGB or grayscale")

            if width != 2 ** int(np.floor(np.log2(width))):
                error(
                    "Image width/height after scale and crop are required to be power-of-two"
                )

        elif dataset_attrs != cur_image_attrs:
            err = [
                f"  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}"
                for k in dataset_attrs.keys()
            ]
            error(
                f"Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n"
                + "\n".join(err)
            )

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, {1: "L", 3: "RGB"}[channels])

        image_bits = io.BytesIO()

        img.save(image_bits, format="png", compress_level=0, optimize=False)

        save_bytes(
            os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer()
        )

        labels.append(
            [archive_fname, image["label"]] if image["label"] is not None else None
        )

    metadata = {"labels": labels if all(x is not None for x in labels) else None}

    save_bytes(os.path.join(archive_root_dir, "dataset.json"), json.dumps(metadata))

    close_dest()


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset()  # pylint: disable=no-value-for-parameter
