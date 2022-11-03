import functools
import pickle
import sys
import tarfile
import gzip
import zipfile
from pathlib import Path
from typing import Callable, Tuple, Union, Optional
import os
import io
import json
import numpy as np
import PIL.Image

def error(msg):
    print("\nℹ️ Error: " + msg)
    sys.exit(1)


def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a


def file_ext(name: Union[str, Path]) -> str:
    return str(name).split(".")[-1]


def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f".{ext}" in PIL.Image.EXTENSION  # type: ignore


def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [
        str(f) for f in sorted(Path(source_dir).rglob("*"))
        if is_image_ext(f) and os.path.isfile(f)
    ]

    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, "dataset.json")
    if os.path.isfile(meta_fname):
        with open(meta_fname, "r") as file:
            labels = json.load(file)["labels"]
            if labels is not None:
                labels = {x[0]: x[1] for x in labels}
            else:
                labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace("\\", "/")
            img = np.array(PIL.Image.open(fname))
            yield dict(img=img, label=labels.get(arch_fname))
            if idx >= max_idx - 1:
                break

    return max_idx, iterate_images()


def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode="r") as z:
        input_images = [
            str(f) for f in sorted(z.namelist()) if is_image_ext(f)
        ]

        # Load labels.
        labels = {}
        if "dataset.json" in z.namelist():
            with z.open("dataset.json", "r") as file:
                labels = json.load(file)["labels"]
                if labels is not None:
                    labels = {x[0]: x[1] for x in labels}
                else:
                    labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode="r") as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, "r") as file:
                    img = PIL.Image.open(file)  # type: ignore
                    img = np.array(img)
                yield dict(img=img, label=labels.get(fname))
                if idx >= max_idx - 1:
                    break

    return max_idx, iterate_images()


def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  # pip install opencv-python
    import lmdb  # pip install lmdb # pylint: disable=import-error

    with lmdb.open(lmdb_dir, readonly=True,
                   lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()["entries"], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True,
                       lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(
                            np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError("cv2.imdecode failed")
                        img = img[:, :, ::-1]  # BGR => RGB
                    except IOError:
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, label=None)
                    if idx >= max_idx - 1:
                        break
                except:
                    print(sys.exc_info()[1])

    return max_idx, iterate_images()


def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, "r:gz") as tar:
        for batch in range(1, 6):
            member = tar.getmember(f"cifar-10-batches-py/data_batch_{batch}")
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding="latin1")
            images.append(data["data"].reshape(-1, 3, 32, 32))
            labels.append(data["labels"])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1])  # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000, ) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx - 1:
                break

    return max_idx, iterate_images()


def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace("-images-idx3-ubyte.gz",
                                  "-labels-idx1-ubyte.gz")
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open(labels_gz, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0, 0), (2, 2), (2, 2)],
                    "constant",
                    constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000, ) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx - 1:
                break

    return max_idx, iterate_images()


def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int],
    resize_filter: str,
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    resample = {
        "box": PIL.Image.BOX,
        "lanczos": PIL.Image.LANCZOS
    }[resize_filter]

    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), resample)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2:(img.shape[0] + crop) // 2,
                  (img.shape[1] - crop) // 2:(img.shape[1] + crop) // 2, ]
        img = PIL.Image.fromarray(img, "RGB")
        img = img.resize((width, height), resample)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2:(img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, "RGB")
        img = img.resize((width, height), resample)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2:(width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == "center-crop":
        if (output_width is None) or (output_height is None):
            error("must specify --width and --height when using " + transform +
                  "transform")
        return functools.partial(center_crop, output_width, output_height)
    if transform == "center-crop-wide":
        if (output_width is None) or (output_height is None):
            error("must specify --width and --height when using " + transform +
                  " transform")
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, "unknown transform"


def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        if source.rstrip("/").endswith("_lmdb"):
            return open_lmdb(source, max_images=max_images)
        else:
            return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if os.path.basename(source) == "cifar-10-python.tar.gz":
            return open_cifar10(source, max_images=max_images)
        elif os.path.basename(source) == "train-images-idx3-ubyte.gz":
            return open_mnist(source, max_images=max_images)
        elif file_ext(source) == "zip":
            return open_image_zip(source, max_images=max_images)
        else:
            assert False, "unknown archive type"
    else:
        error(f"Missing input file or directory: {source}")


def open_dest(
    dest: str,
) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == "zip":
        if os.path.dirname(dest) != "":
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest,
                             mode="w",
                             compression=zipfile.ZIP_STORED)

        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)

        return "", zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error("--dest folder must be empty")
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, "wb") as fout:
                if isinstance(data, str):
                    data = data.encode("utf8")
                fout.write(data)

        return dest, folder_write_bytes, lambda: None
