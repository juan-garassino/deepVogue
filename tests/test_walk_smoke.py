"""Pure-numpy smoke tests for the walk module: anchor loading + interpolation.

The full mp4 pipeline requires torch/CUDA/imageio-ffmpeg; here we exercise the
parts of walk.py that have no torch dependency, which is enough to catch
regressions in interpolation order, anchor loading, and frames_index ordering.
"""
import json
import numpy as np
from pathlib import Path
from deepVogue import walk


def test_load_anchors_alphabetical(tmp_path):
    for i in range(3):
        d = tmp_path / f"{i:08d}"; d.mkdir()
        np.savez(d / "projected_w.npz", w=np.full((1, 4, 8), float(i), dtype=np.float32))
    arr = walk._load_anchors(tmp_path, order_json=None)
    assert arr.shape == (3, 4, 8)
    assert arr[0, 0, 0] == 0.0 and arr[2, 0, 0] == 2.0


def test_load_anchors_uses_frames_index(tmp_path):
    # write anchors out of order on disk; index dictates 1→0→2
    for i in (0, 1, 2):
        d = tmp_path / f"{i:08d}"; d.mkdir()
        np.savez(d / "projected_w.npz", w=np.full((1, 2, 4), float(i), dtype=np.float32))
    idx = {"fps": 1, "resolution": 64, "frames": [
        {"kept_index": 1, "video_id": "v", "source_frame": 1, "timecode_s": 1.0, "filename": "img00000001.png"},
        {"kept_index": 0, "video_id": "v", "source_frame": 0, "timecode_s": 0.0, "filename": "img00000000.png"},
        {"kept_index": 2, "video_id": "v", "source_frame": 2, "timecode_s": 2.0, "filename": "img00000002.png"},
    ]}
    j = tmp_path / "frames_index.json"; j.write_text(json.dumps(idx))
    arr = walk._load_anchors(tmp_path, order_json=j)
    assert [arr[i, 0, 0] for i in range(3)] == [1.0, 0.0, 2.0]


def test_interpolate_shape_and_modes():
    a = np.random.RandomState(0).randn(3, 4, 8).astype(np.float32)
    for mode in ("linear", "slerp", "bezier"):
        traj = walk._interpolate_anchors(a, frames_per_segment=8, mode=mode)
        # 2 segments × 8 frames + final anchor
        assert traj.shape == (17, 4, 8), mode
        assert np.allclose(traj[0], a[0], atol=1e-5)
        assert np.allclose(traj[-1], a[-1], atol=1e-5)


def test_slerp_endpoints_exact():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.allclose(walk._slerp(0.0, a, b), a)
    assert np.allclose(walk._slerp(1.0, a, b), b)
    mid = walk._slerp(0.5, a, b)
    assert abs(np.linalg.norm(mid) - 1.0) < 1e-5
