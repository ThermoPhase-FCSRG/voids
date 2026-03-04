from __future__ import annotations

import numpy as np
import pytest

from voids.geom import area_equivalent_diameter, characteristic_size
from voids.workflows import (
    binarize_grayscale_volume,
    crop_nonzero_cylindrical_volume,
    extract_spanning_porespy_network,
    infer_sample_axes,
    largest_true_rectangle,
    preprocess_grayscale_cylindrical_volume,
)


def test_area_equivalent_diameter_and_characteristic_size_priority() -> None:
    """Test public characteristic-size helpers used by diagnostics and plotting."""

    area = np.array([np.pi, 4.0 * np.pi])
    assert np.allclose(area_equivalent_diameter(area), np.array([2.0, 4.0]))

    store = {
        "diameter_equivalent": np.array([5.0, 6.0]),
        "diameter_inscribed": np.array([3.0, 4.0]),
        "radius_inscribed": np.array([1.0, 2.0]),
        "area": np.array([np.pi, 4.0 * np.pi]),
    }
    values, label = characteristic_size(store, expected_shape=(2,))
    assert label == "diameter_equivalent"
    assert np.array_equal(values, np.array([5.0, 6.0]))

    radius_values, radius_label = characteristic_size(
        {"radius_inscribed": np.array([1.0, 2.0])},
        expected_shape=(2,),
    )
    assert radius_label == "radius_inscribed"
    assert np.array_equal(radius_values, np.array([2.0, 4.0]))

    with pytest.raises(KeyError, match="characteristic size fields"):
        characteristic_size({})


def test_largest_true_rectangle_and_crop_fill_internal_holes() -> None:
    """Test maximal rectangle detection and slice-wise support hole filling."""

    mask = np.array(
        [
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ],
        dtype=bool,
    )
    assert largest_true_rectangle(mask) == (1, 4, 1, 4)

    raw = np.zeros((3, 6, 8), dtype=float)
    raw[:, 1:5, 1:7] = 10.0
    raw[:, 2:4, 3:5] = 2.0
    raw[1, 2:4, 3:5] = 0.0  # interior hole that should be filled in the specimen support

    crop = crop_nonzero_cylindrical_volume(raw)

    assert crop.crop_bounds_yx == (1, 5, 1, 7)
    assert crop.cropped.shape == (3, 4, 6)
    assert crop.specimen_mask[1, 2:4, 3:5].all()
    assert crop.common_mask[1:5, 1:7].all()


def test_preprocess_grayscale_cylindrical_volume_segments_dark_voids() -> None:
    """Test grayscale crop plus automatic thresholding for dark void segmentation."""

    raw = np.zeros((3, 6, 8), dtype=float)
    raw[:, 1:5, 1:7] = 10.0
    raw[:, 2:4, 3:5] = 2.0

    seg = preprocess_grayscale_cylindrical_volume(raw, threshold_method="otsu", void_phase="dark")

    assert seg.crop.crop_bounds_yx == (1, 5, 1, 7)
    assert 2.0 < seg.threshold < 10.0
    assert seg.binary.shape == (3, 4, 6)
    assert seg.binary[:, 1:3, 2:4].all()
    assert not seg.binary[:, 0, 0].any()

    bright_binary, used_threshold = binarize_grayscale_volume(
        seg.crop.cropped, threshold=6.0, void_phase="bright"
    )
    assert used_threshold == pytest.approx(6.0)
    assert bright_binary[:, 0, 0].all()
    assert not bright_binary[:, 1:3, 2:4].any()


def test_infer_axes_and_extract_spanning_porespy_network() -> None:
    """Test shared PoreSpy extraction workflow metadata and imported networks."""

    _, axis_lengths, axis_areas, flow_axis = infer_sample_axes((12, 16, 16), voxel_size=1.0)
    assert flow_axis == "y"
    assert axis_lengths == {"x": 12.0, "y": 16.0, "z": 16.0}
    assert axis_areas == {"x": 256.0, "y": 192.0, "z": 192.0}

    im = np.zeros((12, 16, 16), dtype=int)
    im[:, 5:11, 5:11] = 1
    im[2:4, 1:3, 1:3] = 1

    result = extract_spanning_porespy_network(
        im,
        voxel_size=1.0,
        flow_axis="x",
        length_unit="voxel",
        provenance_notes={"case": "tiny"},
    )

    assert result.flow_axis == "x"
    assert result.porespy_version is not None
    assert result.provenance.user_notes["case"] == "tiny"
    assert result.sample.units["length"] == "voxel"
    assert result.net_full.Np >= result.net.Np
    assert result.net_full.Nt >= result.net.Nt
    assert np.array_equal(result.image, im)
    assert result.pore_indices.ndim == 1
    assert result.throat_mask.shape == (result.net_full.Nt,)
