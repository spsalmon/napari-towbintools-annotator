# Panoptic Annotator Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the unimplemented "panoptic" project type in `napari-towbintools-annotator`, porting the standalone `napari-panoptic-annotator`'s point-based per-instance class annotation into the existing project framework.

**Architecture:** A new `PanopticProject` (project.yaml model) is wired into the existing project creator and loader. A new `PanopticAnnotatorWidget` reuses the framework shell (project label, file list, navigation, save) and folds in the standalone's logic. The core conversion logic (points↔CSV rows, color↔class) is extracted into pure module-level functions so it is testable without a live napari viewer. Class colors are derived deterministically from a shared palette by class index.

**Tech Stack:** Python 3.10+, qtpy (Qt widgets), napari 0.7.0 (Viewer, Labels/Points layers), pandas, numpy, scikit-image (`regionprops`), tifffile/imageio (I/O), natsort, pytest.

**Test environment:** All tests run in the `napari` micromamba env: `micromamba run -n napari python -m pytest`. There is no `pytest-qt`; the one viewer-backed test constructs `napari.Viewer(show=False)` directly.

---

## File Structure

- `src/napari_towbintools_annotator/colors.py` — **new.** Shared palette + color helpers (`CLASS_PALETTE`, `hex_to_rgba_float`, `class_hex`).
- `src/napari_towbintools_annotator/project.py` — **modify.** Add `PanopticProject`; register it in `Project.load` dispatch.
- `src/napari_towbintools_annotator/panoptic_annotator.py` — **new.** Pure logic functions (`nearest_class_id`, `points_to_rows`, `rows_to_points`) + `PanopticAnnotatorWidget`.
- `src/napari_towbintools_annotator/project_creator.py` — **modify.** Add `scan_panoptic_files` (module-level), `_run_panoptic_creation`, panoptic creator-UI visibility, panoptic routing in `create_project` and `create_annotator_widget`.
- `src/napari_towbintools_annotator/annotators.py` — **modify.** Import `CLASS_PALETTE` from `colors.py` instead of defining `_CLASS_PALETTE` locally.
- `src/napari_towbintools_annotator/_tests/test_panoptic.py` — **new.** Tests for colors, pure logic, `PanopticProject`, `scan_panoptic_files`, and a viewer-backed widget smoke test.
- `pyproject.toml` — **modify.** Add runtime dependencies.

`napari.yaml` and `__init__.py` are **not** changed: panoptic is reached through the existing `TowbintoolsAnnotatorWidget` command, not a new napari contribution.

---

## Task 1: Add runtime dependencies

`scikit-image` (for `regionprops`) is newly required. `pandas`, `tifffile`, `imageio` are already imported by `annotators.py`/`project_creator.py` but undeclared — add them for correctness. (`magicgui` is **not** added: the standalone's layer-picker widgets are dropped in this integration.)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update the dependencies array**

In `pyproject.toml`, replace the current `dependencies` block:

```toml
dependencies = [
    "numpy",
    "napari_guitils",
    "natsort",
]
```

with:

```toml
dependencies = [
    "numpy",
    "pandas",
    "napari_guitils",
    "natsort",
    "scikit-image",
    "tifffile",
    "imageio",
]
```

- [ ] **Step 2: Verify the package still imports**

Run: `micromamba run -n napari python -c "import skimage.measure, pandas, tifffile, imageio; print('deps ok')"`
Expected: `deps ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "Declare panoptic runtime dependencies"
```

---

## Task 2: Shared color helpers (`colors.py`)

**Files:**
- Create: `src/napari_towbintools_annotator/colors.py`
- Modify: `src/napari_towbintools_annotator/annotators.py`
- Test: `src/napari_towbintools_annotator/_tests/test_panoptic.py`

- [ ] **Step 1: Write the failing test**

Create `src/napari_towbintools_annotator/_tests/test_panoptic.py` with:

```python
from napari_towbintools_annotator.colors import (
    CLASS_PALETTE,
    class_hex,
    hex_to_rgba_float,
)


def test_hex_to_rgba_float_white():
    assert hex_to_rgba_float("#ffffff") == (1.0, 1.0, 1.0, 1.0)


def test_hex_to_rgba_float_strips_hash_and_scales():
    r, g, b, a = hex_to_rgba_float("#4C72B0")
    assert a == 1.0
    assert abs(r - 0x4C / 255) < 1e-9
    assert abs(g - 0x72 / 255) < 1e-9
    assert abs(b - 0xB0 / 255) < 1e-9


def test_class_hex_indexes_palette_and_wraps():
    classes = list(range(len(CLASS_PALETTE) + 2))
    assert class_hex(classes, 0) == CLASS_PALETTE[0]
    assert class_hex(classes, len(CLASS_PALETTE)) == CLASS_PALETTE[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/test_panoptic.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'napari_towbintools_annotator.colors'`

- [ ] **Step 3: Create `colors.py`**

```python
# Palette of visually distinct colors shared across annotators.
CLASS_PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]


def class_hex(classes, name):
    """Return the palette hex color for a class, by its index in ``classes``."""
    idx = classes.index(name)
    return CLASS_PALETTE[idx % len(CLASS_PALETTE)]


def hex_to_rgba_float(hex_str):
    """Convert a ``#RRGGBB`` string to an RGBA float tuple in [0, 1]."""
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/test_panoptic.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Point `annotators.py` at the shared palette**

In `src/napari_towbintools_annotator/annotators.py`, remove the local palette definition (the `_CLASS_PALETTE = [ ... ]` block including its comment, lines ~18-31) and add an import near the other local imports (after `from .project import ClassificationProject`):

```python
from .colors import CLASS_PALETTE as _CLASS_PALETTE
```

Leave all other uses of `_CLASS_PALETTE` in `annotators.py` unchanged.

- [ ] **Step 6: Verify the existing module still imports and tests pass**

Run: `micromamba run -n napari python -c "import napari_towbintools_annotator.annotators; print('ok')"`
Expected: `ok`

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/ -q`
Expected: PASS (existing tests + new color tests)

- [ ] **Step 7: Commit**

```bash
git add src/napari_towbintools_annotator/colors.py src/napari_towbintools_annotator/annotators.py src/napari_towbintools_annotator/_tests/test_panoptic.py
git commit -m "Add shared color helpers and reuse them in annotators"
```

---

## Task 3: `PanopticProject` model

**Files:**
- Modify: `src/napari_towbintools_annotator/project.py`
- Test: `src/napari_towbintools_annotator/_tests/test_panoptic.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_panoptic.py`:

```python
import pytest

from napari_towbintools_annotator.project import PanopticProject, Project


def _make_panoptic_project(tmp_path, classes=("a", "b")):
    return PanopticProject(
        name="p",
        image_type="multichannel",
        annotation_directories=["annotations"],
        annotation_df_path="annotations/annotations.csv",
        data_directories=[str(tmp_path / "ref")],
        mask_directories=[str(tmp_path / "seg")],
        classes=list(classes),
        project_dir=str(tmp_path),
    )


def test_panoptic_project_save_load_roundtrip(tmp_path):
    proj = _make_panoptic_project(tmp_path)
    proj.save()
    loaded = Project.load(str(tmp_path))
    assert isinstance(loaded, PanopticProject)
    assert loaded.project_type == "panoptic"
    assert loaded.classes == ["a", "b"]
    assert loaded.mask_directories == [str(tmp_path / "seg")]
    assert loaded.data_directories == [str(tmp_path / "ref")]
    assert loaded.annotation_df_path == "annotations/annotations.csv"


def test_panoptic_project_requires_classes(tmp_path):
    with pytest.raises(ValueError):
        PanopticProject(
            name="p",
            image_type="multichannel",
            annotation_directories=["annotations"],
            annotation_df_path="annotations/annotations.csv",
            data_directories=[str(tmp_path / "ref")],
            mask_directories=[str(tmp_path / "seg")],
            classes=[],
            project_dir=str(tmp_path),
        )


def test_panoptic_project_requires_mask_dirs(tmp_path):
    with pytest.raises(ValueError):
        PanopticProject(
            name="p",
            image_type="multichannel",
            annotation_directories=["annotations"],
            annotation_df_path="annotations/annotations.csv",
            data_directories=[str(tmp_path / "ref")],
            mask_directories=[],
            classes=["a"],
            project_dir=str(tmp_path),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/test_panoptic.py -q`
Expected: FAIL with `ImportError: cannot import name 'PanopticProject'`

- [ ] **Step 3: Add `PanopticProject` and register it**

In `src/napari_towbintools_annotator/project.py`, add `"panoptic"` to the dispatch dict inside `Project.load`:

```python
        dispatch = {
            "classification": ClassificationProject.load,
            "panoptic": PanopticProject.load,
        }
```

Then append this class at the end of the file:

```python
class PanopticProject(Project):
    def __init__(
        self,
        name: str,
        image_type: str,
        annotation_directories: list,
        annotation_df_path: str,
        project_dir: str,
        classes: list,
        data_directories: list = None,
        mask_directories: list = None,
        ignored_images: list = None,
    ):
        if not classes:
            raise ValueError(
                "Classes must be provided for panoptic projects."
            )
        if not data_directories:
            raise ValueError(
                "data_directories must be provided for panoptic projects."
            )
        if not mask_directories:
            raise ValueError(
                "mask_directories must be provided for panoptic projects."
            )

        super().__init__(
            name=name,
            image_type=image_type,
            project_type="panoptic",
            annotation_directories=annotation_directories,
            data_directories=data_directories or [],
            project_dir=project_dir,
            ignored_images=ignored_images,
        )

        self.annotation_df_path = annotation_df_path
        self.classes = classes
        self.mask_directories = mask_directories or []

    def save(self):
        project_data = {
            "name": self.name,
            "image_type": self.image_type,
            "project_type": self.project_type,
            "annotation_directories": self.annotation_directories,
            "annotation_df_path": self.annotation_df_path,
            "data_directories": self.data_directories,
            "project_dir": self.project_dir,
            "ignored_images": self.ignored_images,
            "classes": self.classes,
            "mask_directories": self.mask_directories,
        }

        with open(f"{self.project_dir}/project.yaml", "w") as file:
            yaml.dump(project_data, file)

    @classmethod
    def load(cls, project_dir: str, project_data: dict):
        return cls(
            name=project_data["name"],
            image_type=project_data["image_type"],
            annotation_directories=project_data["annotation_directories"],
            annotation_df_path=project_data["annotation_df_path"],
            data_directories=project_data["data_directories"],
            project_dir=project_dir,
            classes=project_data.get("classes", []),
            mask_directories=project_data.get("mask_directories", []),
            ignored_images=project_data.get("ignored_images", []),
        )
```

Note: `Project.load` references `PanopticProject` at call time (not import time), so defining the class after `Project` in the same module is fine.

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/test_panoptic.py -q`
Expected: PASS (color + project tests)

- [ ] **Step 5: Commit**

```bash
git add src/napari_towbintools_annotator/project.py src/napari_towbintools_annotator/_tests/test_panoptic.py
git commit -m "Add PanopticProject model"
```

---

## Task 4: Pure conversion logic (`panoptic_annotator.py`)

These module-level functions hold all the points↔CSV logic so it is testable without a viewer. Color→class matching uses nearest-color (squared distance) rather than exact tuple equality, which is robust to float storage differences in napari.

**Files:**
- Create: `src/napari_towbintools_annotator/panoptic_annotator.py`
- Test: `src/napari_towbintools_annotator/_tests/test_panoptic.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_panoptic.py`:

```python
import numpy as np
import pandas as pd

from napari_towbintools_annotator.panoptic_annotator import (
    nearest_class_id,
    points_to_rows,
    rows_to_points,
)


def test_nearest_class_id_handles_float_noise():
    id_to_color = {0: (1, 0, 0, 1), 1: (0, 0, 1, 1)}
    assert nearest_class_id((0.98, 0.01, 0.0, 1.0), id_to_color) == 0
    assert nearest_class_id((0.0, 0.02, 0.97, 1.0), id_to_color) == 1


def test_points_to_rows_2d():
    label_data = np.zeros((10, 10), dtype=int)
    label_data[2:4, 2:4] = 5
    label_data[6:8, 6:8] = 9
    id_to_color = {0: (1, 0, 0, 1), 1: (0, 0, 1, 1)}
    id_to_name = {0: "a", 1: "b"}
    points = np.array([[3, 3], [7, 7]])
    colors = np.array([(1, 0, 0, 1), (0, 0, 1, 1)], dtype=float)
    rows = points_to_rows(points, colors, label_data, id_to_color, id_to_name)
    assert rows == [
        {"Label": 5, "ClassID": 0, "Class": "a"},
        {"Label": 9, "ClassID": 1, "Class": "b"},
    ]


def test_points_to_rows_3d_includes_plane():
    label_data = np.zeros((3, 10, 10), dtype=int)
    label_data[1, 2:4, 2:4] = 7
    id_to_color = {0: (1, 0, 0, 1)}
    id_to_name = {0: "a"}
    points = np.array([[1, 3, 3]])
    colors = np.array([(1, 0, 0, 1)], dtype=float)
    rows = points_to_rows(
        points, colors, label_data, id_to_color, id_to_name, plane_axis="Z"
    )
    assert rows == [{"Z": 1, "Label": 7, "ClassID": 0, "Class": "a"}]


def test_points_to_rows_skips_out_of_bounds():
    label_data = np.zeros((5, 5), dtype=int)
    rows = points_to_rows(
        np.array([[100, 100]]),
        np.array([(1, 0, 0, 1)], dtype=float),
        label_data,
        {0: (1, 0, 0, 1)},
        {0: "a"},
    )
    assert rows == []


def test_rows_to_points_2d_centroid():
    label_data = np.zeros((10, 10), dtype=int)
    label_data[2:4, 2:4] = 5
    df = pd.DataFrame([{"Label": 5, "ClassID": 0, "Class": "a"}])
    placements = rows_to_points(df, label_data, {0: (1, 0, 0, 1)})
    assert len(placements) == 1
    point, color = placements[0]
    assert color == (1, 0, 0, 1)
    np.testing.assert_allclose(point, [2.5, 2.5])


def test_rows_to_points_3d_centroid():
    label_data = np.zeros((3, 10, 10), dtype=int)
    label_data[1, 2:4, 2:4] = 7
    df = pd.DataFrame([{"Z": 1, "Label": 7, "ClassID": 0, "Class": "a"}])
    placements = rows_to_points(
        df, label_data, {0: (1, 0, 0, 1)}, plane_axis="Z"
    )
    assert len(placements) == 1
    point, _ = placements[0]
    np.testing.assert_allclose(point, [1, 2.5, 2.5])


def test_rows_to_points_skips_missing_label():
    label_data = np.zeros((10, 10), dtype=int)
    df = pd.DataFrame([{"Label": 99, "ClassID": 0, "Class": "a"}])
    assert rows_to_points(df, label_data, {0: (1, 0, 0, 1)}) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/test_panoptic.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'napari_towbintools_annotator.panoptic_annotator'`

- [ ] **Step 3: Create `panoptic_annotator.py` with the pure functions**

```python
import numpy as np
from skimage.measure import regionprops


def nearest_class_id(color, id_to_color):
    """Return the class id whose RGBA color is closest to ``color``."""
    target = np.asarray(color, dtype=float)
    best_id, best_dist = -1, float("inf")
    for class_id, class_color in id_to_color.items():
        dist = np.sum((target - np.asarray(class_color, dtype=float)) ** 2)
        if dist < best_dist:
            best_dist, best_id = dist, class_id
    return best_id


def points_to_rows(
    points, face_colors, label_data, id_to_color, id_to_name, plane_axis=None
):
    """Convert annotation points + colors into per-instance annotation rows.

    Each point is rounded to integer coordinates, used to read the label value
    under it, and its color is matched to the nearest class. Points outside the
    label array are skipped. In 3D (``plane_axis`` set) the first-axis index is
    recorded under that column name.
    """
    rows = []
    shape = label_data.shape
    for point, color in zip(points, face_colors):
        index = tuple(int(round(coord)) for coord in point)
        if len(index) != len(shape):
            continue
        if any(i < 0 or i >= s for i, s in zip(index, shape)):
            continue
        label_value = int(label_data[index])
        class_id = nearest_class_id(color, id_to_color)
        class_name = id_to_name.get(class_id, "unknown")
        if plane_axis is not None:
            row = {
                plane_axis: index[0],
                "Label": label_value,
                "ClassID": class_id,
                "Class": class_name,
            }
        else:
            row = {
                "Label": label_value,
                "ClassID": class_id,
                "Class": class_name,
            }
        rows.append(row)
    return rows


def _centroid(mask):
    props = regionprops(mask.astype(int))
    if not props:
        return None
    return props[0].centroid


def rows_to_points(annotations_df, label_data, id_to_color, plane_axis=None):
    """Convert annotation rows back into ``(point_coords, rgba)`` placements.

    For each row the label's centroid is used as the point location. Rows with
    an unknown class id, or whose label is absent from the (plane of the) label
    array, are skipped.
    """
    placements = []
    if plane_axis is not None:
        for plane in annotations_df[plane_axis].unique():
            plane = int(plane)
            if plane < 0 or plane >= label_data.shape[0]:
                continue
            plane_df = annotations_df[annotations_df[plane_axis] == plane]
            for _, row in plane_df.iterrows():
                color = id_to_color.get(int(row["ClassID"]))
                if color is None:
                    continue
                mask = label_data[plane] == int(row["Label"])
                if not mask.any():
                    continue
                centroid = _centroid(mask)
                if centroid is None:
                    continue
                placements.append(
                    (np.array([plane, centroid[0], centroid[1]]), color)
                )
    else:
        for _, row in annotations_df.iterrows():
            color = id_to_color.get(int(row["ClassID"]))
            if color is None:
                continue
            mask = label_data == int(row["Label"])
            if not mask.any():
                continue
            centroid = _centroid(mask)
            if centroid is None:
                continue
            placements.append(
                (np.array([centroid[0], centroid[1]]), color)
            )
    return placements
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/test_panoptic.py -q`
Expected: PASS (all logic tests green)

- [ ] **Step 5: Commit**

```bash
git add src/napari_towbintools_annotator/panoptic_annotator.py src/napari_towbintools_annotator/_tests/test_panoptic.py
git commit -m "Add panoptic point/CSV conversion logic"
```

---

## Task 5: `scan_panoptic_files` helper

A module-level function in `project_creator.py` so the file-scanning + count-matching logic is testable without constructing the Qt widget.

**Files:**
- Modify: `src/napari_towbintools_annotator/project_creator.py`
- Test: `src/napari_towbintools_annotator/_tests/test_panoptic.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_panoptic.py`:

```python
from napari_towbintools_annotator.project_creator import scan_panoptic_files


def test_scan_panoptic_files_matches(tmp_path):
    ref = tmp_path / "ref"
    seg = tmp_path / "seg"
    ref.mkdir()
    seg.mkdir()
    for name in ("a.tif", "b.tif"):
        (ref / name).write_text("x")
        (seg / name).write_text("x")
    refs, segs = scan_panoptic_files([str(ref)], [str(seg)])
    assert len(refs) == 2
    assert len(segs) == 2
    assert all(r.endswith(".tif") for r in refs)


def test_scan_panoptic_files_mismatch_raises(tmp_path):
    ref = tmp_path / "ref"
    seg = tmp_path / "seg"
    ref.mkdir()
    seg.mkdir()
    (ref / "a.tif").write_text("x")
    (seg / "a.tif").write_text("x")
    (seg / "b.tif").write_text("x")
    with pytest.raises(ValueError):
        scan_panoptic_files([str(ref)], [str(seg)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/test_panoptic.py -q`
Expected: FAIL with `ImportError: cannot import name 'scan_panoptic_files'`

- [ ] **Step 3: Add `scan_panoptic_files`**

In `src/napari_towbintools_annotator/project_creator.py`, add this module-level function just after the existing `convert_path_to_dir_name` function:

```python
def scan_panoptic_files(data_directories, mask_directories):
    """Scan reference and segmentation directories for a panoptic project.

    Returns ``(reference_files, segmentation_files)`` as natsorted absolute
    paths. Raises ``ValueError`` if the counts do not match.
    """
    reference_files = natsorted(
        [
            os.path.join(d, f)
            for d in data_directories
            for f in os.listdir(d)
            if os.path.isfile(os.path.join(d, f))
        ]
    )
    segmentation_files = natsorted(
        [
            os.path.join(d, f)
            for d in mask_directories
            for f in os.listdir(d)
            if os.path.isfile(os.path.join(d, f))
        ]
    )
    if len(reference_files) != len(segmentation_files):
        raise ValueError(
            f"File count mismatch: {len(reference_files)} reference files vs "
            f"{len(segmentation_files)} segmentation files."
        )
    return reference_files, segmentation_files
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/test_panoptic.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/napari_towbintools_annotator/project_creator.py src/napari_towbintools_annotator/_tests/test_panoptic.py
git commit -m "Add scan_panoptic_files helper"
```

---

## Task 6: `PanopticAnnotatorWidget`

The widget reuses the framework shell and folds in the standalone's logic via the pure functions from Task 4. No standalone-style directory/file-management UI or layer pickers — layers are opened automatically per file.

**Files:**
- Modify: `src/napari_towbintools_annotator/panoptic_annotator.py`

- [ ] **Step 1: Add the widget imports**

At the top of `src/napari_towbintools_annotator/panoptic_annotator.py`, above the existing `import numpy as np` line, add:

```python
import os
import threading
```

and below the `from skimage.measure import regionprops` line add:

```python
import pandas as pd
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QButtonGroup
from qtpy.QtWidgets import QHBoxLayout
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QListWidget
from qtpy.QtWidgets import QListWidgetItem
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QRadioButton
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

from .colors import CLASS_PALETTE
from .colors import hex_to_rgba_float
```

- [ ] **Step 2: Append the widget class**

Add at the end of `panoptic_annotator.py`:

```python
_PLANE_AXIS = "Z"
_DONE_COLOR = "#55A868"


class PanopticAnnotatorWidget(QWidget):
    def __init__(self, napari_viewer, project, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.project = project

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.project_label = QLabel(f"Project: {project.name}")
        self.main_layout.addWidget(self.project_label)

        self.annotation_df_path = os.path.join(
            project.project_dir, project.annotation_df_path
        )
        self.annotation_df = pd.read_csv(self.annotation_df_path)
        for col in ("Reference", "Segmentation", "Annotation"):
            if col in self.annotation_df.columns:
                self.annotation_df[col] = (
                    self.annotation_df[col].fillna("").astype(str)
                )
        self.reference_files = self.annotation_df["Reference"].tolist()

        # Class lookups; colors derived from palette by class index.
        self.classes = list(project.classes)
        self.class_id_to_name = dict(enumerate(self.classes))
        self.class_name_to_id = {c: i for i, c in enumerate(self.classes)}
        self.class_id_to_color = {
            i: hex_to_rgba_float(CLASS_PALETTE[i % len(CLASS_PALETTE)])
            for i in range(len(self.classes))
        }
        self.class_name_to_color = {
            c: self.class_id_to_color[i]
            for i, c in enumerate(self.classes)
        }
        self.selected_class = self.classes[0] if self.classes else None

        # Layer + write state.
        self._reference_layer = None
        self._segmentation_layer = None
        self._annotation_layer = None
        self._write_lock = threading.Lock()
        self._pending_write = False

        # File list.
        self.file_list_widget = QListWidget()
        self._populate_file_list()
        self.current_file_idx = self._find_resume_index()
        if self.current_file_idx >= len(self.reference_files):
            self.current_file_idx = 0
        self.file_list_widget.setCurrentRow(self.current_file_idx)
        self.file_list_widget.itemClicked.connect(self.choose_file_from_list)
        self.main_layout.addWidget(self.file_list_widget)

        # Navigation.
        nav_layout = QHBoxLayout()
        self.previous_button = QPushButton("Previous [H]")
        self.next_button = QPushButton("Next [J]")
        self.previous_button.clicked.connect(self.previous_file)
        self.next_button.clicked.connect(self.next_file)
        nav_layout.addWidget(self.previous_button)
        nav_layout.addWidget(self.next_button)
        self.main_layout.addLayout(nav_layout)

        # Class radio buttons.
        self.class_buttons_widget = QWidget()
        self.class_buttons_layout = QVBoxLayout()
        self.class_buttons_widget.setLayout(self.class_buttons_layout)
        self.class_buttons = QButtonGroup(self)
        for class_name in self.classes:
            button = QRadioButton(class_name)
            self._style_class_button(button, class_name)
            self.class_buttons.addButton(button)
            self.class_buttons_layout.addWidget(button)
            if class_name == self.selected_class:
                button.setChecked(True)
        self.class_buttons.buttonClicked.connect(self._on_class_button)
        self.main_layout.addWidget(self.class_buttons_widget)

        self.save_button = QPushButton("Save annotations [S]")
        self.save_button.clicked.connect(self.save_annotations)
        self.main_layout.addWidget(self.save_button)

        # Key bindings.
        self._bound_keys = {
            "Up": self._cycle_class_up,
            "Down": self._cycle_class_down,
            "j": self._next_file_key,
            "h": self._previous_file_key,
            "s": self._save_key,
        }
        for key, callback in self._bound_keys.items():
            self.viewer.bind_key(key, callback, overwrite=True)

        self._load_file()

    # ----- file list -----
    def _populate_file_list(self):
        self.file_list_widget.clear()
        for i, path in enumerate(self.reference_files):
            item = QListWidgetItem(os.path.basename(path))
            self._apply_item_color(item, i)
            self.file_list_widget.addItem(item)

    def _apply_item_color(self, item, idx):
        annotation = str(self.annotation_df.loc[idx, "Annotation"]).strip()
        if annotation in ("", "nan", "None"):
            item.setBackground(QColor("transparent"))
            item.setForeground(QColor("white"))
        else:
            item.setBackground(QColor(_DONE_COLOR))
            item.setForeground(QColor("white"))

    def _find_resume_index(self):
        annotated = self.annotation_df["Annotation"].astype(str).str.strip()
        for i, value in enumerate(annotated):
            if value in ("", "nan", "None"):
                return i
        return 0

    # ----- class selection -----
    def _style_class_button(self, button, class_name):
        color = CLASS_PALETTE[
            self.class_name_to_id[class_name] % len(CLASS_PALETTE)
        ]
        bg = QColor(color)
        luminance = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
        text_color = "black" if luminance > 128 else "white"
        button.setStyleSheet(
            f"QRadioButton {{ background-color: {color}; "
            f"color: {text_color}; padding: 3px; border-radius: 3px; }}"
        )

    def _on_class_button(self, button):
        self.selected_class = button.text()
        self._update_point_color()

    def _cycle_class(self, delta):
        if not self.classes:
            return
        idx = (
            self.classes.index(self.selected_class) + delta
        ) % len(self.classes)
        self.selected_class = self.classes[idx]
        for button in self.class_buttons.buttons():
            if button.text() == self.selected_class:
                button.setChecked(True)
        self._update_point_color()

    def _update_point_color(self):
        if self._annotation_layer is None or self.selected_class is None:
            return
        self._annotation_layer.selected_data = set()
        self._annotation_layer.current_face_color = (
            self.class_name_to_color[self.selected_class]
        )

    # ----- file loading -----
    def _plane_axis(self):
        if self._segmentation_layer is None:
            return None
        return _PLANE_AXIS if self._segmentation_layer.ndim == 3 else None

    def _add_annotation_layer(self):
        ndim = self._segmentation_layer.ndim
        self._annotation_layer = self.viewer.add_points(
            np.zeros((0, ndim)), name="Annotations", ndim=ndim, size=10
        )
        self._update_point_color()

    def _replay_annotations(self, csv_path):
        df = pd.read_csv(csv_path)
        label_data = np.asarray(self._segmentation_layer.data)
        placements = rows_to_points(
            df, label_data, self.class_id_to_color, self._plane_axis()
        )
        if not placements:
            return
        coords = np.array([point for point, _ in placements])
        colors = np.array([color for _, color in placements], dtype=float)
        self._annotation_layer.data = coords
        self._annotation_layer.face_color = colors

    def _load_file(self):
        if not self.reference_files or not (
            0 <= self.current_file_idx < len(self.reference_files)
        ):
            return

        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        self._reference_layer = None
        self._segmentation_layer = None
        self._annotation_layer = None

        row = self.annotation_df.iloc[self.current_file_idx]
        reference_file = row["Reference"]
        segmentation_file = row["Segmentation"]
        annotation_file = str(row["Annotation"]).strip()

        self._reference_layer = self.viewer.open(reference_file)[-1]
        self._segmentation_layer = self.viewer.open(
            segmentation_file, layer_type="labels", opacity=0.5
        )[-1]
        self._add_annotation_layer()

        if annotation_file not in ("", "nan", "None") and os.path.isfile(
            annotation_file
        ):
            self._replay_annotations(annotation_file)

        self.viewer.reset_view()

    def choose_file_from_list(self):
        self.current_file_idx = self.file_list_widget.currentRow()
        self._load_file()

    def next_file(self):
        if not self.reference_files:
            return
        self.current_file_idx = min(
            self.current_file_idx + 1, len(self.reference_files) - 1
        )
        self.file_list_widget.setCurrentRow(self.current_file_idx)
        self._load_file()

    def previous_file(self):
        if not self.reference_files:
            return
        self.current_file_idx = max(self.current_file_idx - 1, 0)
        self.file_list_widget.setCurrentRow(self.current_file_idx)
        self._load_file()

    # ----- saving -----
    def save_annotations(self):
        if self._annotation_layer is None or self._segmentation_layer is None:
            return
        label_data = np.asarray(self._segmentation_layer.data)
        rows = points_to_rows(
            np.asarray(self._annotation_layer.data),
            np.asarray(self._annotation_layer.face_color),
            label_data,
            self.class_id_to_color,
            self.class_id_to_name,
            self._plane_axis(),
        )
        df = pd.DataFrame(rows)

        reference = self.annotation_df.loc[self.current_file_idx, "Reference"]
        name = os.path.splitext(os.path.basename(reference))[0]
        annotations_dir = os.path.dirname(self.annotation_df_path)
        out_path = os.path.join(annotations_dir, f"{name}.csv")
        df.to_csv(out_path, index=False)

        self.annotation_df.loc[self.current_file_idx, "Annotation"] = out_path
        item = self.file_list_widget.item(self.current_file_idx)
        self._apply_item_color(item, self.current_file_idx)
        self._save_master_async()

    def _save_master_sync(self):
        with self._write_lock:
            self._pending_write = False
            self.annotation_df.to_csv(self.annotation_df_path, index=False)

    def _save_master_async(self):
        snapshot = self.annotation_df.copy()
        path = self.annotation_df_path

        def write():
            with self._write_lock:
                self._pending_write = False
                snapshot.to_csv(path, index=False)

        self._pending_write = True
        threading.Thread(target=write, daemon=True).start()

    # ----- key callbacks (napari passes the viewer) -----
    def _cycle_class_up(self, viewer=None):
        self._cycle_class(-1)

    def _cycle_class_down(self, viewer=None):
        self._cycle_class(1)

    def _next_file_key(self, viewer=None):
        self.next_file()

    def _previous_file_key(self, viewer=None):
        self.previous_file()

    def _save_key(self, viewer=None):
        self.save_annotations()

    def closeEvent(self, event):
        for key in self._bound_keys:
            try:
                self.viewer.bind_key(key, None, overwrite=True)
            except Exception:
                pass
        if self._pending_write:
            self._save_master_sync()
        super().closeEvent(event)
```

- [ ] **Step 3: Verify the module imports**

Run: `micromamba run -n napari python -c "from napari_towbintools_annotator.panoptic_annotator import PanopticAnnotatorWidget; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Run the existing test file (no regressions)**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/test_panoptic.py -q`
Expected: PASS (the pure-function and project tests still green)

- [ ] **Step 5: Commit**

```bash
git add src/napari_towbintools_annotator/panoptic_annotator.py
git commit -m "Add PanopticAnnotatorWidget"
```

---

## Task 7: Wire panoptic into the project creator and loader

**Files:**
- Modify: `src/napari_towbintools_annotator/project_creator.py`

- [ ] **Step 1: Import the panoptic pieces**

In `src/napari_towbintools_annotator/project_creator.py`, update the local imports block:

```python
from .annotators import ClassificationAnnotatorWidget
from .panoptic_annotator import PanopticAnnotatorWidget
from .project import ClassificationProject
from .project import PanopticProject
from .project import Project
```

- [ ] **Step 2: Route panoptic in `create_annotator_widget`**

Replace the body of `create_annotator_widget` (currently classification-only) with:

```python
def create_annotator_widget(napari_viewer, project, parent=None):
    if project.project_type == "classification":
        return ClassificationAnnotatorWidget(
            napari_viewer, project, parent=parent
        )
    if project.project_type == "panoptic":
        return PanopticAnnotatorWidget(napari_viewer, project, parent=parent)
    raise NotImplementedError(
        f"Unsupported project type: {project.project_type}"
    )
```

- [ ] **Step 3: Show the right creator controls for panoptic**

Replace `toggle_project_type_options` with:

```python
    def toggle_project_type_options(self):
        is_classification = self.project_type_classification.isChecked()
        is_panoptic = self.project_type_panoptic.isChecked()

        self.classification_options_layout.gbox.setVisible(
            is_classification or is_panoptic
        )
        self.display_mode_group.gbox.setVisible(is_classification)

        if is_panoptic:
            # Panoptic always needs references (images) and segmentations.
            self.mask_dir_selector_widget.setVisible(True)
            self.data_selection_widget.setVisible(True)
        elif is_classification:
            self._toggle_mask_dir_selector()
        else:
            self.mask_dir_selector_widget.setVisible(False)
            self.data_selection_widget.setVisible(True)
```

- [ ] **Step 4: Rename the class options group so it reads sensibly for panoptic**

Find this line in `__init__`:

```python
        self.classification_options_layout = VHGroup(
            "Classification", orientation="G"
        )
```

and change the title to:

```python
        self.classification_options_layout = VHGroup(
            "Annotation Classes", orientation="G"
        )
```

- [ ] **Step 5: Route panoptic in `create_project`**

In `create_project`, the current structure is `if project_type == "classification": ... else: <other>`. Insert a `panoptic` branch between them. Replace:

```python
        else:
            data_directories = list(self.data_directories)

            def task(status):
                return self._run_other_creation(
                    project_name,
                    image_type,
                    project_type,
                    project_dir,
                    data_directories,
                    copy_data,
                    status,
                )
```

with:

```python
        elif project_type == "panoptic":
            if not self.mask_directories:
                self._show_error(
                    "No segmentation (mask) directories selected."
                )
                return
            classes = self._get_classes()
            if not classes:
                self._show_error("No classes defined.")
                return

            data_directories = list(self.data_directories)
            mask_directories = list(self.mask_directories)

            def task(status):
                return self._run_panoptic_creation(
                    project_name,
                    image_type,
                    project_dir,
                    data_directories,
                    mask_directories,
                    classes,
                    copy_data,
                    status,
                )

        else:
            data_directories = list(self.data_directories)

            def task(status):
                return self._run_other_creation(
                    project_name,
                    image_type,
                    project_type,
                    project_dir,
                    data_directories,
                    copy_data,
                    status,
                )
```

- [ ] **Step 6: Add `_run_panoptic_creation`**

Add this method to `ProjectCreatorWidget`, right after `_run_classification_creation`:

```python
    def _run_panoptic_creation(
        self,
        project_name,
        image_type,
        project_dir,
        data_directories,
        mask_directories,
        classes,
        copy_data,
        status,
    ):
        os.makedirs(project_dir, exist_ok=True)
        annotations_save_dir = os.path.join(project_dir, "annotations")
        os.makedirs(annotations_save_dir, exist_ok=True)

        if copy_data:
            local_data_dir = os.path.join(project_dir, "data")
            os.makedirs(local_data_dir, exist_ok=True)
            status.emit("Copying data...")
            data_directories = self._copy_data_directories_static(
                data_directories, local_data_dir, status
            )

        status.emit("Scanning files...")
        reference_files, segmentation_files = scan_panoptic_files(
            data_directories, mask_directories
        )

        status.emit("Writing annotation file...")
        annotation_df = pd.DataFrame(
            {
                "Reference": reference_files,
                "Segmentation": segmentation_files,
                "Annotation": [""] * len(reference_files),
            }
        )
        annotation_df_path = os.path.join(
            annotations_save_dir, "annotations.csv"
        )
        annotation_df.to_csv(annotation_df_path, index=False)

        project = PanopticProject(
            name=project_name,
            image_type=image_type,
            annotation_directories=["annotations"],
            annotation_df_path=os.path.relpath(
                annotation_df_path, project_dir
            ),
            data_directories=data_directories,
            mask_directories=mask_directories,
            classes=classes,
            project_dir=project_dir,
        )
        project.save()
        return project_dir
```

- [ ] **Step 7: Verify imports and the module load**

Run: `micromamba run -n napari python -c "import napari_towbintools_annotator.project_creator as pc; print(hasattr(pc.ProjectCreatorWidget, '_run_panoptic_creation'))"`
Expected: `True`

- [ ] **Step 8: Run the full test suite**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/ -q`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/napari_towbintools_annotator/project_creator.py
git commit -m "Wire panoptic project type into creator and loader"
```

---

## Task 8: Viewer-backed widget smoke test

End-to-end check that the widget loads a real project, opens layers, and saves a correct per-image CSV + updates the master index. Builds `napari.Viewer(show=False)` directly (no `pytest-qt` available).

**Files:**
- Test: `src/napari_towbintools_annotator/_tests/test_panoptic.py`

- [ ] **Step 1: Write the test**

Append to `test_panoptic.py`:

```python
import tifffile

from napari_towbintools_annotator.panoptic_annotator import (
    PanopticAnnotatorWidget,
)


def test_panoptic_widget_load_and_save(tmp_path):
    import napari

    project_dir = tmp_path / "proj"
    annotations_dir = project_dir / "annotations"
    annotations_dir.mkdir(parents=True)

    reference = np.zeros((10, 10), dtype=np.uint8)
    segmentation = np.zeros((10, 10), dtype=np.uint16)
    segmentation[2:4, 2:4] = 5
    ref_path = tmp_path / "img.tif"
    seg_path = tmp_path / "img_seg.tif"
    tifffile.imwrite(str(ref_path), reference)
    tifffile.imwrite(str(seg_path), segmentation)

    pd.DataFrame(
        {
            "Reference": [str(ref_path)],
            "Segmentation": [str(seg_path)],
            "Annotation": [""],
        }
    ).to_csv(annotations_dir / "annotations.csv", index=False)

    project = PanopticProject(
        name="p",
        image_type="multichannel",
        annotation_directories=["annotations"],
        annotation_df_path="annotations/annotations.csv",
        data_directories=[str(tmp_path)],
        mask_directories=[str(tmp_path)],
        classes=["a", "b"],
        project_dir=str(project_dir),
    )

    viewer = napari.Viewer(show=False)
    try:
        widget = PanopticAnnotatorWidget(viewer, project)
        assert widget.file_list_widget.count() == 1
        assert widget._annotation_layer is not None

        # Drop one point on instance 5, colored as class 0 ("a").
        widget._annotation_layer.data = np.array([[3, 3]])
        widget._annotation_layer.face_color = np.array(
            [widget.class_id_to_color[0]], dtype=float
        )
        widget.save_annotations()
        widget._save_master_sync()
    finally:
        viewer.close()

    out_csv = annotations_dir / "img.csv"
    assert out_csv.exists()
    saved = pd.read_csv(out_csv)
    assert int(saved.loc[0, "Label"]) == 5
    assert int(saved.loc[0, "ClassID"]) == 0
    assert saved.loc[0, "Class"] == "a"

    master = pd.read_csv(annotations_dir / "annotations.csv")
    assert str(master.loc[0, "Annotation"]) == str(out_csv)
```

- [ ] **Step 2: Run the test**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/test_panoptic.py::test_panoptic_widget_load_and_save -q`
Expected: PASS

If it fails on `face_color` not surviving the `data` assignment (napari version quirk), set color first then data, or assign both inside a single `add`/refresh — adjust the test and, if needed, `_replay_annotations`/`save_annotations` consistently. The expected saved values (Label 5, ClassID 0, Class "a") are the contract.

- [ ] **Step 3: Commit**

```bash
git add src/napari_towbintools_annotator/_tests/test_panoptic.py
git commit -m "Add panoptic widget smoke test"
```

---

## Task 9: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `micromamba run -n napari python -m pytest src/napari_towbintools_annotator/_tests/ -v`
Expected: all tests PASS (existing `test_project.py` + new `test_panoptic.py`)

- [ ] **Step 2: Lint the changed files**

Run: `micromamba run -n napari ruff check src/napari_towbintools_annotator/`
Expected: no errors (fix any reported issues, then re-run)

- [ ] **Step 3: Manual launch smoke check**

Run: `micromamba run -n napari python -c "import napari; from napari_towbintools_annotator.project_creator import TowbintoolsAnnotatorWidget; v=napari.Viewer(show=False); w=TowbintoolsAnnotatorWidget(v); v.window.add_dock_widget(w); print('widget docked ok'); v.close()"`
Expected: `widget docked ok`

- [ ] **Step 4: Final commit (if any lint fixes were made)**

```bash
git add -A
git commit -m "Lint fixes for panoptic integration"
```

---

## Self-Review notes

- **Spec coverage:** PanopticProject (Task 3) ✔; auto colors by index (Task 2/6) ✔; per-image CSV + master index (Tasks 5/6/7) ✔; creator wiring incl. Panoptic radio path, mask+class visibility (Task 7) ✔; dropped layer pickers / auto layer management (Task 6) ✔; keybindings up/down/j/h with unbind on close (Task 6) ✔; 3D plane handling via segmentation ndim (Tasks 4/6) ✔; error handling: count mismatch (Task 5), unknown color/empty mask skipped (Task 4), missing CSV → empty points (Task 6) ✔; deps incl. scikit-image (Task 1) ✔; tests (Tasks 2–8) ✔.
- **Spec deviation (intentional):** `magicgui` was listed in the spec's dependency section but is **not** added — the layer-picker widgets that used it are dropped, so it is unused. Noted in Task 1.
- **Type consistency:** `points_to_rows` / `rows_to_points` / `nearest_class_id` signatures match between Task 4 (definition + tests) and Task 6 (widget calls). `scan_panoptic_files` signature matches between Task 5 and its use in Task 7. `class_id_to_color` / `class_id_to_name` / `class_name_to_color` names are consistent across Task 6.
- **No placeholders.**
