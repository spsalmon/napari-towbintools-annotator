# Panoptic Annotator Integration — Design

Date: 2026-06-10

## Goal

Integrate the standalone [`napari-panoptic-annotator`](https://github.com/spsalmon/napari-panoptic-annotator)
into `napari-towbintools-annotator` as a fully supported **panoptic** project
type, reusing the existing project framework (project creation, loading, file
list, navigation, save) rather than dropping in a parallel widget.

The project creator already exposes a "Panoptic" radio button, but the path is
unimplemented: `create_annotator_widget` raises `NotImplementedError` for any
non-classification project type. This work wires that path up.

## Decisions (from brainstorming)

- **Full integration** into the project framework (not a drop-in widget).
- **Classes via the existing class list; colors auto-derived** from the shared
  palette by class index. No per-class color picker, no colors stored on disk.
- **Per-image CSVs (your existing format) + a master index CSV** to drive the
  file list, resume, and done-status.
- The `magicgui` layer-picker dropdowns from the standalone are **dropped** —
  layers are opened and managed automatically per file.

## Data model

New `PanopticProject(Project)` subclass in `project.py`, mirroring
`ClassificationProject`:

```
PanopticProject:
  name: str
  image_type: str               # multichannel | zstack | time_series
  project_type = "panoptic"
  classes: list[str]            # ordered; ClassID = index in this list
  data_directories: list        # reference / image directories
  mask_directories: list        # segmentation (labels) directories
  annotation_df_path: str       # relative path to the master index CSV
  project_dir: str
```

Colors are **not** persisted. A class's color is `palette[classes.index(name) %
len(palette)]`, so `ClassID` and color stay stable across sessions. This is the
same scheme `ClassificationAnnotatorWidget` already uses for file-list
highlighting.

`PanopticProject` is registered in `Project.load`'s dispatch dict next to
`classification`, and implements `save()` / `load(project_dir, project_data)`.

### On-disk layout

```
<project_dir>/
  project.yaml
  annotations/
    annotations.csv             # master index
    <reference_basename>.csv    # per-image annotation, written on save
```

**Master `annotations.csv`** columns:

| Column       | Meaning                                                        |
|--------------|----------------------------------------------------------------|
| Reference    | absolute path to the reference image                           |
| Segmentation | absolute path to the matching segmentation (labels) image      |
| Annotation   | absolute path to the per-image CSV, or `""` if not yet saved   |

Done-status is derived from `Annotation != ""`.

**Per-image `<reference_basename>.csv`** columns (your existing standalone
format, unchanged so downstream tooling keeps working):

| Column  | Meaning                                                      |
|---------|--------------------------------------------------------------|
| (plane) | only in 3D: the first axis name (e.g. `Z`), the point's slice |
| Label   | the label/instance value under the point                     |
| ClassID | class index                                                  |
| Class   | class name                                                   |

## Components

### 1. `project.py` — `PanopticProject`

- Constructor validates that `classes` is non-empty and that both
  `data_directories` and `mask_directories` are provided.
- `save()` writes all fields to `project.yaml`.
- `load(project_dir, project_data)` reconstructs from the parsed yaml.
- Added to `Project.load` dispatch: `{"classification": ..., "panoptic": PanopticProject.load}`.

### 2. `colors.py` — shared color helpers (new)

Extracted so both annotators share one source of truth:

- `CLASS_PALETTE` — the list of hex colors currently inlined in `annotators.py`.
- `hex_to_rgba_float(hex_str) -> tuple[float, float, float, float]` — converts a
  palette entry to the RGBA floats napari `face_color` expects.
- `class_color(classes, name)` helper returning the palette hex for a class
  name by its index.

`annotators.py` is updated to import `CLASS_PALETTE` from here instead of
defining `_CLASS_PALETTE` locally (behavior unchanged).

### 3. `project_creator.py` — creator wiring

- When **Panoptic** is selected in `toggle_project_type_options`:
  - show the image-directory selector (references),
  - show the mask-directory selector (segmentations),
  - show the **Classes** group (today it is classification-only; generalize the
    visibility so it shows for classification *and* panoptic),
  - hide the classification-only Display Mode group.
- New `_run_panoptic_creation(project_name, image_type, project_dir,
  data_directories, mask_directories, classes, copy_data, status)`:
  - validates at least one reference dir, one mask dir, and one class,
  - natsorted-scans reference and segmentation files,
  - raises `ValueError` on count mismatch (same guard as classification "both"),
  - builds the master `annotations.csv` (`Reference`, `Segmentation`,
    `Annotation=""`),
  - constructs and `save()`s a `PanopticProject`.
- `create_project` routes `project_type == "panoptic"` to the new task instead
  of `_run_other_creation`.
- `create_annotator_widget` gains a `panoptic` branch returning
  `PanopticAnnotatorWidget`.

### 4. `panoptic_annotator.py` — `PanopticAnnotatorWidget` (new file)

Kept in its own module so `annotators.py` stays focused.

Constructor `(napari_viewer, project, parent=None)`:
- Loads the master index into `self.annotation_df`.
- Builds the file list from `Reference` basenames, with done-status coloring
  (annotated rows tinted, mirroring the classification widget).
- Computes `class_values` (`name -> index`), `class_colors`
  (`name -> rgba float`), and reverse maps (`color -> id`, `id -> color`,
  `id -> name`).
- Class radio buttons built from `project.classes`, each tinted its color.
- Resume index = first row whose `Annotation` is empty (fall back to 0).
- Binds keys on the viewer: `up`/`down` cycle class, `j`/`h` prev/next file.
- Loads the current file.

Behavior (ported from the standalone, with framework integration):
- `_load_file()`: clear layers → open reference image → open segmentation as
  Labels (opacity 0.5) → create a Points "Annotations" layer sized 2D/3D from
  the segmentation `ndim`; if the row's `Annotation` CSV exists, replay it.
- `_replay_annotations(csv_path)`: for each row, find the label's centroid via
  `regionprops` and add a point colored by `ClassID`. Guards empty masks and
  unknown class ids (skip + log).
- Class selection updates the points layer's `current_face_color`.
- `cycle_class_up/down`: rotate selected class and sync the radio buttons.
- `next_file` / `previous_file` / `choose_file_from_list`: clamp to range,
  set the list row, load.
- `save_annotations()`: convert points → rows (`Label`, `ClassID`, `Class`,
  plus the plane column in 3D) by reading the label value under each rounded
  point and mapping its `face_color` → class; write
  `annotations/<reference_basename>.csv`; update the master index's
  `Annotation` cell and persist it; recolor the file-list item as done.
- `closeEvent`: unbind the viewer keys and flush any pending master-index write.

Axes order is derived from the segmentation layer's `ndim` (`YX` / `ZYX`); no
manual axes-order text field (the standalone's field is dropped along with the
layer pickers).

## Data flow

```
Create panoptic project
  -> scan reference + segmentation dirs (count-matched)
  -> write master annotations.csv (Annotation cells empty)
  -> save project.yaml
  -> load project -> PanopticAnnotatorWidget

Annotate a file
  -> open reference + segmentation (+ existing per-image CSV if present)
  -> user drops class-colored points on instances
  -> Save: points -> per-image CSV; master index Annotation cell updated;
     file-list item marked done

Resume
  -> master index read; first un-annotated row selected
```

## Error handling

- Reference/segmentation **count mismatch** at creation → `ValueError`, shown via
  the creator's existing `_show_error` dialog.
- Missing classes / dirs at creation → `ValueError` via the same dialog.
- Point on background or **unknown color** at save → skip that point, log a
  message (current standalone behavior).
- **Empty mask** for a label on replay → guard the `regionprops` call, skip.
- Missing per-image CSV on load → just open reference + segmentation with an
  empty points layer.

## Dependencies

Add to `pyproject.toml` `dependencies`:
- `scikit-image` — `regionprops` for centroid placement.
- `magicgui` — points/labels widget helpers (also a napari transitive dep).
- `pandas`, `tifffile`, `imageio` — already imported by the existing
  classification annotator but currently undeclared; add them for correctness.

## Testing

Run in the `napari` micromamba environment (`micromamba run -n napari pytest`),
which has napari 0.7.0, scikit-image, magicgui, pandas, tifffile, imageio.

- `PanopticProject` save → load round-trip preserves all fields and dispatches
  correctly through `Project.load`.
- Panoptic creation: builds a master `annotations.csv` with matched
  reference/segmentation rows and empty `Annotation` cells; raises `ValueError`
  on count mismatch.
- Widget (using napari's `make_napari_viewer` fixture with synthetic label
  arrays):
  - points → CSV yields correct `Label` / `ClassID` / `Class` rows (2D and 3D).
  - CSV → points places points at label centroids with the right colors.
  - class cycling updates the selected class and radio state.
  - `next_file` / `previous_file` clamp at the ends and update the current row.
- Existing classification tests continue to pass after the `colors.py`
  extraction.

## Out of scope

- Editing/renaming classes after project creation.
- Per-class custom colors / color picker UI.
- Changing the per-image CSV schema (kept identical to the standalone).
- Keypoint project type (separate radio, still unimplemented — untouched).
