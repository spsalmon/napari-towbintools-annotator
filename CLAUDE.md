# napari-align-annotator

A napari plugin providing project-based image annotation workflows for use with
`align_pipeline`. A *project* is a directory holding a `project.yaml`
descriptor plus a master annotation CSV; opening it launches the annotator
widget matching its `project_type`.

## Environment

**All Python commands — tests, linting, scratch scripts — must run in the
`ALIGN` micromamba environment.** Never use bare `python`/`pytest` (there
is no `python` on `PATH`), and do not use the `napari` env for this project.

```bash
micromamba run -n align_pipeline python -m pytest src/napari_align_annotator/_tests/ -q
```

The env already has napari 0.9 + PyQt6, `napari_guitils`, `natsort`, and this
package installed editable (`pip install -e . --no-deps`). If an import goes
missing, install it into `ALIGN` rather than switching envs.

Run the full suite before claiming work is done, and paste the real output —
never assert that tests pass without having seen them pass.

## Dependencies

Runtime deps are declared in `pyproject.toml`. Keep that list in sync when
adding an import.

Prefer **not** adding heavy new runtime dependencies. Where a small,
well-understood algorithm is all that is needed from a large package,
reimplement it against numpy/scikit-image rather than taking the dependency —
this is a deliberate choice for this project, not an oversight.

## Layout

```
src/napari_align_annotator/
├── project.py               # Project (base), ClassificationProject, PanopticProject
│                            #   — YAML load/save; Project.load dispatches on project_type
├── project_creator.py       # ALIGNAnnotatorWidget (napari entry point),
│                            #   ProjectCreatorWidget, scan_panoptic_files,
│                            #   create_annotator_widget (project_type -> widget)
├── classification_annotator.py  # whole-image class annotation
├── panoptic_annotator.py    # per-instance (point-on-label) class annotation
├── colors.py                # CLASS_PALETTE, hex_to_rgba_float, class_hex
└── _tests/
```

The single napari contribution is
`project_creator:ALIGNAnnotatorWidget` (see `napari.yaml`). New project
types are reached through it — they do **not** add napari contributions.

### Adding a project type

1. Subclass `Project` in `project.py` with `save()` / `load()`, and register it
   in the `Project.load` dispatch dict.
2. Add a `_run_<type>_creation` method plus UI visibility wiring in
   `project_creator.py`, and route it in `create_project`.
3. Add the annotator widget and route it in `create_annotator_widget`.

## Conventions

- **Pure logic outside the widget.** Conversion, geometry, and label logic live
  as module-level functions (`points_to_rows`, `rows_to_points`,
  `nearest_class_id`, …) so they are testable without a live napari viewer.
  Widget classes should hold Qt/viewer state and delegate the real work.
- **Annotation CSV columns are a public interface.** Downstream
  `align_pipeline` code reads them. Add columns additively; do not rename
  or repurpose existing ones, and do not change what a `Label` value refers to
  (it indexes the segmentation file on disk).
- Style: black + ruff, **line length 79**, target py310. `E501` is ignored —
  let black wrap.
- Tests use plain `pytest` with `tmp_path`; there is no `pytest-qt`. Widget
  tests take the `viewer` fixture from `_tests/conftest.py` — a
  `napari.components.ViewerModel` (no vispy canvas, so no OpenGL) — and the
  conftest defaults `QT_QPA_PLATFORM=offscreen`. Do not construct
  `napari.Viewer` in tests: it needs a GL context, which headless and macOS
  CI runners do not reliably have.
- `napari` and the Qt binding are test/`all` extras, not runtime deps; a
  clean `pip install .[testing]` must be enough to run the suite.

## Git

Do not commit Claude Code / superpowers artifacts. `docs/` and `.claude/` are
gitignored; this `CLAUDE.md` is likewise not to be committed unless explicitly
asked.
