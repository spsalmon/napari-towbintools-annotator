# napari-align-annotator

[![License BSD-3](https://img.shields.io/github/license/spsalmon/napari-align-annotator?color=green)](https://github.com/spsalmon/napari-align-annotator/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://python.org)
[![tests](https://github.com/spsalmon/napari-align-annotator/actions/workflows/test_and_deploy.yml/badge.svg?branch=main)](https://github.com/spsalmon/napari-align-annotator/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/spsalmon/napari-align-annotator/branch/main/graph/badge.svg)](https://codecov.io/gh/spsalmon/napari-align-annotator)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A [napari] plugin for building annotated datasets for
[ALIGN](https://github.com/spsalmon/align_pipeline).
Annotation work is organised into *projects*: a directory holding a
`project.yaml` descriptor and a master annotation CSV, so a session can be
closed and resumed at any time.

----------------------------------

## Installation

Install the plugin into an environment that has napari and a Qt binding:

    pip install napari-align-annotator

or, to get napari and PyQt together:

    pip install "napari-align-annotator[all]"

The plugin can also be installed from napari's plugin manager
(`Plugins > Install/Uninstall Plugins...`).

To install the latest development version:

    pip install git+https://github.com/spsalmon/napari-align-annotator.git

## Usage

Open napari and start the widget from
`Plugins > ALIGN Annotator`. From there you can either
**Create Project** or **Load Project** (pick a directory containing a
`project.yaml`).

When creating a project you choose a name, a location, the image type
(multichannel, z-stack or time series), the data directories, and
optionally whether the data should be copied into the project. Two project
types are currently supported:

### Classification

Assign one class to each whole image.

- Define the list of classes, and choose whether to display the image, its
  mask, or both (masks require mask directories).
- Click a class button to label the current image and move to the next
  one. **Ignore** removes the current image from the project.
- Labels are written to `annotations/annotations.csv`, with one row per
  image (`ImagePath`, `MaskPath`, `Class`).

### Panoptic

Assign a class to each segmented instance of an image.

- Requires raw image directories and matching segmentation (label)
  directories, plus a list of classes.
- Select a class, then place a point on an instance to label it. For
  z-stacks, instances are linked across planes (stitched by IoU, threshold
  adjustable with **Re-stitch**) so a label placed on one plane propagates to
  the rest of the object.
- Keyboard shortcuts: `J` next image, `H` previous image, `S` save.
- The master CSV lists `Reference`, `Segmentation` and `Annotation` paths;
  each per-image annotation file has one row per labelled instance
  (`Label`, `ClassID`, `Class`, plus `Z` and `InstanceID` for z-stacks).
  `Label` is the value of the instance in the segmentation file.

The *Keypoint* project type is shown in the creator but is not supported
yet.

## Contributing

Contributions are very welcome. Tests can be run with [tox] or directly with
`pytest` after `pip install -e ".[testing]"`; please ensure the coverage at
least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-align-annotator" is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed
description.

This [napari] plugin was generated with [copier] using the
[napari-plugin-template].

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[file an issue]: https://github.com/spsalmon/napari-align-annotator/issues
[tox]: https://tox.readthedocs.io/en/latest/
