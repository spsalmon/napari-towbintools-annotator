"""Shared fixtures.

Widget tests run against a ``napari.components.ViewerModel`` rather than a
full ``napari.Viewer``: the model carries the same layer / dims / keybinding
API the widgets use, but no vispy canvas, so no OpenGL context is needed.
That keeps the suite runnable on headless CI runners (and macOS runners,
whose GL setup is the usual source of flaky napari tests).
"""

import os

# Must be set before Qt is imported anywhere. An explicit value from the
# environment (e.g. a developer wanting to watch widgets) wins.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def viewer(qapp):
    from napari.components import ViewerModel

    model = ViewerModel()
    try:
        yield model
    finally:
        # Removing layers disconnects the widgets' layer callbacks, which is
        # what Viewer.close() did for the old GL-backed tests.
        model.layers.clear()
        qapp.processEvents()
