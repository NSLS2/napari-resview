# tests/test_resview_widget.py

from __future__ import annotations

import napari
from qtpy import QtWidgets

from napari_resview.resview_widget import ResviewDockWidget


def test_resview_dockwidget_constructs_and_has_tabs(
    make_napari_viewer, monkeypatch
):
    """
    Ensures ResviewDockWidget can be constructed both with an explicit viewer
    and with viewer=None (using napari.current_viewer fallback), and that the
    3 vertical tabs exist with expected labels.

    Requires napari's pytest fixture: make_napari_viewer
    (from napari plugin testing helpers).
    """
    viewer = make_napari_viewer()

    # ---- Case 1: explicit viewer works
    w1 = ResviewDockWidget(viewer=viewer)
    assert w1.viewer is viewer
    assert isinstance(w1, QtWidgets.QWidget)

    # Tab widget exists and has 3 tabs labeled Data / Build / View
    tabs = w1.findChild(QtWidgets.QTabWidget)
    assert tabs is not None, "Expected a QTabWidget in the dock widget"
    assert tabs.count() == 3
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Data",
        "Build",
        "View",
    ]

    # Sanity-check one widget on the Data tab
    assert w1.data_file_w.label == "DATA folder"

    # ---- Case 2: viewer=None works if napari.current_viewer() returns one
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)
    w2 = ResviewDockWidget(viewer=None)
    assert w2.viewer is viewer

    # ---- Cleanup
    w1.close()
    w2.close()
