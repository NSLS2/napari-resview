#!/usr/bin/env python3
"""
ResView (ResviewDockWidget) — napari dock widget with YAML persistence + ISR/CMS profiles

What’s implemented in this version:

1) Two YAML profile groups (ISR + CMS)
   - YAML stores: active_profile + profiles:{ISR:{...}, CMS:{...}}
   - Switching Loader dropdown:
       a) saves current UI -> old profile
       b) loads new profile -> UI
       c) persists active_profile

2) UI layout changes requested earlier
   - Crop controls moved to the Data tab
   - UB controls moved to the Build tab
   - Crop is applied when "Crop from ROI" button is clicked:
     * Updates DataFrame intensity with cropped frames
     * Updates ExperimentSetup with new detector size and beam center
     * Updates intensity viewer display

3) Pipeline behavior changes
   - Data is cropped at the source (DataFrame and setup) when "Crop from ROI" is clicked
   - RSMBuilder receives already-cropped data, no additional cropping needed
   - Downstream operations (build/regrid/view/export) work with cropped data directly

4) View tab additions
   - Stop: cancels the in-flight load worker and stops Run All chaining
   - Refresh: re-runs View with current viewer settings
   - Run All button size adjusted to fit the row better

5) Async Load (thread_worker) + async-safe Run All chaining preserved.

NOTE:
- This file assumes these imports exist in your package:
    from .data_io import RSMDataLoader_ISR, RSMDataloader_CMS, write_rsm_volume_to_vtr
    from .data_viz import IntensityNapariViewer, RSMNapariViewer
    from .rsm3d import RSMBuilder
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import re
from typing import Any

import napari
import numpy as np
import yaml
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    FloatSpinBox,
    Label,
    LineEdit,
    PushButton,
    SpinBox,
    TextEdit,
)
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info, show_warning
from napari.viewer import Viewer
from qtpy import QtCore, QtGui, QtWidgets

from .data_io import (
    RSMDataloader_CMS,
    RSMDataLoader_ISR,
    write_rsm_volume_to_vtr,
)
from .data_viz import IntensityNapariViewer, RSMNapariViewer
from .rsm3d import RSMBuilder

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Optional icon (does not change napari theme)
# -----------------------------------------------------------------------------
APP_ICON_PATH = (pathlib.Path(__file__).parent / "resview_icon.png").resolve()


def load_app_icon() -> QtGui.QIcon | None:
    if APP_ICON_PATH.is_file():
        icon = QtGui.QIcon(str(APP_ICON_PATH))
        return icon if not icon.isNull() else None
    return None


# -----------------------------------------------------------------------------
# YAML persistence (two-profile schema)
# -----------------------------------------------------------------------------
DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
os.environ.setdefault(
    DEFAULTS_ENV,
    str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
)


def yaml_path() -> str:
    p = os.environ.get(DEFAULTS_ENV, "").strip()
    if p:
        return os.path.abspath(os.path.expanduser(p))
    return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")


def _default_profile_doc(loader_type: str) -> dict[str, Any]:
    # One full profile (all sections) so ISR and CMS can diverge fully.
    return {
        "data": {
            "loader_type": loader_type,
            "data_file": None,
            "scans": "",
            # ISR-only (still stored in profile so switching restores it)
            "spec_file": None,
            "only_hkl": False,
        },
        "ExperimentSetup": {
            "distance": None,
            "pitch": None,
            "ycenter": None,
            "xcenter": None,
            "xpixels": None,
            "ypixels": None,
            "energy": None,
            "wavelength": None,
        },
        "Crystal": {"ub": "1 0 0\n0 1 0\n0 0 1"},
        "build": {
            "ub_includes_2pi": False,
            "center_is_one_based": False,
            "sample_axes": "x+, y+, z-",
            "detector_axes": "x+",
        },
        "crop": {
            "enable": False,
            "y_min": 0,
            "y_max": 0,
            "x_min": 0,
            "x_max": 0,
        },
        "regrid": {
            "space": "hkl",
            "grid_shape": "200,*,*",
            "fuzzy": False,
            "fuzzy_width": 0.0,
            "normalize": "mean",
        },
        "view": {
            "log_view": False,
            "cmap": "inferno",
            "rendering": "attenuated_mip",
            "contrast_lo": 1.0,
            "contrast_hi": 99.8,
        },
        "export": {"vtr_path": None},
    }


def ensure_yaml(path: str) -> None:
    if os.path.isfile(path):
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    seed: dict[str, Any] = {
        "active_profile": "ISR",
        "profiles": {
            "ISR": _default_profile_doc("ISR"),
            "CMS": _default_profile_doc("CMS"),
        },
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(seed, f, sort_keys=False)
    except (OSError, UnicodeEncodeError, yaml.YAMLError):
        logger.exception("Failed to create defaults YAML at %s", path)


def load_yaml(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        return doc or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        logger.warning("Failed to load YAML %s: %s", path, e)
        return {}


def save_yaml(path: str, doc: dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
    except (OSError, UnicodeEncodeError, yaml.YAMLError) as e:
        logger.exception("Failed to write YAML: %s", e)
        show_error(f"Failed to write YAML: {e}")


def as_path_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return os.fspath(v)
    except TypeError:
        return str(v)


# -----------------------------------------------------------------------------
# Parsing / formatting
# -----------------------------------------------------------------------------
def format_ub_matrix(ub: Any) -> str:
    if ub is None:
        return ""
    try:
        arr = np.asarray(ub, dtype=float)
    except (TypeError, ValueError):
        return str(ub)
    if arr.ndim == 0:
        return f"{arr.item():.6g}"
    if arr.ndim == 1:
        return " ".join(f"{v:.6g}" for v in arr)
    if arr.ndim == 2:
        return "\n".join(" ".join(f"{v:.6g}" for v in row) for row in arr)
    return np.array2string(arr, precision=6, separator=" ")


def parse_ub_matrix(text: str) -> np.ndarray | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    rows: list[list[float]] = []
    for line in stripped.splitlines():
        parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
        if parts:
            rows.append([float(p) for p in parts])
    if not rows:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("UB rows must have equal length.")
    return np.array(rows, dtype=float)


def parse_scan_list(text: str) -> list[int]:
    if not text or not text.strip():
        return []
    out: set[int] = set()
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = a.strip(), b.strip()
            if a.isdigit() and b.isdigit():
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
            else:
                raise ValueError(f"Bad scan range: '{part}'")
        else:
            if part.isdigit():
                out.add(int(part))
            else:
                raise ValueError(f"Bad scan id: '{part}'")
    return sorted(out)


def parse_axes_list(text: str) -> list[str]:
    if not text:
        return []
    return [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]


def parse_grid_shape(text: str) -> tuple[int | None, int | None, int | None]:
    if text is None:
        return (None, None, None)
    s = text.strip()
    if not s:
        return (None, None, None)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 1:
        parts += ["*", "*"]
    if len(parts) != 3:
        raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")

    def one(p: str) -> int | None:
        if p in ("*", "", None):
            return None
        if not p.isdigit():
            raise ValueError(f"Grid size must be integer or '*', got '{p}'")
        v = int(p)
        if v <= 0:
            raise ValueError("Grid sizes must be > 0")
        return v

    return tuple(one(p) for p in parts)  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def hsep(height: int = 10) -> Label:
    """Safe separator inside magicgui.Container (do NOT use Qt QFrame there)."""
    w = Label(value="")
    with contextlib.suppress(AttributeError):
        w.native.setFixedHeight(height)
        w.native.setMinimumHeight(height)
    return w


def make_group(
    title: str, inner_widget: QtWidgets.QWidget
) -> QtWidgets.QGroupBox:
    box = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(12, 12, 12, 12)
    lay.setSpacing(8)
    lay.addWidget(inner_widget)
    return box


def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    wrapper = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(wrapper)
    v.setContentsMargins(8, 8, 8, 8)
    v.setSpacing(8)
    v.addWidget(inner)
    sc = QtWidgets.QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QtWidgets.QFrame.NoFrame)
    sc.setWidget(wrapper)
    return sc


def set_file_button_symbol(
    fe: FileEdit, symbol: str = "📂"
) -> QtWidgets.QPushButton | None:
    native = getattr(fe, "native", None)
    if native is None:
        return None
    for btn in native.findChildren(QtWidgets.QPushButton):
        with contextlib.suppress(AttributeError, TypeError):
            btn.setText(symbol)
            btn.setMinimumWidth(32)
            btn.setMaximumWidth(36)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
        return btn
    return None


def attach_dir_picker(fe: FileEdit, button: QtWidgets.QPushButton) -> None:
    def pick_dir() -> None:
        start = as_path_str(fe.value).strip() or os.path.expanduser("~")
        path = QtWidgets.QFileDialog.getExistingDirectory(
            button, "Select folder", start
        )
        if path:
            fe.value = path
        else:
            show_warning("No folder selected.")

    button.clicked.connect(pick_dir)


# -----------------------------------------------------------------------------
# Dock widget
# -----------------------------------------------------------------------------
class ResviewDockWidget(QtWidgets.QWidget):
    """ResView UI as a napari dock widget (tabs: Data/Build/View)."""

    def __init__(
        self,
        viewer: Viewer | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        if viewer is None:
            try:
                viewer = napari.current_viewer()
            except RuntimeError:
                viewer = None
        if viewer is None:
            raise TypeError("ResviewDockWidget requires a napari Viewer.")

        self.viewer: Viewer = viewer
        self._icon = load_app_icon()

        # YAML init
        self._yaml_path = yaml_path()
        ensure_yaml(self._yaml_path)
        self._ydoc: dict[str, Any] = load_yaml(self._yaml_path)
        self._migrate_yaml_schema_if_needed()
        self._profile_loading = (
            False  # guard against recursive saves while swapping
        )
        self._switching_profile = (
            False  # prevent widget saves during profile switch
        )
        self._auto_save_enabled = (
            False  # only save to YAML after successful data load
        )

        # Runtime state
        self._state: dict[str, Any] = {
            "loader": None,
            "setup": None,
            "ub": None,
            "df": None,
            "builder": None,
            "grid": None,
            "edges": None,
            "intensity_frames": None,
            "rsm_viewer": None,
            "intensity_viewer": None,
        }

        # Worker + pipeline control
        self._load_worker_instance = None
        self._run_all_pending = False

        # ---------------------------------------------------------------------
        # DATA TAB (loader inputs + setup + crop)
        # ---------------------------------------------------------------------
        self.loader_type_w = ComboBox(
            label="Loader Profile", choices=["ISR", "CMS"]
        )

        # Common required
        self.data_file_w = FileEdit(mode="r", label="DATA folder")
        data_btn = set_file_button_symbol(self.data_file_w, "📂")
        if data_btn is not None:
            attach_dir_picker(self.data_file_w, data_btn)

        self.scans_w = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
        self.scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"

        common_container = Container(
            layout="vertical",
            widgets=[
                self.data_file_w,
                self.scans_w,
            ],
        )
        self.common_group = make_group("Data", common_container.native)

        # ISR group
        self.spec_file_w = FileEdit(mode="r", label="SPEC file")
        _ = set_file_button_symbol(self.spec_file_w, "📂")
        self.only_hkl_w = CheckBox(label="Only HKL scans")

        isr_container = Container(
            layout="vertical",
            widgets=[
                self.spec_file_w,
                self.only_hkl_w,
            ],
        )
        self.isr_group = make_group("ISR metadata", isr_container.native)

        # CMS group
        self.angle_step_w = FloatSpinBox(
            label="Angle step (°)", min=0.0, max=360.0, step=0.01
        )
        cms_container = Container(
            layout="vertical",
            widgets=[
                self.angle_step_w,
            ],
        )
        self.cms_group = make_group("CMS metadata", cms_container.native)

        # Experiment setup
        self.distance_w = FloatSpinBox(
            label="Distance (m)", min=-1e9, max=1e9, step=1e-6
        )
        self.pitch_w = FloatSpinBox(
            label="Pitch (m)", min=-1e9, max=1e9, step=1e-9
        )
        self.ypixels_w = SpinBox(
            label="Detector H (px)", min=0, max=10_000_000, step=1
        )
        self.xpixels_w = SpinBox(
            label="Detector W (px)", min=0, max=10_000_000, step=1
        )
        self.ycenter_w = SpinBox(
            label="BeamCenter H (px)", min=0, max=10_000_000, step=1
        )
        self.xcenter_w = SpinBox(
            label="BeamCenter W (px)", min=0, max=10_000_000, step=1
        )
        self.energy_w = FloatSpinBox(
            label="Energy (keV)", min=-1e9, max=1e9, step=1e-3
        )
        self.wavelength_w = FloatSpinBox(
            label="Wavelength (Å)", min=0, max=1e6, step=1e-3
        )
        self.wavelength_w.value = 0.0  # default empty

        setup_container = Container(
            layout="vertical",
            widgets=[
                self.distance_w,
                self.pitch_w,
                self.ypixels_w,
                self.xpixels_w,
                self.ycenter_w,
                self.xcenter_w,
                self.energy_w,
                self.wavelength_w,
            ],
        )
        self.setup_group = make_group(
            "Experimental Setup", setup_container.native
        )

        # Crop moved to DATA tab
        self.crop_enable_w = CheckBox(label="Crop before Build")
        self.y_min_w = SpinBox(label="H_t", min=0, max=10_000_000, step=1)
        self.y_max_w = SpinBox(label="H_b", min=0, max=10_000_000, step=1)
        self.x_min_w = SpinBox(label="W_l", min=0, max=10_000_000, step=1)
        self.x_max_w = SpinBox(label="W_r", min=0, max=10_000_000, step=1)

        # Arrange crop spinboxes in 2x2 grid
        crop_row1 = Container(
            layout="horizontal", widgets=[self.y_min_w, self.y_max_w]
        )
        crop_row2 = Container(
            layout="horizontal", widgets=[self.x_min_w, self.x_max_w]
        )
        crop_grid = Container(
            layout="vertical", widgets=[crop_row1, crop_row2]
        )

        # Tight spacing for crop grid
        for row in [crop_row1, crop_row2]:
            if hasattr(row, "native") and hasattr(row.native, "layout"):
                row.native.layout().setSpacing(4)
                row.native.layout().setContentsMargins(0, 0, 0, 0)
        if hasattr(crop_grid, "native") and hasattr(
            crop_grid.native, "layout"
        ):
            crop_grid.native.layout().setSpacing(2)
            crop_grid.native.layout().setContentsMargins(0, 0, 0, 0)

        self.btn_crop_from_roi = PushButton(text="🔲 Crop from ROI")

        crop_container = Container(
            layout="vertical",
            widgets=[
                self.crop_enable_w,
                crop_grid,
                self.btn_crop_from_roi,
            ],
        )
        self.crop_group = make_group("Crop", crop_container.native)

        self.btn_load = PushButton(text="📂 Load Data")
        self.btn_intensity = PushButton(text="📈 View Intensity")

        btn_row1 = QtWidgets.QWidget()
        row1 = QtWidgets.QHBoxLayout(btn_row1)
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(8)
        row1.addWidget(self.btn_load.native)
        row1.addWidget(self.btn_intensity.native)
        row1.addStretch(1)

        data_inner = QtWidgets.QWidget()
        data_lay = QtWidgets.QVBoxLayout(data_inner)
        data_lay.setContentsMargins(0, 0, 0, 0)
        data_lay.setSpacing(10)

        data_lay.addWidget(self.loader_type_w.native)
        data_lay.addWidget(self.common_group)
        data_lay.addWidget(self.isr_group)
        data_lay.addWidget(self.cms_group)
        data_lay.addWidget(self.setup_group)
        data_lay.addWidget(self.crop_group)
        data_lay.addWidget(btn_row1)
        data_lay.addStretch(1)

        tab_data = make_scroll(data_inner)

        # ---------------------------------------------------------------------
        # BUILD TAB (UB moved here + build + regrid)
        # ---------------------------------------------------------------------
        # UB section moved to Build tab
        self.ub_matrix_w = TextEdit()
        self.ub_matrix_w.value = "1 0 0\n0 1 0\n0 0 1"
        with contextlib.suppress(AttributeError):
            self.ub_matrix_w.native.setMinimumHeight(80)

        # 3x3 UB grid (visible), TextEdit kept as YAML-backed source
        def _make_cell() -> FloatSpinBox:
            w = FloatSpinBox(min=-1e9, max=1e9, step=1e-6)
            w.value = 0.0
            return w

        self._ub_cells: list[list[FloatSpinBox]] = [
            [_make_cell() for _ in range(3)] for _ in range(3)
        ]
        row_containers = [
            Container(layout="horizontal", widgets=r) for r in self._ub_cells
        ]
        self.ub_matrix_grid = Container(
            layout="vertical", widgets=row_containers
        )

        # Set tight spacing for UB grid
        for row_container in row_containers:
            if hasattr(row_container, "native") and hasattr(
                row_container.native, "layout"
            ):
                row_container.native.layout().setSpacing(2)
                row_container.native.layout().setContentsMargins(0, 0, 0, 0)
        if hasattr(self.ub_matrix_grid, "native") and hasattr(
            self.ub_matrix_grid.native, "layout"
        ):
            self.ub_matrix_grid.native.layout().setSpacing(2)
            self.ub_matrix_grid.native.layout().setContentsMargins(0, 0, 0, 0)

        # tighten UB cells
        for i in range(3):
            for j in range(3):
                c = self._ub_cells[i][j]
                native = getattr(c, "native", None)
                if isinstance(native, QtWidgets.QDoubleSpinBox):
                    with contextlib.suppress(AttributeError):
                        native.setButtonSymbols(
                            QtWidgets.QAbstractSpinBox.NoButtons
                        )
                    with contextlib.suppress(AttributeError):
                        native.setDecimals(6)
                    with contextlib.suppress(AttributeError):
                        native.setFixedWidth(60)
                    with contextlib.suppress(AttributeError):
                        native.setMaximumHeight(24)
                    with contextlib.suppress(AttributeError):
                        native.setAlignment(
                            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
                        )
                    # Remove margin on spinbox parent container
                    with contextlib.suppress(AttributeError):
                        parent = native.parent()
                        if (
                            parent
                            and hasattr(parent, "layout")
                            and parent.layout()
                        ):
                            parent.layout().setContentsMargins(0, 0, 0, 0)
                            parent.layout().setSpacing(0)

        # hide text editor visually (still used for YAML binding)
        with contextlib.suppress(AttributeError):
            self.ub_matrix_w.native.setVisible(False)

        def _ub_grid_to_text() -> str:
            rows: list[str] = []
            for r in self._ub_cells:
                vals = []
                for c in r:
                    vals.append(f"{float(c.value):.6g}")
                rows.append(" ".join(vals))
            return "\n".join(rows)

        def _populate_ub_grid_from_text(txt: str) -> None:
            try:
                arr = parse_ub_matrix(txt)
            except ValueError:
                arr = None
            if arr is None or arr.shape != (3, 3):
                arr = np.eye(3, dtype=float)
            for i in range(3):
                for j in range(3):
                    self._ub_cells[i][j].value = float(arr[i, j])

        self._populate_ub_grid_from_text = _populate_ub_grid_from_text
        self._ub_grid_to_text = _ub_grid_to_text

        # connect grid -> hidden text sync
        for r in self._ub_cells:
            for c in r:
                c.changed.connect(
                    lambda *_: setattr(
                        self.ub_matrix_w, "value", self._ub_grid_to_text()
                    )
                )

        # Align grid to the left with a horizontal stretch
        grid_with_stretch = Container(
            layout="horizontal",
            widgets=[self.ub_matrix_grid],
        )
        if hasattr(grid_with_stretch, "native") and hasattr(
            grid_with_stretch.native, "layout"
        ):
            grid_with_stretch.native.layout().addStretch()
            grid_with_stretch.native.layout().setContentsMargins(0, 0, 0, 0)
            grid_with_stretch.native.layout().setSpacing(0)

        # Create UB includes 2π checkbox for crystal container
        self.ub_2pi_w = CheckBox(label="UB includes 2π")

        crystal_container = Container(
            layout="vertical",
            widgets=[grid_with_stretch, self.ub_matrix_w, self.ub_2pi_w],
        )
        # Reduce spacing inside crystal container
        if hasattr(crystal_container, "native") and hasattr(
            crystal_container.native, "layout"
        ):
            crystal_container.native.layout().setSpacing(2)
            crystal_container.native.layout().setContentsMargins(0, 0, 0, 0)

        crystal_group = make_group("Crystal", crystal_container.native)

        # Build settings
        self.sample_axes_w = LineEdit(label="Sample axes")
        self.sample_axes_w.value = "x+, y+, z-"
        self.detector_axes_w = LineEdit(label="Detector axes")
        self.detector_axes_w.value = "x+"
        self.center_one_based_w = CheckBox(label="1-based center")

        # Regrid settings
        self.space_w = ComboBox(label="Space", choices=["hkl", "q"])
        self.grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
        self.grid_shape_w.value = "200,*,*"
        self.fuzzy_w = CheckBox(label="Fuzzy gridder")
        self.fuzzy_width_w = FloatSpinBox(
            label="Width (fuzzy)", min=0.0, max=1e9, step=0.01
        )
        self.normalize_w = ComboBox(label="Normalize", choices=["mean", "sum"])
        self.normalize_w.value = "mean"

        self.btn_build = PushButton(text="🔧 Build RSM Map")
        self.btn_regrid = PushButton(text="🧮 Regrid")

        build_container = Container(
            layout="vertical",
            widgets=[
                Label(value="<b>RSM Builder</b>"),
                self.center_one_based_w,
                self.sample_axes_w,
                self.detector_axes_w,
                hsep(),
                Label(value="<b>Grid Settings</b>"),
                self.space_w,
                self.grid_shape_w,
                self.fuzzy_w,
                self.fuzzy_width_w,
                self.normalize_w,
            ],
        )
        build_group = make_group("Build / Regrid", build_container.native)

        btn_row2 = QtWidgets.QWidget()
        row2 = QtWidgets.QHBoxLayout(btn_row2)
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(8)
        row2.addWidget(self.btn_build.native)
        row2.addWidget(self.btn_regrid.native)
        row2.addStretch(1)

        build_inner = QtWidgets.QWidget()
        build_lay = QtWidgets.QVBoxLayout(build_inner)
        build_lay.setContentsMargins(0, 0, 0, 0)
        build_lay.setSpacing(10)
        build_lay.addWidget(crystal_group)
        build_lay.addWidget(build_group)
        build_lay.addWidget(btn_row2)
        build_lay.addStretch(1)

        tab_build = make_scroll(build_inner)

        # ---------------------------------------------------------------------
        # VIEW TAB (stop + refresh + adjusted Run All)
        # ---------------------------------------------------------------------
        self.log_view_w = CheckBox(label="Log view")
        self.cmap_w = ComboBox(
            label="Colormap",
            choices=["viridis", "inferno", "magma", "plasma", "cividis"],
        )
        self.cmap_w.value = "inferno"
        self.rendering_w = ComboBox(
            label="Rendering", choices=["attenuated_mip", "mip", "translucent"]
        )
        self.rendering_w.value = "attenuated_mip"
        self.contrast_lo_w = FloatSpinBox(
            label="Contrast low (%)", min=0.0, max=100.0, step=0.1
        )
        self.contrast_hi_w = FloatSpinBox(
            label="Contrast high (%)", min=0.0, max=100.0, step=0.1
        )
        self.contrast_lo_w.value = 1.0
        self.contrast_hi_w.value = 99.8

        self.status_w = TextEdit(value="")
        with contextlib.suppress(AttributeError):
            self.status_w.native.setReadOnly(True)
            self.status_w.native.setMinimumHeight(220)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.export_vtr_w = FileEdit(mode="w", label="Output VTK (.vtr)")
        _ = set_file_button_symbol(self.export_vtr_w, "📂")

        self.btn_view = PushButton(text="🔭 View RSM")
        self.btn_export = PushButton(text="💾 Export to VTK")

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_refresh = QtWidgets.QPushButton("Refresh")

        self.btn_run_all_native = QtWidgets.QPushButton("Run All")
        self.btn_run_all_native.setMinimumHeight(
            34
        )  # smaller to fit row better
        self.btn_run_all_native.setMinimumWidth(110)
        if self._icon is not None:
            with contextlib.suppress(AttributeError):
                self.btn_run_all_native.setIcon(self._icon)

        view_controls = Container(
            layout="vertical",
            widgets=[
                Label(value="<b>Napari Viewer</b>"),
                self.log_view_w,
                self.cmap_w,
                self.rendering_w,
                self.contrast_lo_w,
                self.contrast_hi_w,
            ],
        )
        view_group = make_group("View", view_controls.native)

        export_row = QtWidgets.QWidget()
        export_lay = QtWidgets.QHBoxLayout(export_row)
        export_lay.setContentsMargins(0, 0, 0, 0)
        export_lay.setSpacing(8)
        export_lay.addWidget(self.export_vtr_w.native)
        export_lay.addWidget(self.btn_export.native)

        action_row = QtWidgets.QWidget()
        action_lay = QtWidgets.QHBoxLayout(action_row)
        action_lay.setContentsMargins(0, 0, 0, 0)
        action_lay.setSpacing(8)
        action_lay.addWidget(self.btn_view.native)
        action_lay.addWidget(self.btn_refresh)
        action_lay.addWidget(self.btn_stop)
        action_lay.addStretch(1)
        action_lay.addWidget(self.btn_run_all_native)

        view_inner = QtWidgets.QWidget()
        view_lay = QtWidgets.QVBoxLayout(view_inner)
        view_lay.setContentsMargins(0, 0, 0, 0)
        view_lay.setSpacing(10)
        view_lay.addWidget(view_group)
        view_lay.addWidget(make_group("Status / Output", self.status_w.native))
        view_lay.addWidget(self.progress)
        view_lay.addWidget(make_group("Export", export_row))
        view_lay.addWidget(action_row)
        view_lay.addStretch(1)

        tab_view = make_scroll(view_inner)

        # Tabs
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(tab_data, "Data")
        tabs.addTab(tab_build, "Build")
        tabs.addTab(tab_view, "View")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        outer.addWidget(tabs)

        self.status_bar = QtWidgets.QStatusBar()
        outer.addWidget(self.status_bar)

        # ---------------------------------------------------------------------
        # YAML <-> UI binding (NOTE: map includes crop in Data tab and UB in Build tab)
        # ---------------------------------------------------------------------
        self._widget_map: dict[str, dict[str, Any]] = {
            "data": {
                "loader_type": self.loader_type_w,
                "data_file": self.data_file_w,
                "scans": self.scans_w,
                "spec_file": self.spec_file_w,
                "only_hkl": self.only_hkl_w,
                "cms_angle_step": self.angle_step_w,
            },
            "ExperimentSetup": {
                "distance": self.distance_w,
                "pitch": self.pitch_w,
                "ycenter": self.ycenter_w,
                "xcenter": self.xcenter_w,
                "xpixels": self.xpixels_w,
                "ypixels": self.ypixels_w,
                "energy": self.energy_w,
                "wavelength": self.wavelength_w,
            },
            "Crystal": {"ub": self.ub_matrix_w},
            "build": {
                "ub_includes_2pi": self.ub_2pi_w,
                "center_is_one_based": self.center_one_based_w,
                "sample_axes": self.sample_axes_w,
                "detector_axes": self.detector_axes_w,
            },
            "crop": {
                "enable": self.crop_enable_w,
                "y_min": self.y_min_w,
                "y_max": self.y_max_w,
                "x_min": self.x_min_w,
                "x_max": self.x_max_w,
            },
            "regrid": {
                "space": self.space_w,
                "grid_shape": self.grid_shape_w,
                "fuzzy": self.fuzzy_w,
                "fuzzy_width": self.fuzzy_width_w,
                "normalize": self.normalize_w,
            },
            "view": {
                "log_view": self.log_view_w,
                "cmap": self.cmap_w,
                "rendering": self.rendering_w,
                "contrast_lo": self.contrast_lo_w,
                "contrast_hi": self.contrast_hi_w,
            },
            "export": {"vtr_path": self.export_vtr_w},
        }

        # Apply active profile from YAML to UI on startup
        # This ensures UI always reflects the saved configuration
        self._apply_yaml_to_widgets()
        self._connect_widget_changes()

        # Signals
        self.loader_type_w.changed.connect(
            lambda *_: self._on_loader_profile_changed()
        )
        self.btn_load.clicked.connect(self.on_load)
        self.btn_intensity.clicked.connect(self.on_view_intensity)
        self.btn_crop_from_roi.clicked.connect(self.on_crop_from_roi)
        self.btn_build.clicked.connect(self.on_build)
        self.btn_regrid.clicked.connect(self.on_regrid)
        self.btn_view.clicked.connect(self.on_view)
        self.btn_export.clicked.connect(self.on_export_vtk)
        self.btn_run_all_native.clicked.connect(self.on_run_all)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_refresh.clicked.connect(self.on_refresh)

        # Note: Crop geometry updates to detector size and beam center are now only
        # applied when the user clicks "Crop from ROI" button, not automatically
        # when crop widgets change. This gives users control over when to apply.

        # Energy <-> Wavelength synchronization (λ[Å] = 12.398 / E[keV])
        self._energy_wavelength_syncing = False

        def _on_energy_changed(*_):
            if self._energy_wavelength_syncing:
                return
            energy = float(self.energy_w.value or 0)
            if energy > 0:
                self._energy_wavelength_syncing = True
                self.wavelength_w.value = 12.398419843320026 / energy
                self._energy_wavelength_syncing = False

        def _on_wavelength_changed(*_):
            if self._energy_wavelength_syncing:
                return
            wavelength = float(self.wavelength_w.value or 0)
            if wavelength > 0:
                self._energy_wavelength_syncing = True
                self.energy_w.value = 12.398419843320026 / wavelength
                self._energy_wavelength_syncing = False

        self.energy_w.changed.connect(_on_energy_changed)
        self.wavelength_w.changed.connect(_on_wavelength_changed)

        # Initial visibility
        self._update_loader_visibility()
        if not as_path_str(self.data_file_w.value).strip():
            self.data_file_w.value = os.path.expanduser("~")

    # -------------------------------------------------------------------------
    # Profile schema + switching
    # -------------------------------------------------------------------------
    def _migrate_yaml_schema_if_needed(self) -> None:
        """If YAML is old flat schema, wrap into profiles['ISR'] and create profiles['CMS']."""
        profiles = self._ydoc.get("profiles")
        if isinstance(profiles, dict) and profiles:
            self._ydoc.setdefault("active_profile", "ISR")
            return

        # Treat entire current doc as ISR profile if it looks like the old schema
        old = dict(self._ydoc) if isinstance(self._ydoc, dict) else {}
        active_guess = str(
            old.get("data", {}).get("loader_type") or "ISR"
        ).upper()
        if active_guess not in ("ISR", "CMS"):
            active_guess = "ISR"

        self._ydoc = {
            "active_profile": active_guess,
            "profiles": {
                "ISR": old if old else _default_profile_doc("ISR"),
                "CMS": _default_profile_doc("CMS"),
            },
        }
        save_yaml(self._yaml_path, self._ydoc)

    def _current_profile_name(self) -> str:
        name = str(self._ydoc.get("active_profile") or "ISR").upper()
        return name if name in ("ISR", "CMS") else "ISR"

    def _profile_doc(self, name: str | None = None) -> dict[str, Any]:
        pname = (name or self._current_profile_name()).upper()
        profiles = self._ydoc.setdefault("profiles", {})
        if not isinstance(profiles, dict):
            profiles = {}
            self._ydoc["profiles"] = profiles
        if pname not in profiles or not isinstance(profiles.get(pname), dict):
            profiles[pname] = _default_profile_doc(pname)
        return profiles[pname]

    def _save_widgets_into_profile(self, profile_name: str) -> None:
        """Save all current UI widget values into specified profile.

        PROFILE ISOLATION GUARANTEE:
        - Called AFTER successful data load (_on_load_done)
        - Saves current UI state to the active profile in YAML
        - Prevents any widget changes from ISR profile overwriting CMS profile and vice versa
        - Each profile is only updated after data is successfully processed with that profile
        - Ensures loader_type field always matches the profile name
        """
        prof = self._profile_doc(profile_name)
        for section, mapping in self._widget_map.items():
            prof.setdefault(section, {})
            for key, widget in mapping.items():
                prof[section][key] = self._widget_value_for_yaml(widget)

        # Ensure the profile's loader_type matches its name to prevent corruption
        prof.setdefault("data", {})
        prof["data"]["loader_type"] = profile_name

        save_yaml(self._yaml_path, self._ydoc)

    def _on_loader_profile_changed(self) -> None:
        """Profile switching handler: Load new profile from YAML without modifying saved data.

        PROFILE ISOLATION ALGORITHM (Updated):
        ======================================
        When user changes Loader dropdown (ISR <-> CMS), this sequence ensures
        profiles are loaded FROM YAML without clearing or modifying saved data:

        1. DISABLE SAVES: Set _switching_profile=True to block widget change handlers
                         (prevents intermediate signals from touching YAML)
        2. SWITCH: Change active_profile in memory and disk (only updates active flag)
        3. RELOAD: Reload entire YAML document from disk to ensure fresh data
        4. LOAD NEW: Apply new profile from YAML to all UI widgets
                     (uses saved configuration, discards any unsaved UI changes)
        5. ENABLE SAVES: Set _switching_profile=False
        6. RESET AUTO-SAVE: Disable auto-save until data is loaded

        This ensures:
        - ISR profile data in YAML is never modified when switching to CMS
        - CMS profile data in YAML is never modified when switching to ISR
        - Each profile preserves its saved configuration in YAML
        - UI always loads from the saved YAML configuration
        - No data is cleared from YAML - only the active_profile flag changes
        - Manual UI changes are NOT saved until data is successfully loaded
        - Loaders always read ExperimentSetup from their designated profile
        """
        new_name = str(self.loader_type_w.value or "ISR").upper()
        if new_name not in ("ISR", "CMS"):
            new_name = "ISR"

        old_name = self._current_profile_name()
        if new_name == old_name:
            self._update_loader_visibility()
            return

        # Guard against recursion - if we're already switching, ignore
        if self._switching_profile:
            return

        # Prevent widget change signals from saving during switch
        self._switching_profile = True
        try:
            # Switch active profile in memory and disk
            self._ydoc["active_profile"] = new_name
            save_yaml(self._yaml_path, self._ydoc)

            # IMPORTANT: Reload YAML from disk to ensure we have fresh profile data
            # This prevents any in-memory state from overwriting saved configurations
            try:
                self._ydoc = load_yaml(self._yaml_path)
            except Exception as e:
                logger.exception(
                    "Failed to reload YAML after profile switch: %s", e
                )
                show_error(
                    f"Failed to reload configuration file. Please check {self._yaml_path} for errors."
                )
                # Try to recover by reloading from disk without the changes
                try:
                    self._ydoc = load_yaml(self._yaml_path)
                except (OSError, yaml.YAMLError):
                    # If still fails, use default profiles
                    self._ydoc = {
                        "active_profile": new_name,
                        "profiles": {
                            "ISR": _default_profile_doc("ISR"),
                            "CMS": _default_profile_doc("CMS"),
                        },
                    }
                return

            # Load new profile FROM YAML -> UI (uses freshly loaded YAML data)
            self._apply_yaml_to_widgets()
            self._update_loader_visibility()

            # Disable auto-save until data is loaded with this profile
            self._auto_save_enabled = False

            # Log profile switch for transparency
            logger.info(
                "Switched loader profile: %s → %s (loaded from YAML)",
                old_name,
                new_name,
            )
            self.status(f"Loaded {new_name} profile from saved configuration")
        finally:
            self._switching_profile = False

    # -------------------------------------------------------------------------
    # Loader-specific visibility
    # -------------------------------------------------------------------------
    def _update_loader_visibility(self) -> None:
        choice = str(self.loader_type_w.value or "ISR").upper()
        is_cms = choice.startswith("CMS")
        self.isr_group.setVisible(not is_cms)
        self.cms_group.setVisible(is_cms)

    # -------------------------------------------------------------------------
    # YAML binding helpers (profile-aware)
    # -------------------------------------------------------------------------
    def _apply_yaml_to_widgets(self) -> None:
        """Load profile data into UI widgets with full protection against saves.

        ISOLATION PROTECTION:
        - Sets _profile_loading=True during entire load process
        - All widget value changes silently update UI (no YAML saves)
        - Geometry operations (UB grid, crop) also protected
        - Only after full load completes can manual changes trigger YAML saves

        IMPORTANT: This method ONLY reads from YAML and updates UI.
        It never writes to YAML - the _profile_loading flag blocks all saves.
        Each profile's data in YAML is preserved and loaded as-is.
        """
        prof = self._profile_doc()

        def set_widget(widget: Any, value: Any) -> None:
            if value is None:
                return
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                if isinstance(widget, FloatSpinBox):
                    widget.value = float(value)
                elif isinstance(widget, SpinBox):
                    widget.value = int(value)
                elif isinstance(widget, CheckBox):
                    widget.value = bool(value)
                elif isinstance(widget, ComboBox):
                    sval = str(value)
                    if sval in widget.choices:
                        widget.value = sval
                elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
                    widget.value = str(value)

        self._profile_loading = True
        try:
            for section, mapping in self._widget_map.items():
                vals = prof.get(section, {}) or {}
                for key, widget in mapping.items():
                    set_widget(widget, vals.get(key))

            # sync UB grid after all widgets loaded
            with contextlib.suppress(Exception):
                self._populate_ub_grid_from_text(self.ub_matrix_w.value)

            # Note: Crop geometry is NOT automatically applied when loading profiles.
            # User must click "Crop from ROI" to update detector size and beam center.
        finally:
            self._profile_loading = False

        # enforce dropdown = active profile (after loading completes)
        # Only update if value differs to prevent triggering unnecessary change signals
        with contextlib.suppress(Exception):
            expected_profile = self._current_profile_name()
            if str(self.loader_type_w.value) != expected_profile:
                self.loader_type_w.value = expected_profile

    def _widget_value_for_yaml(self, widget: Any) -> Any:
        if isinstance(widget, FloatSpinBox):
            return float(widget.value)
        if isinstance(widget, SpinBox):
            return int(widget.value)
        if isinstance(widget, CheckBox):
            return bool(widget.value)
        if isinstance(widget, ComboBox):
            return str(widget.value)
        if isinstance(widget, (LineEdit, TextEdit, FileEdit)):
            txt = str(widget.value).strip()
            return txt if txt else None
        return widget.value

    def _connect_widget_changes(self) -> None:
        """Connect all widgets to save handler with triple-level protection.

        TRIPLE-LEVEL SAVE PROTECTION:
        ==============================
        1. _profile_loading: Blocks saves during _apply_yaml_to_widgets()
           (prevents YAML corruption when loading from file)

        2. _switching_profile: Blocks saves during _on_loader_profile_changed()
           (prevents intermediate changes during profile switch)

        3. _auto_save_enabled: Only allows saves after successful data load
           (prevents manual UI changes from saving until data is processed)

        RESULT:
        - Profile switch: Loads from YAML, does NOT save old UI state
        - Manual user changes: Does NOT auto-save until data is loaded
        - After data load: Saves current UI state, enables future auto-saves
        - Programmatic UI changes (during load/switch): NO saves

        This ensures ISR/CMS profiles remain isolated and are only updated
        after actual data processing, not just from manual edits.
        """

        def on_changed(section: str, key: str, widget: Any) -> None:
            # Prevent saves while loading, switching, or before data is loaded
            if (
                self._profile_loading
                or self._switching_profile
                or not self._auto_save_enabled
            ):
                return
            prof = self._profile_doc()
            prof.setdefault(section, {})
            prof[section][key] = self._widget_value_for_yaml(widget)
            save_yaml(self._yaml_path, self._ydoc)

        for section, mapping in self._widget_map.items():
            for key, widget in mapping.items():
                widget.changed.connect(
                    lambda *_, s=section, k=key, w=widget: on_changed(s, k, w)
                )

    # -------------------------------------------------------------------------
    # Crop geometry application (beamcenter + detector size)
    # -----------------------------------------------------------------------------
    def _crop_bounds_valid(self) -> bool:
        if not bool(self.crop_enable_w.value):
            return False
        ymin, ymax = int(self.y_min_w.value), int(self.y_max_w.value)
        xmin, xmax = int(self.x_min_w.value), int(self.x_max_w.value)
        if ymin < 0 or xmin < 0:
            return False
        return not (ymax <= ymin or xmax <= xmin)

    def _apply_crop_geometry_to_setup(self) -> None:
        """
        DEPRECATED: This method is no longer used.

        Cropping is now applied directly in on_crop_from_roi() which:
        - Updates DataFrame intensity with cropped frames
        - Updates setup object with new detector size and beam center
        - Updates intensity viewer display

        This ensures the data, setup, and UI are all synchronized after cropping.
        """
        if not self._crop_bounds_valid():
            return

        ymin, ymax = int(self.y_min_w.value), int(self.y_max_w.value)
        xmin, xmax = int(self.x_min_w.value), int(self.x_max_w.value)

        # Use "current" geometry as the pre-crop geometry.
        # If the user toggles crop repeatedly, they may want to reset manually;
        # this function only applies crop when enabled.
        try:
            old_ypx = int(self.ypixels_w.value)
            old_xpx = int(self.xpixels_w.value)
            old_yc = int(self.ycenter_w.value)
            old_xc = int(self.xcenter_w.value)
        except (TypeError, ValueError):
            return

        # bounds should be within detector; if not, warn but still clamp
        if old_ypx > 0:
            ymax = min(ymax, old_ypx)
        if old_xpx > 0:
            xmax = min(xmax, old_xpx)

        if ymax <= ymin or xmax <= xmin:
            show_warning(
                "Crop bounds out of range for current detector geometry."
            )
            return

        new_ypx = ymax - ymin
        new_xpx = xmax - xmin
        new_yc = old_yc - ymin
        new_xc = old_xc - xmin

        # keep beamcenter in bounds (warn if goes out)
        if not (0 <= new_yc < new_ypx) or not (0 <= new_xc < new_xpx):
            show_warning(
                "Crop moved beamcenter outside cropped detector. Check crop bounds/beamcenter."
            )

        # Apply changes (these will persist via widget.changed -> profile save)
        self.ypixels_w.value = int(new_ypx)
        self.xpixels_w.value = int(new_xpx)
        self.ycenter_w.value = int(max(0, new_yc))
        self.xcenter_w.value = int(max(0, new_xc))

    # -------------------------------------------------------------------------
    # Status/progress/busy helpers
    # -------------------------------------------------------------------------
    def status(self, msg: str) -> None:
        with contextlib.suppress(AttributeError):
            self.status_w.native.append(msg)
        self.status_bar.showMessage(msg, 3000)
        logger.info("ResView: %s", msg)

    def set_progress(self, value: int | None, *, busy: bool = False) -> None:
        if busy:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(value or 0))

    def set_busy(self, b: bool) -> None:
        for btn in (
            self.btn_load,
            self.btn_intensity,
            self.btn_build,
            self.btn_regrid,
            self.btn_view,
            self.btn_export,
        ):
            with contextlib.suppress(AttributeError):
                btn.native.setEnabled(not b)
        self.btn_run_all_native.setEnabled(not b)
        self.btn_stop.setEnabled(True)

    # -------------------------------------------------------------------------
    # Background load worker (async)
    # -------------------------------------------------------------------------
    @thread_worker
    def _load_worker(
        self,
        loader_name: str,
        spec: str,
        dpath: str,
        scans: list[int],
        only_hkl: bool,
        profile_name: str,
    ) -> dict[str, Any]:
        # Ensure loaders read ExperimentSetup from the correct profile section
        # by explicitly using a profile-aware YAML path or section
        yaml_file = yaml_path()

        if loader_name.startswith("CMS"):
            loader = RSMDataloader_CMS(
                yaml_file,
                dpath,
                selected_scans=scans,
                crop_window=None,
                angle_step=float(self.angle_step_w.value),
            )
        else:
            loader = RSMDataLoader_ISR(
                spec,
                yaml_file,
                dpath,
                selected_scans=scans,
                process_hklscan_only=bool(only_hkl),
            )

        load_result = loader.load()

        setup = None
        ub = None
        df = None
        frames = None

        if isinstance(load_result, tuple):
            if len(load_result) >= 1:
                setup = load_result[0]
            if len(load_result) >= 2:
                ub = load_result[1]
            if len(load_result) >= 3:
                df = load_result[2]

        # Fallback to loader attributes if tuple unpacking didn't work
        if setup is None:
            setup = getattr(loader, "setup", None)
        if ub is None:
            ub = getattr(loader, "ub", None)
        if df is None:
            df = getattr(loader, "df", None)

        if df is not None and hasattr(df, "intensity"):
            with contextlib.suppress(TypeError):
                frames = list(df.intensity)

        return {
            "loader": loader,
            "setup": setup,
            "ub": ub,
            "df": df,
            "frames": frames,
        }

    def _on_load_done(self, result: dict[str, Any]) -> None:
        try:
            self._state["loader"] = result.get("loader")
            self._state["setup"] = result.get("setup")
            self._state["ub"] = result.get("ub")
            self._state["df"] = result.get("df")
            self._state["intensity_frames"] = result.get("frames")

            # Update detector size from intensity frame dimensions (actual data)
            # This ensures loaded data dimensions override YAML values
            frames = self._state["intensity_frames"]
            if frames is not None:
                # frames can be a list of 2D arrays or a 3D array
                if isinstance(frames, (list, tuple)) and len(frames) > 0:
                    first_frame = frames[0]
                    if (
                        hasattr(first_frame, "shape")
                        and len(first_frame.shape) >= 2
                    ):
                        H, W = first_frame.shape[-2], first_frame.shape[-1]
                        self.ypixels_w.value = int(H)
                        self.xpixels_w.value = int(W)
                        logger.info(
                            "Updated detector size from loaded data: %sx%s",
                            H,
                            W,
                        )
                elif hasattr(frames, "shape") and len(frames.shape) >= 2:
                    # frames is a numpy array (T, H, W) or (H, W)
                    H, W = frames.shape[-2], frames.shape[-1]
                    self.ypixels_w.value = int(H)
                    self.xpixels_w.value = int(W)
                    logger.info(
                        "Updated detector size from loaded data: %sx%s", H, W
                    )

            # update UB in UI
            self.ub_matrix_w.value = format_ub_matrix(self._state["ub"])
            with contextlib.suppress(ValueError):
                self._populate_ub_grid_from_text(self.ub_matrix_w.value)

            # reset downstream
            self._state["builder"] = None
            self._state["grid"] = None
            self._state["edges"] = None

            # Save current UI state to profile YAML after successful data load
            # This is the ONLY time we update the profile from UI values
            current_profile = self._current_profile_name()
            self._save_widgets_into_profile(current_profile)
            logger.info(
                "Saved current UI state to %s profile after successful data load",
                current_profile,
            )

            # Enable auto-save for future manual changes
            self._auto_save_enabled = True

            self.set_progress(25, busy=False)
            self.status("Data loaded and configuration saved.")
            show_info("Data loaded.")
        finally:
            self.set_busy(False)

        if self._run_all_pending:
            self.status("Run All: starting Build…")
            QtCore.QTimer.singleShot(0, self._run_all_build)

    def _on_worker_error(self, e: BaseException) -> None:
        self.set_progress(0, busy=False)
        self.set_busy(False)
        self.status(f"Error: {e}")
        show_error(str(e))
        if self._run_all_pending:
            self._run_all_pending = False
            self.btn_run_all_native.setEnabled(True)

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------
    def on_stop(self) -> None:
        """Stop in-flight load + stop Run All chaining."""
        self._run_all_pending = False

        worker = self._load_worker_instance
        if worker is not None:
            # napari worker supports quit(); also try .cancel() if present
            with contextlib.suppress(AttributeError):
                worker.quit()
            with contextlib.suppress(AttributeError):
                worker.cancel()

        self.btn_run_all_native.setEnabled(True)
        self.set_progress(0, busy=False)
        self.status("Stopped.")

    def on_refresh(self) -> None:
        """Refresh the RSM viewer with current view settings (if grid exists)."""
        self.on_view()

    def on_load(self) -> None:
        loader_name = str(
            self.loader_type_w.value or self._current_profile_name() or "ISR"
        ).upper()
        is_cms = loader_name.startswith("CMS")

        dpath = as_path_str(self.data_file_w.value).strip()
        try:
            scans = parse_scan_list((self.scans_w.value or "").strip())
        except ValueError as e:
            show_error(str(e))
            return

        if not os.path.isdir(dpath):
            show_error("Select a valid DATA folder.")
            return
        if not scans:
            show_error("Enter at least one scan (e.g. '17, 18-22').")
            return

        spec = ""
        if not is_cms:
            spec = as_path_str(self.spec_file_w.value).strip()
            if not spec or not os.path.isfile(spec):
                show_error("Select a valid SPEC file (required for ISR).")
                return

        self.set_busy(True)
        self.set_progress(None, busy=True)
        self.status(f"Loading ({'CMS' if is_cms else 'ISR'}) scans {scans}…")

        # Pass current profile name to ensure loader reads from correct YAML section
        current_profile = self._current_profile_name()
        worker = self._load_worker(
            loader_name,
            spec,
            dpath,
            scans,
            bool(self.only_hkl_w.value),
            current_profile,
        )
        self._load_worker_instance = worker

        worker.returned.connect(self._on_load_done)
        worker.errored.connect(self._on_worker_error)
        worker.start()

    def on_view_intensity(self) -> None:
        frames = self._state.get("intensity_frames")
        if frames is None:
            show_error("No cached intensity frames. Load data first.")
            return
        try:
            intensity_viewer = IntensityNapariViewer(
                frames,
                name="Intensity",
                log_view=True,
                contrast_percentiles=(1.0, 99.8),
                cmap=str(self.cmap_w.value or "inferno"),
                rendering=str(self.rendering_w.value or "attenuated_mip"),
                add_timeseries=True,
                add_volume=True,
                scale_tzyx=(1.0, 1.0, 1.0),
                pad_value=np.nan,
            )
            intensity_viewer.launch(viewer=self.viewer)
            self._state["intensity_viewer"] = intensity_viewer

            # Connect ROI change events to auto-update crop widgets
            roi_layer = getattr(intensity_viewer, "_roi_layer", None)
            if roi_layer is not None:
                # Connect to all available layer events
                def _safe_connect(event_name):
                    try:
                        event = getattr(roi_layer.events, event_name, None)
                        if event is not None:
                            event.connect(self._on_roi_changed)
                            logger.debug(
                                "Connected to roi_layer.events.%s", event_name
                            )
                    except (AttributeError, TypeError) as e:
                        logger.debug(
                            "Could not connect to %s: %s", event_name, e
                        )

                # Try connecting to multiple event types
                for event_name in [
                    "data",
                    "set_data",
                    "transform",
                    "properties",
                ]:
                    _safe_connect(event_name)

                # Add periodic check using viewer's paint event
                # This ensures we update after any interaction
                original_paint = getattr(self.viewer, "_qt_window", None)
                if original_paint is not None:
                    # Create a simple periodic check
                    from qtpy.QtCore import QTimer

                    check_timer = QTimer()
                    check_timer.setInterval(500)  # Check every 500ms
                    check_timer.timeout.connect(lambda: self._on_roi_changed())
                    check_timer.start()
                    self._roi_check_timer = check_timer  # Keep reference

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception("Failed to open intensity viewer")
            show_error(f"Failed to open intensity viewer: {e}")

    def _on_roi_changed(self, _event=None) -> None:
        """Update crop widgets when ROI is dragged/resized."""
        intensity_viewer = self._state.get("intensity_viewer")
        if intensity_viewer is None:
            return
        roi_layer = getattr(intensity_viewer, "_roi_layer", None)
        if roi_layer is None or not roi_layer.data:
            return
        try:
            corners = np.round(roi_layer.data[0]).astype(int)
            y_coords = corners[:, 0]
            x_coords = corners[:, 1]
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            x_min, x_max = int(x_coords.min()), int(x_coords.max())

            # Only update if values have changed
            if y_max > y_min and x_max > x_min:
                changed = False
                if self.y_min_w.value != y_min:
                    self.y_min_w.value = y_min
                    changed = True
                if self.y_max_w.value != y_max:
                    self.y_max_w.value = y_max
                    changed = True
                if self.x_min_w.value != x_min:
                    self.x_min_w.value = x_min
                    changed = True
                if self.x_max_w.value != x_max:
                    self.x_max_w.value = x_max
                    changed = True

                if changed:
                    logger.debug(
                        "_on_roi_changed: Updated crop bounds to y=(%s, %s), x=(%s, %s)",
                        y_min,
                        y_max,
                        x_min,
                        x_max,
                    )
        except (AttributeError, IndexError, ValueError) as e:
            logger.debug("_on_roi_changed: Error %s", e)

    def on_crop_from_roi(self) -> None:
        """Copy ROI rectangle bounds to crop widgets and update detector size + beam center."""
        intensity_viewer = self._state.get("intensity_viewer")
        if intensity_viewer is None:
            show_error(
                "No intensity viewer open. Click 'View Intensity' first."
            )
            return

        # Access the ROI layer from the IntensityNapariViewer instance
        roi_layer = getattr(intensity_viewer, "_roi_layer", None)
        if roi_layer is None or not roi_layer.data:
            show_error("No ROI found in intensity viewer.")
            return

        try:
            # ROI corners: 4 x 2 array of (y, x) coordinates
            corners = np.round(roi_layer.data[0]).astype(int)
            y_coords = corners[:, 0]
            x_coords = corners[:, 1]

            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            x_min, x_max = int(x_coords.min()), int(x_coords.max())

            if y_max <= y_min or x_max <= x_min:
                show_error(
                    "Invalid ROI bounds. Make sure the ROI has positive area."
                )
                return

            # Update crop widgets
            self.y_min_w.value = y_min
            self.y_max_w.value = y_max
            self.x_min_w.value = x_min
            self.x_max_w.value = x_max
            self.crop_enable_w.value = True

            # Get current beam center
            old_yc = int(self.ycenter_w.value)
            old_xc = int(self.xcenter_w.value)

            # Update detector size to cropped size
            new_ypx = y_max - y_min
            new_xpx = x_max - x_min
            self.ypixels_w.value = int(new_ypx)
            self.xpixels_w.value = int(new_xpx)

            # Update beam center (shift by crop origin)
            new_yc = old_yc - y_min
            new_xc = old_xc - x_min

            # Warn if beam center is outside cropped region
            if not (0 <= new_yc < new_ypx) or not (0 <= new_xc < new_xpx):
                show_warning(
                    "Beam center is outside the cropped region. Check ROI position."
                )
                new_yc = max(0, min(new_yc, new_ypx - 1))
                new_xc = max(0, min(new_xc, new_xpx - 1))

            self.ycenter_w.value = int(new_yc)
            self.xcenter_w.value = int(new_xc)

            # Update the DataFrame's intensity data with cropped frames
            df = self._state.get("df")
            if df is not None and hasattr(df, "intensity"):
                try:
                    # Crop each intensity frame in the DataFrame
                    cropped_intensity = []
                    for frame in df["intensity"]:
                        if hasattr(frame, "shape") and len(frame.shape) >= 2:
                            cropped_frame = frame[y_min:y_max, x_min:x_max]
                            cropped_intensity.append(cropped_frame)
                        else:
                            cropped_intensity.append(frame)

                    # Update the DataFrame with cropped intensity
                    df["intensity"] = cropped_intensity
                    self._state["df"] = df

                    # Update intensity_frames in state for consistency
                    self._state["intensity_frames"] = cropped_intensity

                    logger.info(
                        "Cropped %d intensity frames to shape (%d, %d)",
                        len(cropped_intensity),
                        new_ypx,
                        new_xpx,
                    )
                except (
                    AttributeError,
                    TypeError,
                    KeyError,
                    IndexError,
                ) as crop_err:
                    logger.warning(
                        "Failed to crop DataFrame intensity: %s", crop_err
                    )

            # Update the setup object with new detector dimensions
            setup = self._state.get("setup")
            if setup is not None:
                try:
                    setup.ypixels = int(new_ypx)
                    setup.xpixels = int(new_xpx)
                    setup.ycenter = int(new_yc)
                    setup.xcenter = int(new_xc)
                    self._state["setup"] = setup
                    logger.info(
                        "Updated setup: detector=%dx%d, center=(%d, %d)",
                        new_xpx,
                        new_ypx,
                        new_xc,
                        new_yc,
                    )
                except (AttributeError, TypeError) as setup_err:
                    logger.warning(
                        "Failed to update setup object: %s", setup_err
                    )

            # Update the intensity viewer in place with cropped data and adjusted ROI
            # The viewer's apply_crop() method will:
            # - Update the displayed image to cropped data
            # - Reposition the ROI box to the new coordinate system
            # - Keep the same viewer window open (no need to reopen)
            intensity_viewer.apply_crop(y_min, y_max, x_min, x_max)

            self.status(
                f"Crop applied: y=[{y_min}, {y_max}), x=[{x_min}, {x_max}), detector={new_xpx}×{new_ypx}"
            )

        except Exception as e:
            logger.exception("Failed to extract ROI bounds")
            show_error(f"Failed to extract ROI bounds: {e}")

    def on_build(self) -> None:
        # Get loaded data from state (no longer need the loader)
        setup = self._state.get("setup")
        ub = self._state.get("ub")
        df = self._state.get("df")

        if setup is None or df is None:
            show_error("Load data first.")
            return

        self.set_busy(True)
        self.set_progress(None, busy=True)
        self.status("Computing Q/HKL/intensity…")

        # Ensure UB text reflects grid UI (in case user edited cells)
        with contextlib.suppress(AttributeError):
            self.ub_matrix_w.value = self._ub_grid_to_text()

        # Parse UB from UI (user may have edited it)
        ub_arr = None
        try:
            ub_arr = parse_ub_matrix(str(self.ub_matrix_w.value or ""))
        except ValueError as e:
            show_error(f"Invalid UB: {e}")
            self.set_busy(False)
            self.set_progress(0, busy=False)
            return

        # Use UI UB if available, otherwise use loaded UB
        if ub_arr is None:
            ub_arr = ub

        # Update state with current UB
        self._state["ub"] = ub_arr

        try:
            # Pass setup, UB, df directly to RSMBuilder (no loader re-execution)
            b = RSMBuilder(
                setup,
                ub_arr,
                df,
                sample_axes=parse_axes_list(self.sample_axes_w.value),
                detector_axes=parse_axes_list(self.detector_axes_w.value),
                ub_includes_2pi=bool(self.ub_2pi_w.value),
                center_is_one_based=bool(self.center_one_based_w.value),
            )
            b.compute_full(verbose=False)

            # Note: Cropping is now done directly on the DataFrame and setup
            # when the "Crop from ROI" button is clicked, so no need to crop here

            self._state["builder"] = b
            self._state["grid"] = None
            self._state["edges"] = None

            self.set_progress(50, busy=False)
            self.status("RSM map built.")
        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            show_error(f"Build error: {e}")
            self.set_progress(40, busy=False)
            self.status(f"Build failed: {e}")
        finally:
            self.set_busy(False)

    def on_regrid(self) -> None:
        b = self._state.get("builder")
        if b is None:
            show_error("Build the RSM map first.")
            return

        try:
            gx, gy, gz = parse_grid_shape(self.grid_shape_w.value)
        except ValueError as e:
            show_error(str(e))
            return
        if gx is None:
            show_error("Grid X (first value) is required (e.g., 200,*,*).")
            return

        self.set_busy(True)
        self.set_progress(None, busy=True)
        self.status(
            f"Regridding to {str(self.space_w.value).upper()} grid {(gx, gy, gz)}…"
        )

        try:
            kwargs: dict[str, Any] = {
                "space": self.space_w.value,
                "grid_shape": (gx, gy, gz),
                "fuzzy": bool(self.fuzzy_w.value),
                "normalize": self.normalize_w.value,
                "stream": True,
            }
            if (
                bool(self.fuzzy_w.value)
                and float(self.fuzzy_width_w.value or 0) > 0
            ):
                kwargs["width"] = float(self.fuzzy_width_w.value)

            grid, edges = b.regrid_xu(**kwargs)
            self._state["grid"], self._state["edges"] = grid, edges

            self.set_progress(75, busy=False)
            self.status("Regrid completed.")
        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            show_error(f"Regrid error: {e}")
            self.set_progress(60, busy=False)
            self.status(f"Regrid failed: {e}")
        finally:
            self.set_busy(False)

    def on_view(self) -> None:
        if self._state.get("grid") is None or self._state.get("edges") is None:
            show_error("Regrid first.")
            return

        try:
            lo = float(self.contrast_lo_w.value)
            hi = float(self.contrast_hi_w.value)
            if not (0 <= lo < hi <= 100):
                raise ValueError(
                    "Contrast % must satisfy 0 ≤ low < high ≤ 100"
                )
        except ValueError as e:
            show_error(str(e))
            return

        self.set_progress(None, busy=True)
        self.status("Opening RSM viewer…")

        try:
            viz = RSMNapariViewer(
                self._state["grid"],
                self._state["edges"],
                space=self.space_w.value,
                name="RSM3D",
                log_view=bool(self.log_view_w.value),
                contrast_percentiles=(lo, hi),
                cmap=self.cmap_w.value,
                rendering=self.rendering_w.value,
            )
            viewer_local = viz.launch(viewer=self.viewer)
            self._state["rsm_viewer"] = viewer_local
            self.set_progress(100, busy=False)
            self.status("RSM viewer opened.")
        except (RuntimeError, ValueError, TypeError) as e:
            show_error(f"View error: {e}")
            self.set_progress(80, busy=False)
            self.status(f"View failed: {e}")

    def on_export_vtk(self) -> None:
        if self._state.get("grid") is None or self._state.get("edges") is None:
            show_error("Regrid first, then export.")
            return

        out_path = as_path_str(self.export_vtr_w.value).strip()
        if not out_path:
            show_error("Choose an output .vtr file path.")
            return
        if not out_path.lower().endswith(".vtr"):
            out_path += ".vtr"

        self.set_busy(True)
        self.set_progress(None, busy=True)
        self.status(f"Exporting → {out_path}")

        try:
            write_rsm_volume_to_vtr(
                self._state["grid"],
                self._state["edges"],
                out_path,
                binary=False,
                compress=True,
            )
            self.set_progress(100, busy=False)
            self.status(f"Exported: {out_path}")
            show_info(f"Exported: {out_path}")
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            show_error(f"Export error: {e}")
            self.set_progress(0, busy=False)
            self.status(f"Export failed: {e}")
        finally:
            self.set_busy(False)

    # -------------------------------------------------------------------------
    # Run All pipeline (async-safe)
    # -------------------------------------------------------------------------
    def on_run_all(self) -> None:
        self.btn_run_all_native.setEnabled(False)
        self._run_all_pending = True
        self.set_progress(0, busy=False)
        self.status("Run All: starting Load…")
        self.on_load()

    def _run_all_build(self) -> None:
        try:
            self.on_build()
        except (RuntimeError, ValueError, TypeError) as e:
            show_error(f"Run All failed during Build: {e}")
            self._run_all_pending = False
            self.btn_run_all_native.setEnabled(True)
            return

        if self._state.get("builder") is None:
            self._run_all_pending = False
            self.btn_run_all_native.setEnabled(True)
            return

        self.status("Run All: starting Regrid…")
        QtCore.QTimer.singleShot(0, self._run_all_regrid)

    def _run_all_regrid(self) -> None:
        try:
            self.on_regrid()
        except (RuntimeError, ValueError, TypeError) as e:
            show_error(f"Run All failed during Regrid: {e}")
            self._run_all_pending = False
            self.btn_run_all_native.setEnabled(True)
            return

        if self._state.get("grid") is None or self._state.get("edges") is None:
            self._run_all_pending = False
            self.btn_run_all_native.setEnabled(True)
            return

        self.status("Run All: starting View…")
        QtCore.QTimer.singleShot(0, self._run_all_view)

    def _run_all_view(self) -> None:
        try:
            self.on_view()
            self.status("Run All completed.")
        except (RuntimeError, ValueError, TypeError) as e:
            show_error(f"Run All failed during View: {e}")
        finally:
            self._run_all_pending = False
            self.btn_run_all_native.setEnabled(True)


# -----------------------------------------------------------------------------
# npe2 dock widget provider
# -----------------------------------------------------------------------------
def napari_experimental_provide_dock_widget():
    return ResviewDockWidget
