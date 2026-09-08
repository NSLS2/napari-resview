#!/usr/bin/env python3
"""
spec_parser.py

Defines ExperimentSetup, Crystal, ScanAngles, and SpecParser classes to parse
SPEC-format files and return scan data as a pandas or Dask DataFrame.
"""

import warnings
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import yaml


class ExperimentSetup:
    """
    Load experiment parameters from a YAML file. Wavelength is optional:
      • if provided and >1e-3 Å, used directly
      • if provided in meters (<1e-3), converted to Å
      • if omitted or non‐positive, computed from energy [Å] = 12.398419843320026 / E[keV]

    Required keys (either top-level or inside `ExperimentSetup:`):
      distance, pitch, ycenter, xcenter, xpixels, ypixels, energy

    Optional key:
      wavelength
    """

    REQUIRED_KEYS = (
        "distance",
        "pitch",
        "ycenter",
        "xcenter",
        "xpixels",
        "ypixels",
        "energy",
    )

    def __init__(
        self,
        distance: float,
        pitch: float,
        ycenter: int,
        xcenter: int,
        xpixels: int,
        ypixels: int,
        energy: float,
        wavelength: float | None = None,
    ):
        self.distance = float(distance)
        self.pitch = float(pitch)
        self.ycenter = int(ycenter)
        self.xcenter = int(xcenter)
        self.xpixels = int(xpixels)
        self.ypixels = int(ypixels)
        self.energy = float(energy)
        self.energy_keV = float(energy)
        if self.distance <= 0:
            raise ValueError("ExperimentSetup: 'distance' must be > 0")
        if self.pitch <= 0:
            raise ValueError("ExperimentSetup: 'pitch' must be > 0")
        if self.xpixels <= 0 or self.ypixels <= 0:
            raise ValueError(
                "ExperimentSetup: 'xpixels' and 'ypixels' must be > 0"
            )
        if self.energy_keV <= 0:
            raise ValueError("ExperimentSetup: 'energy' (keV) must be > 0")

        lam_A: float | None = None
        if wavelength is not None:
            try:
                lam_A = float(wavelength)
            except (TypeError, ValueError):
                lam_A = None
        if lam_A is not None and 0.0 < lam_A < 1e-3:
            lam_A *= 1e10
        if lam_A is None or lam_A <= 0.0:
            lam_A = self._energy_keV_to_lambda_A(self.energy_keV)
        if lam_A <= 0.0:
            raise ValueError(
                "ExperimentSetup: computed wavelength is non-positive"
            )
        self.wavelength = lam_A

    @staticmethod
    def _energy_keV_to_lambda_A(E_keV: float) -> float:
        return 12.398419843320026 / float(E_keV)

    @staticmethod
    def _to_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            try:
                return float(str(v).replace("_", "").strip())
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"Expected float-compatible value, got {v!r}"
                ) from err

    @staticmethod
    def _to_int(v):
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            try:
                return int(float(str(v).replace("_", "").strip()))
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"Expected int-compatible value, got {v!r}"
                ) from err

    @classmethod
    def _extract_section(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError(
                "Top-level YAML must be a mapping of keys to values."
            )

        # Handle profile-based YAML structure (new format)
        if "profiles" in data and isinstance(data.get("profiles"), dict):
            active_profile = data.get("active_profile", "ISR").upper()
            profile_data = data["profiles"].get(active_profile, {})
            if isinstance(profile_data, dict):
                # Look for ExperimentSetup in the active profile
                for key in (
                    "ExperimentSetup",
                    "experiment",
                    "experiment_setup",
                ):
                    sec = profile_data.get(key)
                    if isinstance(sec, dict):
                        return sec
                # Check if profile data itself has required keys
                if any(k in profile_data for k in cls.REQUIRED_KEYS):
                    return profile_data

        # Handle flat YAML structure (old format or fallback)
        for key in ("ExperimentSetup", "experiment", "experiment_setup"):
            sec = data.get(key)
            if isinstance(sec, dict):
                return sec
        if any(k in data for k in cls.REQUIRED_KEYS):
            return data
        for v in data.values():
            if isinstance(v, dict) and any(k in v for k in cls.REQUIRED_KEYS):
                return v
        raise ValueError(
            "Could not find experiment setup in YAML. "
            "Expected an 'ExperimentSetup' section or flat keys."
        )

    @classmethod
    def from_yaml(cls, path: str | Path):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Experiment YAML not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        sec = cls._extract_section(doc)
        merged = {}
        for k in cls.REQUIRED_KEYS + ("wavelength",):
            if k in sec:
                merged[k] = sec[k]
            elif k in doc:
                merged[k] = doc[k]
        missing = [
            k
            for k in cls.REQUIRED_KEYS
            if merged.get(k) in (None, "", "None", "null")
        ]
        if missing:
            raise ValueError(f"Missing required keys in YAML: {missing}")
        params = {
            "distance": cls._to_float(merged["distance"]),
            "pitch": cls._to_float(merged["pitch"]),
            "ycenter": cls._to_int(merged["ycenter"]),
            "xcenter": cls._to_int(merged["xcenter"]),
            "xpixels": cls._to_int(merged["xpixels"]),
            "ypixels": cls._to_int(merged["ypixels"]),
            "energy": cls._to_float(merged["energy"]),
            "wavelength": merged.get("wavelength"),
        }
        return cls(**params)

    def __repr__(self):
        return (
            f"<ExperimentSetup: distance={self.distance} m, pitch={self.pitch} m, "
            f"xcenter={self.xcenter}, ycenter={self.ycenter}, "
            f"xpixels={self.xpixels}, ypixels={self.ypixels}, "
            f"energy={self.energy} keV, wavelength={self.wavelength} Å>"
        )


class Crystal:
    """
    Holds crystal lattice parameters and orientation matrix (UB).
    """

    def __init__(self, params, ub_matrix):
        # Parse crystal lattice parameters from #G1 line
        self.a, self.b, self.c = params[0:3]
        self.alpha, self.beta, self.gamma = params[3:6]
        self.a_hkl, self.b_hkl, self.c_hkl = params[6:9]
        self.alpha_hkl, self.beta_hkl, self.gamma_hkl = params[9:12]
        self.H_or0, self.K_or0, self.L_or0 = params[12:15]
        self.H_or1, self.K_or1, self.L_or1 = params[15:18]
        self.u00, self.u01, self.u02, self.u03, self.u04, self.u05 = params[
            18:24
        ]
        self.u10, self.u11, self.u12, self.u13, self.u14, self.u15 = params[
            24:30
        ]
        self.lambda0, self.lambda1 = params[30:32]
        self.u06, self.u16 = params[32:34]
        # Build UB matrix from the global #G3 (if any)
        ub = np.array(ub_matrix, dtype=float)
        self.UB = ub.reshape((3, 3))

    @classmethod
    def from_spec(cls, filename):
        """
        Parse the first encountered #G1 and #G3 lines from the SPEC file to construct a Crystal.
        """
        g1_vals, g3_vals = None, None
        with open(filename) as f:
            for line in f:
                if line.startswith("#G1 "):
                    g1_vals = [float(x) for x in line.split()[1:]]
                elif line.startswith("#G3 "):
                    g3_vals = [float(x) for x in line.split()[1:]]
                if g1_vals is not None and g3_vals is not None:
                    break
        if g1_vals is None or g3_vals is None:
            raise RuntimeError("Missing #G1 or #G3 in SPEC file for Crystal.")
        return cls(g1_vals, g3_vals)

    def __repr__(self):
        return (
            f"Crystal(a={self.a}, b={self.b}, c={self.c}, alpha={self.alpha}, beta={self.beta}, "
            f"gamma={self.gamma}, UB=\n{self.UB})"
        )


class ScanAngles:
    # Logical goniometer/reciprocal axes mapped to the many motor/column names
    # different SPEC configurations use. Matching is case-insensitive and order
    # independent, so a scan header may list these in any order under any of the
    # accepted aliases. Extend/override via the ``axis_aliases`` constructor arg.
    DEFAULT_AXIS_ALIASES = {
        "tth": (
            "vtth", "tth", "twotheta", "two_theta", "2theta", "ttheta",
            "del", "delta", "detth", "det_tth",
        ),
        "th": (
            "vth", "th", "theta", "eta", "omega", "om", "samth", "sth",
        ),
        "chi": ("chi", "vchi", "schi"),
        "phi": ("phi", "vphi", "sphi"),
        "h": ("h",),
        "k": ("k",),
        "l": ("l",),
    }
    # Logical angles that end up as goniometer columns in the output records.
    ANGLE_KEYS = ("tth", "th", "chi", "phi")

    # Modified __init__ to also accept a Crystal object (if needed later)
    def __init__(
        self,
        filename,
        crystal,
        npartitions=1,
        selected_scans=None,
        axis_aliases=None,
    ):
        self.filename = filename
        self.npartitions = npartitions
        self.crystal = crystal  # store the Crystal object
        self._alias_map = self._build_alias_map(axis_aliases)
        # Normalize selected_scans to a set of ints for fast membership tests
        if selected_scans is None:
            self._selected_scans = None
        else:
            try:
                if isinstance(selected_scans, (int, str)):
                    self._selected_scans = {int(selected_scans)}
                else:
                    self._selected_scans = {int(s) for s in selected_scans}
            except (TypeError, ValueError):
                self._selected_scans = {int(selected_scans)}

    @classmethod
    def _build_alias_map(cls, axis_aliases=None):
        """Build a reverse map {normalized_name: logical_axis}."""
        aliases = dict(cls.DEFAULT_AXIS_ALIASES)
        if axis_aliases:
            for logical, names in axis_aliases.items():
                if isinstance(names, str):
                    names = (names,)
                aliases[logical] = tuple(names)
        rev = {}
        for logical, names in aliases.items():
            for name in names:
                rev[str(name).strip().lower()] = logical
        return rev

    def _resolve(self, name):
        """Return the logical axis for a motor/column name, or None."""
        if name is None:
            return None
        return self._alias_map.get(str(name).strip().lower())

    def _find_col_for_logical(self, cols, logical):
        for i, c in enumerate(cols):
            if self._resolve(c) == logical:
                return i
        return None

    def _find_scan_col(self, cols, scan_motor):
        """Locate the scanned-motor column in an ascan #L header."""
        if scan_motor:
            target = str(scan_motor).strip().lower()
            for i, c in enumerate(cols):
                if str(c).strip().lower() == target:
                    return i
            logical = self._resolve(scan_motor)
            if logical is not None:
                idx = self._find_col_for_logical(cols, logical)
                if idx is not None:
                    return idx
        # In SPEC the scanned motor is conventionally the first data column.
        return 0 if cols else None

    def parse_all_scans(self):
        """
        Read the SPEC file and return a list of dicts with scan data.
        Each record will include:
          - scan_number: zero-padded scan number
          - data_number: zero-based, zero-padded data row index
          - type: scan type ('ascan' or 'hklscan')
          - tth, th, chi, phi: goniometer angles
          - h, k, l: reciprocal-lattice coordinates
          - ub: 3×3 UB matrix read from the "#G3" line within that scan (if present), otherwise None.

        Motor and column names are matched case-insensitively through the axis
        alias table, so SPEC files that use different names/orderings for the
        goniometer angles are handled without failing on "unknown" keywords.
        """
        # Collect every global "#O<n>" motor-name group so fixed-motor positions
        # from any "#P<n>" line can be resolved, regardless of grouping.
        o_groups = self._read_motor_name_groups()
        if not o_groups:
            warnings.warn(
                "SPEC file has no '#O' motor-name lines; fixed-motor angles "
                "for point scans may be unavailable.",
                stacklevel=2,
            )

        results = []
        cur_scan = None
        cur_type = None
        scan_motor = None  # scanned motor name from the "#S" command (ascan)
        skip_current = False  # scans not requested via selected_scans
        motor_pos = {}  # raw {motor_name: value} from "#P<n>" lines
        ctx = {}  # per-scan data-block parsing context
        in_data = False
        counter = 0
        current_ub = None  # per-scan UB

        with open(self.filename) as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("#S "):
                    parts = line.split()
                    cur_scan = int(parts[1])
                    cur_type = parts[2] if len(parts) > 2 else ""
                    # For point scans the scanned motor is the first command arg.
                    scan_motor = parts[3] if len(parts) > 3 else None
                    skip_current = (
                        self._selected_scans is not None
                        and cur_scan not in self._selected_scans
                    )
                    motor_pos = {}
                    ctx = {}
                    in_data = False
                    counter = 0
                    current_ub = None
                    continue

                if skip_current or cur_scan is None:
                    continue

                # Look for UB update within a scan: "#G3" line.
                if line.startswith("#G3 "):
                    ub_vals = [float(x) for x in line.split()[1:]]
                    if len(ub_vals) != 9:
                        raise RuntimeError(
                            f"Scan {cur_scan}: UB line does not have 9 values."
                        )
                    current_ub = np.array(ub_vals).reshape((3, 3))
                    continue

                # Fixed motor positions from any "#P<n>" line.
                if line.startswith("#P") and len(line) > 2 and line[2].isdigit():
                    head = line.split()
                    idx = head[0][2:]
                    names = o_groups.get(int(idx), []) if idx.isdigit() else []
                    for i, val in enumerate(head[1:]):
                        if i < len(names):
                            try:
                                motor_pos[names[i]] = float(val)
                            except ValueError:
                                continue
                    continue

                # Data header (#L): build the parsing context for this scan.
                if line.startswith("#L "):
                    cols = line.split()[1:]
                    ctype = (
                        cur_type.lower() if isinstance(cur_type, str) else ""
                    )
                    ctx = self._build_data_context(
                        ctype, cols, scan_motor, motor_pos
                    )
                    in_data = ctx.get("in_data", False)
                    continue

                if in_data:
                    if not line or (
                        line.startswith("#")
                        and not (len(line) > 1 and line[1].isdigit())
                    ):
                        in_data = False
                        continue
                    parts = line.split()
                    if len(parts) < ctx.get("max_idx", -1) + 1:
                        continue
                    results.append(
                        self._build_record(
                            cur_scan, counter, cur_type, current_ub, ctx, parts
                        )
                    )
                    counter += 1
        return results

    def _read_motor_name_groups(self):
        """Return {group_index: [motor_names]} from the global '#O<n>' lines."""
        o_groups = {}
        with open(self.filename) as f:
            for line in f:
                if line.startswith("#O"):
                    head = line.split(None, 1)
                    idx = head[0][2:]
                    if idx.isdigit():
                        o_groups[int(idx)] = (
                            head[1].split() if len(head) > 1 else []
                        )
                elif line.startswith("#S ") and o_groups:
                    # Motor-name lines live in the header, before the first scan.
                    break
        return o_groups

    def _build_data_context(self, ctype, cols, scan_motor, motor_pos):
        """Resolve column/motor positions for a scan's data block."""
        ctx = {"type": ctype, "in_data": False}

        # H/K/L data columns (resolved case-insensitively).
        hkl_idx = {}
        for i, c in enumerate(cols):
            logical = self._resolve(c)
            if logical in ("h", "k", "l"):
                hkl_idx.setdefault(logical, i)
        ctx["hkl_idx"] = hkl_idx

        if ctype == "hklscan":
            angle_idx = {}
            for key in self.ANGLE_KEYS:
                idx = self._find_col_for_logical(cols, key)
                if idx is not None:
                    angle_idx[key] = idx
            ctx["angle_idx"] = angle_idx
            ctx["in_data"] = True
        elif ctype:
            # Step scans (ascan, a2scan/aNscan, dscan/dNscan, ...): fixed
            # goniometer angles come from the motor positions, while any
            # scanned angle present as a data column is read per row.
            fixed = {k: None for k in self.ANGLE_KEYS}
            for name, val in motor_pos.items():
                logical = self._resolve(name)
                if logical in fixed:
                    fixed[logical] = val
            ctx["fixed_angles"] = fixed
            # Any angle present as a data column (covers multi-motor scans
            # like a2scan) overrides its fixed motor value per row.
            angle_idx = {}
            for key in self.ANGLE_KEYS:
                idx = self._find_col_for_logical(cols, key)
                if idx is not None:
                    angle_idx[key] = idx
            ctx["angle_idx"] = angle_idx
            # Fallback for a single scanned motor whose column name is not a
            # known angle alias: use SPEC's first-data-column convention.
            ctx["scan_col_idx"] = self._find_scan_col(cols, scan_motor)
            ctx["scan_logical"] = self._resolve(scan_motor)
            ctx["in_data"] = True

        indices = list(hkl_idx.values())
        indices.extend(ctx.get("angle_idx", {}).values())
        if ctx.get("scan_col_idx") is not None:
            indices.append(ctx["scan_col_idx"])
        ctx["max_idx"] = max(indices, default=-1)
        return ctx

    def _build_record(self, cur_scan, counter, cur_type, current_ub, ctx, parts):
        """Build one output record for a data row given the scan context."""

        def _get(idx):
            if idx is None:
                return None
            try:
                return float(parts[idx])
            except (IndexError, ValueError, TypeError):
                return None

        hkl_idx = ctx.get("hkl_idx", {})
        rec = {
            "scan_number": f"{cur_scan:03d}",
            "data_number": f"{counter:03d}",
            "type": cur_type,
            "ub": current_ub.copy() if current_ub is not None else None,
            "h": _get(hkl_idx.get("h")),
            "k": _get(hkl_idx.get("k")),
            "l": _get(hkl_idx.get("l")),
        }

        if ctx["type"] == "hklscan":
            angle_idx = ctx.get("angle_idx", {})
            for key in self.ANGLE_KEYS:
                rec[key] = _get(angle_idx.get(key))
        else:  # step scans: prefer per-row column values, fall back to fixed
            fixed = ctx.get("fixed_angles", {})
            angle_idx = ctx.get("angle_idx", {})
            for key in self.ANGLE_KEYS:
                val = _get(angle_idx.get(key)) if key in angle_idx else None
                rec[key] = val if val is not None else fixed.get(key)
            # Fallback: map the primary scanned motor by column position when
            # its column name is not a recognized angle alias.
            scan_logical = ctx.get("scan_logical")
            if scan_logical in self.ANGLE_KEYS and rec.get(scan_logical) is None:
                rec[scan_logical] = _get(ctx.get("scan_col_idx"))
        return rec

    def to_pandas(self):
        data = self.parse_all_scans()
        return pd.DataFrame(data)

    def to_dask(self):
        return dd.from_pandas(self.to_pandas(), npartitions=self.npartitions)


class SpecParser:
    """
    Aggregates ExperimentSetup, Crystal, and ScanAngles for a SPEC file.
    """

    def __init__(
        self,
        filename: str,
        setup_yaml: str,
        npartitions: int = 1,
        selected_scans=None,
        axis_aliases=None,
    ):
        self.filename = filename
        self.setup = ExperimentSetup.from_yaml(setup_yaml)
        self.crystal = Crystal.from_spec(filename)
        self.scans = ScanAngles(
            filename,
            self.crystal,
            npartitions=npartitions,
            selected_scans=selected_scans,
            axis_aliases=axis_aliases,
        )

    def to_pandas(self):
        df = self.scans.to_pandas()
        return df

    def to_dask(self):
        return self.scans.to_dask()
