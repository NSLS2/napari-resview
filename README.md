# napari-resview

[![License BSD-3](https://img.shields.io/pypi/l/napari-resview.svg?color=green)](https://github.com/XYangXRay/napari-resview/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-resview.svg?color=green)](https://pypi.org/project/napari-resview)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-resview.svg?color=green)](https://python.org)
[![tests](https://github.com/XYangXRay/napari-resview/workflows/tests/badge.svg)](https://github.com/XYangXRay/napari-resview/actions)
[![codecov](https://codecov.io/gh/XYangXRay/napari-resview/branch/main/graph/badge.svg)](https://codecov.io/gh/XYangXRay/napari-resview)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-resview)](https://napari-hub.org/plugins/napari-resview)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

**A tool of 3D reciprocal space mapping (RSM) for X-ray diffraction experiments with interactive data processing, visualization and analysis built on Napari platfom.**

---

## Installation

You can install `napari-resview` via [pip]:

```
pip install napari-resview
```

If napari is not already installed, you can install `napari-resview` with napari and Qt via:

```
pip install "napari-resview[all]"
```

To install latest development version:

```
pip install git+https://github.com/XYangXRay/napari-resview.git
```

---

## Overview

`napari-resview` is a comprehensive processing, visualization and analysis tool designed for synchrotron X-ray diffraction experiments. Built as a [napari] plugin, it enables researchers to interactively explore, process, and visualize 3D reciprocal space maps with advanced features for data loading, ROI selection, and crystallographic analysis.

### Key Features

- **🔬 Multi-Beamline Support**: Pre-configured profiles for ISR and CMS beamlines with YAML-based configuration
- **📊 3D Visualization**: Interactive 3D rendering of reciprocal space maps using napari's powerful volume rendering
- **✂️ ROI-Based Cropping**: Define and apply regions of interest with visual feedback
- **🏗️ RSM Construction Pipeline**: Complete workflow from raw SPEC/TIFF data to 3D reciprocal space
- **📐 UB Matrix Management**: Tools for orientation matrix setup, refinement, and validation
- **⚡ Asynchronous Processing**: Non-blocking data loading with progress tracking
- **💾 Multiple Export Formats**: Export to VTR (VTK) format for use with ParaView and other tools
- **🎛️ Profile-Based Workflow**: Save and restore experimental configurations
- **📈 Real-time Data Integration**: Merge SPEC metadata with TIFF intensity frames on-the-fly

### Use Cases

- **Synchrotron Beamline Analysis**: Real-time and offline data processing at NSLS-II beamlines
- **Crystal Structure Studies**: Reciprocal space reconstruction for structural analysis
- **Strain Mapping**: Analyze crystal strain and deformation through reciprocal space features
- **Quality Control**: Quick visualization and validation of diffraction data quality
- **Research & Education**: Interactive tool for teaching crystallography and X-ray diffraction concepts

---

## Quick Start

### Launching the Plugin

After installation, launch napari and open the ResView widget:

```bash
napari
```

Then in napari: **Plugins → napari-resview: ResView Widget**

### Basic Workflow

1. **Configure Loader Profile**

   - Select beamline profile (ISR or CMS)
   - Specify SPEC file, setup YAML, and TIFF directory
   - Configure any additional loader parameters
2. **Load Data**

   - Click "Load" to asynchronously load and merge data
   - Monitor progress in the status panel
   - View intensity frames in the napari viewer
3. **Apply ROI Cropping (Optional)**

   - Draw ROI shapes on the intensity viewer
   - Click "Crop from ROI" to apply cropping
   - Cropped data is used for all subsequent operations
4. **Build RSM**

   - Configure UB matrix parameters
   - Set resolution and bounds
   - Click "Build" to construct reciprocal space map
5. **Visualize**

   - Adjust visualization settings (colormap, opacity, contrast)
   - Add grid overlays and axis markers
   - Export slices or subvolumes
6. **Export**

   - Save RSM volume to VTR format
   - Export for use in ParaView or other visualization tools

---

## Detailed Usage

### Data Loading

The plugin supports two main beamline configurations:

#### ISR Loader

```python
from napari_resview.data_io import RSMDataLoader_ISR

loader = RSMDataLoader_ISR(
    spec_file="path/to/scan.spec",
    setup_file="path/to/setup.yaml",
    tiff_dir="path/to/tiff_images/",
    use_dask=False,
    process_hklscan_only=True,
    selected_scans=[21, 22, 23]
)
setup, ub, df = loader.load()
```

#### CMS Loader

```python
from napari_resview.data_io import RSMDataloader_CMS

loader = RSMDataloader_CMS(
    spec_file="path/to/scan.spec",
    setup_file="path/to/setup.yaml",
    ub_file="path/to/ub.txt",
    use_dask=False
)
setup, ub, df = loader.load()
```

### Building Reciprocal Space Maps

```python
from napari_resview.rsm3d import RSMBuilder

# Initialize builder with loaded data
builder = RSMBuilder(
    setup,
    ub,
    df,
    ub_includes_2pi=True,
    center_is_one_based=False
)

# Build RSM with specified resolution and bounds
rsm_grid, rsm_axes = builder.build(
    resolution=200,
    bounds={'qx': (-2, 2), 'qy': (-2, 2), 'qz': (-2, 2)}
)
```

### Visualization

```python
from napari_resview.data_viz import RSMNapariViewer
import napari

# Create viewer
rsm_viewer = RSMNapariViewer(
    grid=rsm_grid,
    axes=rsm_axes,
    axes_names=['qx', 'qy', 'qz']
)

# Launch in napari
viewer = rsm_viewer.launch()

# Add grid overlay
rsm_viewer.add_grid_overlay(step=0.5, viewer=viewer)
```

### Configuration Files

#### Setup YAML Example

```yaml
beamline: ISR
energy: 10.0  # keV
wavelength: 1.2398  # Angstroms
detector:
  pixel_size: 55e-6  # meters
  distance: 0.5  # meters
  beam_center: [512, 512]
  size: [1024, 1024]
angles:
  omega: 0.0
  chi: 90.0
  phi: 0.0
  tth: 20.0
```

#### Profile Persistence

The plugin automatically saves and restores your configuration in `rsm3d_defaults.yaml`, including:

- Active profile (ISR/CMS)
- File paths and directories
- UB matrix values
- Resolution and bounds settings
- Visualization preferences

---

## Key Components

### Data I/O (`data_io.py`)

- `RSMDataLoader_ISR`: Load ISR beamline SPEC + TIFF data
- `RSMDataloader_CMS`: Load CMS beamline data with HDF5 support
- `ExperimentSetup`: Manage experimental configuration
- `write_rsm_volume_to_vtr()`: Export RSM to VTK format

### RSM Builder (`rsm3d.py`)

- `RSMBuilder`: Construct 3D reciprocal space maps from experimental data
- Supports xrayutilities coordinate transformations
- Flexible motor mapping and axis configuration
- Gridding and interpolation options

### Visualization (`data_viz.py`)

- `RSMNapariViewer`: 3D volume rendering of RSM
- `IntensityNapariViewer`: 2D intensity frame viewer with ROI tools
- Grid overlays and coordinate displays
- Interactive slice extraction

### Widget (`resview_widget.py`)

- `ResviewDockWidget`: Main napari dock widget
- Tabbed interface: Data, Build, View, Export
- Profile management with YAML persistence
- Asynchronous data loading with progress tracking

---

## API Reference

### Main Classes

#### RSMDataLoader_ISR

```python
loader = RSMDataLoader_ISR(
    spec_file: str,          # Path to SPEC file
    setup_file: str,         # Path to setup YAML
    tiff_dir: str,           # Directory containing TIFF images
    use_dask: bool = False,  # Use Dask for large datasets
    process_hklscan_only: bool = False,  # Filter hklscan only
    selected_scans: list = None  # List of scan numbers to process
)
setup, ub, df = loader.load()
```

#### RSMBuilder

```python
builder = RSMBuilder(
    setup,                   # ExperimentSetup object
    UB,                      # UB orientation matrix (3x3)
    df,                      # DataFrame with intensity data
    ub_includes_2pi: bool = True,  # UB includes 2π factor
    center_is_one_based: bool = False,  # Beam center indexing
    sample_axes: list = None,  # xrayutilities sample axes
    detector_axes: list = None  # xrayutilities detector axes
)

grid, axes = builder.build(
    resolution: int = 200,   # Grid resolution
    bounds: dict = None      # {'qx': (min, max), 'qy': (min, max), 'qz': (min, max)}
)
```

#### RSMNapariViewer

```python
viewer = RSMNapariViewer(
    grid: np.ndarray,        # 3D intensity array
    axes,                    # Coordinate axes (list of 3 arrays)
    axes_names: list = ['qx', 'qy', 'qz']  # Axis labels
)

napari_viewer = viewer.launch()  # Open in napari
viewer.add_grid_overlay(step=0.5, viewer=napari_viewer)
viewer.add_slices(positions={'qx': 0, 'qy': 0, 'qz': 0})
```

---

## Development

### Setting Up Development Environment

```bash
git clone https://github.com/NSLS2/napari-resview.git
cd napari-resview
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=napari_resview --cov-report=html

# Run specific test file
pytest tests/test_resview_widget.py
```

### Code Quality

```bash
# Format code with black
black src/

# Sort imports
isort src/

# Type checking
mypy src/
```

### Using tox

The project includes tox configuration for automated testing:

```bash
# Run all tox environments
tox

# Run specific environment
tox -e py310
```

---

## Contributing

Contributions are very welcome! We appreciate bug reports, feature requests, documentation improvements, and code contributions.

### How to Contribute

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/napari-resview.git
   cd napari-resview
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and add tests if applicable
5. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```
6. **Commit your changes** with clear commit messages:
   ```bash
   git commit -m "Add: description of your changes"
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Submit a pull request** on GitHub

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure the test coverage stays the same or improves
- Write clear commit messages
- Keep pull requests focused on a single feature or fix

### Reporting Issues

If you encounter bugs or have feature requests, please [file an issue] with:

- A clear, descriptive title
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Your environment (OS, Python version, napari version)
- Relevant error messages or screenshots

---

## Troubleshooting

### Common Issues

**Plugin doesn't appear in napari menu**

- Ensure napari-resview is installed in the same environment as napari
- Restart napari after installation
- Check: `napari --plugin-info` to verify plugin is detected

**Data loading fails**

- Verify file paths are correct
- Check that TIFF files match SPEC scan numbers
- Ensure setup YAML has correct format
- Review error messages in napari's terminal/console

**Memory errors with large datasets**

- Enable `use_dask=True` for lazy loading
- Reduce resolution in Build settings
- Apply ROI cropping to reduce data size
- Close other memory-intensive applications

**Visualization appears empty or black**

- Adjust contrast limits in napari layer controls
- Check data range with `print(grid.min(), grid.max())`
- Ensure data is non-zero
- Try different colormaps or opacity settings

**UB matrix errors**

- Verify UB matrix dimensions (3×3)
- Check `ub_includes_2pi` setting matches your data
- Ensure proper motor mapping in configuration

### Getting Help

- Check the [napari documentation](https://napari.org/stable/)
- Browse [existing issues](https://github.com/NSLS2/napari-resview/issues)
- Ask questions in napari [community forum](https://forum.image.sc/tag/napari)
- Contact the development team at yangxg@bnl.gov

---

## Citation

If you use napari-resview in your research, please cite:

```bibtex
@software{napari_resview,
  title = {napari-resview: A napari plugin for 3D reciprocal space mapping},
  author = {Yang, Xiaogang and contributors},
  year = {2024},
  url = {https://github.com/NSLS2/napari-resview},
  note = {Developed at Brookhaven National Laboratory, NSLS-II}
}
```

Also consider citing the underlying tools:

- [napari](https://napari.org/): Multi-dimensional image viewer
- [xrayutilities](https://xrayutilities.sourceforge.io/): X-ray diffraction analysis

---

## Acknowledgments

- Developed at **Brookhaven National Laboratory (BNL)**, National Synchrotron Light Source II (NSLS-II)
- Built using the [napari] framework and [napari-plugin-template]
- Powered by [xrayutilities] for crystallographic calculations
- Thanks to the napari community for excellent documentation and support

---

## Related Projects

- [napari](https://napari.org/) - Multi-dimensional image viewer for Python
- [xrayutilities](https://xrayutilities.sourceforge.io/) - X-ray diffraction analysis
- [PyMca](http://pymca.sourceforge.net/) - X-ray fluorescence toolkit
- [DIOPTAS](https://dioptas.readthedocs.io/) - GUI for 2D X-ray diffraction data
- [GSAS-II](https://subversion.xray.aps.anl.gov/trac/pyGSAS) - Crystallographic data analysis

---

## License

Distributed under the terms of the [BSD-3] license, "napari-resview" is free and open source software.

See [LICENSE](LICENSE) for full details.

---

## Issues

If you encounter any problems, please [file an issue] along with:

- A detailed description of the problem
- Steps to reproduce the issue
- Your environment information (OS, Python version, package versions)
- Relevant error messages or logs
- Screenshots if applicable

---

## Changelog

### Version History

See the [GitHub releases](https://github.com/NSLS2/napari-resview/releases) page for version history and changelog.

---

## Contact

- **Maintainer**: Xiaogang Yang (yangxg@bnl.gov)
- **Institution**: Brookhaven National Laboratory, NSLS-II
- **Repository**: https://github.com/NSLS2/napari-resview
- **Issues**: https://github.com/NSLS2/napari-resview/issues

---

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[xrayutilities]: https://xrayutilities.sourceforge.io/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[file an issue]: https://github.com/NSLS2/napari-resview/issues
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
