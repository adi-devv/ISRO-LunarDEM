# ISRO-LunarDEM

## Overview
ISRO-LunarDEM is a Python-based pipeline for generating a conceptual Digital Elevation Model (DEM) of the lunar surface from Chandrayaan-2 orbiter imagery using Shape-from-Shading (SfS) techniques. The project parses ISRO-provided orbit, attitude, and sun parameter files to reconstruct lunar topography from a single image.


## Features
- Parses ISRO OATH, OAT, LBR, and SPM data files (see `data/readme.txt` for formats)
- Implements a conceptual iterative SfS algorithm
- Produces both relative and absolute DEMs (with strong assumptions)
- Visualizes results and saves outputs as PNG and NumPy arrays
- Includes an advanced/alternative pipeline in `main_p01.py` (commented, for further development)

## Data & Directory Structure
- `data/` — Contains required parameter files and the input image:
  - `moonframe.png` — Lunar surface image (grayscale)
  - `params.oath`, `params.oat`, `params.lbr`, `sun_params.spm` — ISRO-provided data files
  - `readme.txt` — Detailed data file format descriptions
- `output_dem.npy`, `output_dem.tif` — Output DEMs
- `output_p0/`, `output_p1/` — Additional outputs from different pipeline versions

## Requirements
- Python 3.7+
- numpy
- matplotlib
- pillow
- scipy

Install dependencies with:
```bash
pip install numpy matplotlib pillow scipy
```

## Usage
1. Place all required data files in the `data/` directory as described above.
2. **Update camera intrinsics**: Edit the `camera_intrinsic` dictionary in `main.py` with the correct values for your Chandrayaan-2 camera (see comments in code).
3. Run the main pipeline:
```bash
python main.py
```
4. Outputs:
   - `generated_lunar_dem_absolute_conceptual.png` — Visual DEM
   - `generated_lunar_dem_absolute_conceptual.npy` — DEM in meters (NumPy array)
   - Plots showing the original image, relative depth, and absolute DEM

## Advanced/Alternative Pipeline
- `main_p01.py` contains an enhanced, modular pipeline (commented, for further development) with advanced gradient computation, albedo/shadow estimation, and GPU support.

## Data File Formats
See `data/readme.txt` for detailed descriptions of the OATH, OAT, LBR, and SPM file formats. These files are parsed by custom routines in `main.py`.

## Troubleshooting
- **File not found**: Ensure all required files are present in `data/`.
- **Camera intrinsics warning**: Update the placeholder values in `main.py` for accurate DEMs.
- **Output DEM is flat or unrealistic**: The pipeline is conceptual and relies on strong assumptions; results may not be geodetically accurate without further calibration.

## Credits
- ISRO for Chandrayaan-2 data formats and mission
- Open-source Python libraries: numpy, matplotlib, pillow, scipy

## References
- See `data/readme.txt` for ISRO data file format details
- [Chandrayaan-2 Mission Overview](https://www.isro.gov.in/Chandrayaan2.html) 
