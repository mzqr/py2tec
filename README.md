

# Tecplot PLT File Generation Library (Python)

This is a Python translation of the MATLAB `liton_ordered_tec` library for generating Tecplot `.plt` files.

## Features

- Full port of MATLAB TEC_FILE and TEC_ZONE classes
- Support for multiple data zones in a single file
- Support for 1D, 2D, and 3D data
- Data subsampling with Skip, Begin, and EEnd parameters
- Support for multiple data types: float32, float64, int32, int16, int8
- Configurable echo modes for debugging
- Column-major (Fortran) data order for Tecplot compatibility

## Files

- **tecplot.py**: Main library containing all classes and functions
- **test_tecplot.py**: Comprehensive test suite demonstrating usage
- **README.md**: This file

## Classes

### TEC_FILE
Main class for creating Tecplot PLT files.

**Properties:**
- `FilePath`: Output directory (default: '.')
- `FileName`: Output filename without extension (default: 'untitled_file')
- `Title`: File title (default: 'untitled')
- `Variables`: List of variable names
- `FileType`: File type (0=FULL, 1=GRID, 2=SOLUTION, default: 0)
- `Zones`: List of TEC_ZONE objects
- `Auxiliary`: Auxiliary data (optional)

**Echo Modes:** 'brief', 'full', 'simple', 'none', 'leave'

**Methods:**
- `write_plt()`: Write the PLT file to disk
- `set_echo_mode(file_mode, zone_mode)`: Set echo modes for file and zones

### TEC_ZONE
Class for managing data zones.

**Properties:**
- `ZoneName`: Zone identifier (default: 'untitled_zone')
- `StrandId`: Strand ID (default: -1)
- `SolutionTime`: Solution time (default: 0.0)
- `Skip`: Skip values for subsampling [i,j,k] (default: [1,1,1])
- `Begin`: Skip values from beginning [i,j,k] (default: [0,0,0])
- `EEnd`: Skip values from end [i,j,k] (default: [0,0,0])
- `Data`: List of numpy arrays (one per variable)
- `Auxiliary`: Auxiliary data (optional)

**Methods:**
- `gather_real_size(n)`: Calculate real data dimensions

## Utility Functions

- `s2i(s)`: Convert string to int32 array with null terminator
- `real_ijk(ijk, skip, begin, eend)`: Calculate real IJK dimensions
- `makebuf(data, skip, begin, eend)`: Create buffer with subsampling
- `gettype(data)`: Get data type code and size

## Supported Data Types

- `float32`: Float (type code: 1)
- `float64`: Double (type code: 2)
- `int32`: LongInt (type code: 3)
- `int16`: ShortInt (type code: 4)
- `int8`: Byte (type code: 5)

## Basic Usage

### Simplified API (Recommended)

```python
import numpy as np
from tecplot import TEC_FILE

# Create TEC_FILE object
tec_file = TEC_FILE()
tec_file.FileName = 'output'
tec_file.Variables = ['x', 'y', 'z', 'pressure']

# Create 2D grid data
nx, ny = 100, 100
x = np.linspace(0, 10, nx).reshape(nx, 1)
y = np.linspace(0, 10, ny).reshape(1, ny)
X = x + np.zeros_like(y)
Y = y + np.zeros_like(x)
Z = np.sin(X) * np.cos(Y)
P = Z * 1000

# Add data one by one (automatic zone creation)
tec_file.AddData(X, name='x')
tec_file.AddData(Y, name='y')
tec_file.AddData(Z, name='z')
tec_file.AddData(P)  # Uses 'pressure' from Variables

# Write PLT file
tec_file.write_plt()
```

### Traditional API

```python
import numpy as np
from tecplot import TEC_FILE, TEC_ZONE

# Create TEC_FILE object
tec_file = TEC_FILE()
tec_file.FileName = 'output'
tec_file.Variables = ['x', 'y', 'z', 'pressure']

# Create TEC_ZONE object
tec_file.Zones = [TEC_ZONE()]

# Create 2D grid data
nx, ny = 100, 100
x = np.linspace(0, 10, nx).reshape(nx, 1)
y = np.linspace(0, 10, ny).reshape(1, ny)
X = x + np.zeros_like(y)
Y = y + np.zeros_like(x)
Z = np.sin(X) * np.cos(Y)
P = Z * 1000

# Assign data (must match variable order)
tec_file.Zones[0].Data = [X, Y, Z, P]
tec_file.Zones[0].ZoneName = 'MyZone'

# Write PLT file
tec_file.write_plt()
```

### Key Differences

| Feature | Simplified API | Traditional API |
|----------|----------------|------------------|
| Zone creation | Automatic (first AddData call) | Manual (TEC_ZONE()) |
| Variable names | Auto from Variables list | Must match order manually |
| Code lines | ~10 lines | ~15 lines |
| Complexity | Low | High |

## Running Tests

```bash
python test_tecplot.py
```

The test suite includes:
1. Simple 2D data
2. Multiple zones
3. Skip and Begin parameters (subsampling)
4. 3D volumetric data
5. Mixed data types
6. Different echo modes

## Comparison with MATLAB Version

This Python translation maintains the same API and functionality as the original MATLAB version:

| MATLAB | Python |
|--------|--------|
| `TEC_FILE()` | `TEC_FILE()` |
| `TEC_ZONE()` | `TEC_ZONE()` |
| Cell arrays | Python lists |
| MATLAB arrays | NumPy arrays |
| `tic/toc` | `time.time()` |
| `fwrite(fid, value, type)` | `struct.pack()` and binary file I/O |

## Notes

- NumPy is required for array operations
- Binary output uses little-endian byte order (compatible with Tecplot)
- Variable data order must match `Variables` list
- Zone names should be unique for proper identification
- **Important**: Data is written in column-major (Fortran) order to match MATLAB output

## Recent Improvements

### Version 1.0 - 2026-01-30
- ✅ Fixed Echo_Mode initialization bug in constructors
- ✅ Added complete type hints throughout the codebase
- ✅ Optimized s2i function using bytearray (better performance)
- ✅ Extracted repeated string formatting to helper functions
- ✅ Added comprehensive documentation with examples
- ✅ Fixed critical data order issue (column-major vs row-major)
- ✅ Full test suite with 6 test cases

## Requirements

- Python 3.6+
- NumPy 1.16+
