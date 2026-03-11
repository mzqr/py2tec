# py2tec

Python library for generating Tecplot binary PLT files.

## Install

```bash
pip install numpy
# Copy tecplot.py to your project
```

## Quick Start

```python
import numpy as np
from tecplot import TEC_FILE

# Create data
x = np.linspace(0, 10, 100).reshape(100, 1)
y = np.linspace(0, 10, 100).reshape(1, 100)
X = x + np.zeros_like(y)
Y = y + np.zeros_like(x)
Z = np.sin(X) * np.cos(Y)

# Write PLT file
tec = TEC_FILE()
tec.FileName = 'output'
tec.Variables = ['x', 'y', 'z']
tec.add_datas([X, Y, Z])
tec.write_plt()
```

## API

### TEC_FILE

Main class for creating Tecplot PLT files.

**Properties:**
- `FilePath` - Output directory (default: '.')
- `FileName` - Output filename without extension
- `Title` - File title
- `Variables` - List of variable names
- `Zones` - List of TEC_ZONE objects
- `Echo_Mode` - 'brief', 'full', 'simple', 'none', 'leave'

**Methods:**
- `write_plt()` - Write PLT file
- `AddData(data, name=None)` - Add single data array
- `add_datas(data_list, name_list=None, auto_name=True)` - Batch add data

### TEC_ZONE

Data zone container.

**Properties:**
- `ZoneName` - Zone name
- `Data` - List of numpy arrays
- `Skip` - Subsample stride [I,J,K]
- `Begin` - Skip from start [I,J,K]
- `EEnd` - Skip from end [I,J,K]

### Utility Functions

- `s2i(s)` - String to int32 array
- `real_ijk(ijk, skip, begin, eend)` - Calculate real dimensions
- `makebuf(data, skip, begin, eend)` - Create subsampled buffer
- `gettype(data)` - Get Tecplot type code

## Data Types

- `float32` → Float (1)
- `float64` → Double (2)
- `int32` → LongInt (3)
- `int16` → ShortInt (4)
- `int8` → Byte (5)

## Notes

- Data uses column-major (Fortran) order
- Binary output uses little-endian byte order
- Variable count must match Data count

## License

MIT
