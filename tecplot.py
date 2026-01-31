"""
Tecplot PLT file generation library for Python
Translation of MATLAB TEC_FILE and TEC_ZONE classes
"""

import struct
import time
from datetime import datetime
from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np


# ============================================================================
# Utility Functions
# ============================================================================

def s2i(s: Union[str, List[str]]) -> List[int]:
    """
    Convert string or list of strings to int32 array with null terminator.
    
    Args:
        s: String or list of strings to convert
        
    Returns:
        List of int32 values (ASCII codes with null terminator)
    """
    if isinstance(s, list):
        # Use bytearray for better performance with large strings
        result = bytearray()
        for item in s:
            result.extend(item.encode('ascii'))
            result.append(0)  # null terminator
        return list(result)
    else:
        # Single string - use bytearray for better performance
        result = bytearray(s.encode('ascii'))
        result.append(0)  # null terminator
        return list(result)


def real_ijk(ijk: List[int], skip: List[int], begin: List[int], eend: List[int]) -> List[int]:
    """
    Calculate real IJK from array dimensions with Skip, Begin and EEnd.
    
    Args:
        ijk: Original dimensions
        skip: Skip values
        begin: Begin values
        eend: EEnd values
        
    Returns:
        Real IJK dimensions
    """
    if len(ijk) == 2:
        ijk = list(ijk) + [1]
    
    rijk = [(ijk[i] - begin[i] - eend[i]) / skip[i] for i in range(3)]
    rijk = [int(np.floor(v)) for v in rijk]
    
    # Adjust if remainder exists
    for i in range(3):
        if (ijk[i] - begin[i] - eend[i]) % skip[i] != 0:
            rijk[i] += 1
    
    return rijk


def makebuf(data: np.ndarray, skip: List[int], begin: List[int], eend: List[int]) -> np.ndarray:
    """
    Make buffer data from array with Skip, Begin and EEnd.
    
    NOTE: MATLAB uses column-major (Fortran) order, while NumPy uses row-major (C) order.
    This function ensures data is flattened in column-major order to match MATLAB output.
    
    Args:
        data: Input data array
        skip: Skip values
        begin: Begin values
        eend: EEnd values
        
    Returns:
        Buffered data array (flattened in column-major order)
    """
    if skip == [1, 1, 1] and begin == [0, 0, 0] and eend == [0, 0, 0]:
        buf = data.flatten(order='F')
    else:
        # Handle different dimensions
        if data.ndim == 1:
            buf = data[begin[0]:data.shape[0]-eend[0]:skip[0]]
        elif data.ndim == 2:
            buf = data[begin[0]:data.shape[0]-eend[0]:skip[0],
                     begin[1]:data.shape[1]-eend[1]:skip[1]]
        else:  # 3D
            buf = data[begin[0]:data.shape[0]-eend[0]:skip[0],
                     begin[1]:data.shape[1]-eend[1]:skip[1],
                     begin[2]:data.shape[2]-eend[2]:skip[2]]
    
    if buf.size == 0:
        raise RuntimeError('one of Skip or Begin or EEnd is error')
    
    # Ensure column-major order for Tecplot compatibility with MATLAB
    return buf.flatten(order='F')


def gettype(data: np.ndarray) -> Tuple[int, int]:
    """
    Get the data type code and size.
    
    Args:
        data: Input data array
        
    Returns:
        Tuple of (type_code, size_in_bytes)
        
    Type codes:
        1: Float (float32)
        2: Double (float64)
        3: LongInt (int32)
        4: ShortInt (int16)
        5: Byte (int8)
    """
    dtype = data.dtype
    
    if dtype == np.float32:
        ty = 1
        si = 4
    elif dtype == np.float64:
        ty = 2
        si = 8
    elif dtype == np.int32:
        ty = 3
        si = 4
    elif dtype == np.int16:
        ty = 4
        si = 2
    elif dtype == np.int8:
        ty = 5
        si = 1
    else:
        raise RuntimeError(f'class({dtype}) not supported')
    
    return ty, si


# ============================================================================
# Base Classes
# ============================================================================

class TEC_FILE_BASE:
    """Base class for TEC_FILE"""
    
    def __init__(self):
        self.FilePath: str = '.'
        self.FileName: str = 'untitled_file'
        self.Title: str = 'untitled'
        self.Variables: List[str] = []
        self.FileType: int = 0  # 0 = FULL, 1 = GRID, 2 = SOLUTION
        self.Auxiliary: List = []


class TEC_ZONE_BASE:
    """Base class for TEC_ZONE"""
    
    def __init__(self):
        self.ZoneName: str = 'untitled_zone'
        self.StrandId: int = -1
        self.SolutionTime: float = 0.0
        self.Skip: List[int] = [1, 1, 1]
        self.Begin: List[int] = [0, 0, 0]
        self.EEnd: List[int] = [0, 0, 0]
        self.Auxiliary: List = []


# ============================================================================
# Helper Functions
# ============================================================================

def _format_dims(label: str, max_label: str, values: List[int], count: int) -> str:
    """
    Format dimension list for echo output.
    
    Args:
        label: Dimension label (e.g., 'Dim', 'Org_Dim')
        max_label: Max dimension label (e.g., 'Real_Max', 'Org_Max')
        values: List of dimension values
        count: Number of dimensions to include
        
    Returns:
        Formatted string
    """
    result = f'     {label} = {count}   {max_label} = ['
    result += ' '.join(str(v) for v in values[:count])
    result += ' ]'
    return result


def _format_vector(label: str, values: List[int]) -> str:
    """
    Format vector list for echo output.
    
    Args:
        label: Vector label (e.g., 'Skip', 'Begin')
        values: List of vector values
        
    Returns:
        Formatted string
    """
    result = f'     {label} = [ {" ".join(str(v) for v in values)} ]'
    return result


# ============================================================================
# Main Classes
# ============================================================================

class TEC_FILE(TEC_FILE_BASE):
    """
    Main TEC_FILE class for generating Tecplot PLT files.
    
    This class handles the creation of Tecplot binary PLT files with support for
    multiple data zones, different data types, and various file formats.
    
    Echo modes: 'brief', 'full', 'simple', 'none', 'leave'
    Echo_Mode flags: [file_head, file_end, variable, section, size, time, usingtime]
    
    Examples:
        >>> import numpy as np
        >>> from tecplot import TEC_FILE, TEC_ZONE
        >>>
        >>> # Create a new TEC_FILE
        >>> tec_file = TEC_FILE()
        >>> tec_file.FileName = 'output'
        >>> tec_file.Variables = ['x', 'y', 'z', 'pressure']
        >>>
        >>> # Add a zone with data
        >>> tec_file.Zones = [TEC_ZONE()]
        >>> x = np.linspace(0, 10, 100).reshape(100, 1)
        >>> y = np.linspace(0, 10, 100).reshape(1, 100)
        >>> X = x + np.zeros_like(y)
        >>> Y = y + np.zeros_like(x)
        >>> Z = np.sin(X) * np.cos(Y)
        >>> P = Z * 1000
        >>>
        >>> tec_file.Zones[0].Data = [X, Y, Z, P]
        >>> tec_file.Zones[0].ZoneName = 'Zone1'
        >>>
        >>> # Write to file
        >>> tec_file.write_plt()
    """
    
    def __init__(self, *args):
        super().__init__()
        self.Zones: List['TEC_ZONE'] = []
        self.last_log: Dict[str, Any] = {}
        # Initialize _echo_mode before setting Echo_Mode property
        self._echo_mode: List[bool] = [True, True, True, False, False, True, False]
        # Now set the Echo_Mode property (will use the setter)
        self.Echo_Mode: Union[str, List[bool]] = 'brief'
        
        if len(args) == 0:
            # Default values from base class
            pass
        elif len(args) == 1:
            n = args[0]
            if isinstance(n, int) and n == n:
                # Create array of TEC_ZONE objects
                self.Zones = [TEC_ZONE() for _ in range(n)]
            else:
                raise RuntimeError('input of TEC_FILE constructor must be a positive integer')
        else:
            raise RuntimeError('TEC_FILE constructor too many input arguments')
    
    @property
    def Echo_Mode(self) -> List[bool]:
        return self._echo_mode
    
    @Echo_Mode.setter
    def Echo_Mode(self, mode: Union[str, List[bool]]):
        if isinstance(mode, list):
            if all(isinstance(x, bool) for x in mode):
                if len(mode) == 7:
                    self._echo_mode = mode
                elif len(mode) == 9:  # zone mode
                    self._echo_mode = mode
                else:
                    raise RuntimeError('echo_mode code size wrong')
            else:
                raise RuntimeError(f'echo_mode type wrong ({type(mode)})')
        elif isinstance(mode, str):
            if mode == 'brief':
                self._echo_mode = [True, True, True, False, False, True, False]
            elif mode == 'full':
                self._echo_mode = [True] * 7
            elif mode == 'simple':
                self._echo_mode = [True, False, False, False, False, True, False]
            elif mode == 'none':
                self._echo_mode = [False] * 7
            elif mode == 'leave':
                pass  # Don't change
            else:
                raise RuntimeError(f'echo_mode code string wrong ("{mode}")')
        else:
            raise RuntimeError(f'echo_mode type wrong ({type(mode)})')
    
    def set_echo_mode(self, file_mode: Union[str, List[bool]] = 'leave', 
                     zone_mode: Union[str, List[bool]] = 'leave') -> 'TEC_FILE':
        """
        Set echo mode for file and all zones.
        
        Args:
            file_mode: File echo mode
            zone_mode: Zone echo mode
            
        Returns:
            Self for chaining
        """
        self.Echo_Mode = file_mode
        for zone in self.Zones:
            zone.Echo_Mode = zone_mode
        return self
    
    def write_plt(self) -> 'TEC_FILE':
        """
        Write PLT file to disk.
        
        Returns:
            Self for chaining
        """
        start_time = time.time()
        
        # Initialize log
        self.last_log = {
            'Time_Begin': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            'Echo_Text': [],
            'Zones': [],
            'Size': 0.0,
            'UsingTime': 0.0,
            'Time_End': ''
        }
        
        # Pre-write validation
        self._write_plt_pre()
        
        # File begin echo
        if self.Echo_Mode[0]:
            if self.Echo_Mode[5]:
                buf = f'[{self.last_log["Time_Begin"]}]'
                self.last_log['Echo_Text'].append(buf)
                print(buf, end='')
            
            buf = f'#### create file {self.FilePath}/{self.FileName}.plt ####'
            if self.Echo_Mode[5]:
                self.last_log['Echo_Text'][-1] += buf
                print(buf)
            else:
                self.last_log['Echo_Text'].append(buf)
                print(buf)
        
        # Open file
        filepath = f'{self.FilePath}/{self.FileName}.plt'
        try:
            with open(filepath, 'wb') as fid:
                # I. HEADER SECTION
                self._write_plt_head(fid)
                
                # EOHMARKER, value=357.0
                fid.write(struct.pack('<f', 357.0))
                
                if self.Echo_Mode[3]:
                    buf = '-------------------------------------'
                    self.last_log['Echo_Text'].append(buf)
                    print(buf)
                
                # II. DATA SECTION
                self._write_plt_data(fid)
                
                # Get file size
                fid.seek(0, 2)  # Seek to end
                pos = fid.tell()
                self.last_log['Size'] = pos / 1024 / 1024
                
                if self.Echo_Mode[4]:
                    buf = f'     file size: {self.last_log["Size"]:.1f} MB'
                    self.last_log['Echo_Text'].append(buf)
                    print(buf)
        
        except Exception as e:
            import os
            if os.path.exists(filepath):
                os.remove(filepath)
            raise
        
        # Using time
        self.last_log['UsingTime'] = time.time() - start_time
        if self.Echo_Mode[6]:
            buf = f'     using time: {self.last_log["UsingTime"]:.5f} s'
            self.last_log['Echo_Text'].append(buf)
            print(buf)
        
        # File end echo
        self.last_log['Time_End'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        if self.Echo_Mode[1]:
            if self.Echo_Mode[5]:
                buf = f'[{self.last_log["Time_End"]}]'
                self.last_log['Echo_Text'].append(buf)
                print(buf, end='')
            
            buf = f'#### create file {self.FilePath}/{self.FileName}.plt ####'
            if self.Echo_Mode[5]:
                self.last_log['Echo_Text'][-1] += buf
                print(buf)
            else:
                self.last_log['Echo_Text'].append(buf)
                print(buf)
        
        return self
    
    def _write_plt_pre(self):
        """Pre-write validation and preparation."""
        if not self.Variables:
            raise RuntimeError(f'FILE[{self.FileName}]: TEC_FILE.Variables is empty')
        
        if not self.Zones:
            raise RuntimeError(f'FILE[{self.FileName}]: TEC_FILE.Zones is empty')
        
        # Validate zones
        for zone in self.Zones:
            zone._write_plt_pre(self)
    
    def _write_plt_head(self, fid):
        """Write header section to file."""
        # I. HEADER SECTION
        # i. Magic number, Version number
        fid.write(b'#!TDV112')  # 8 bytes
        fid.write(struct.pack('<i', 1))  # Integer value of 1 (byte order)
        
        # iii. Title and variable names
        fid.write(struct.pack('<i', self.FileType))  # FileType
        fid.write(struct.pack('<{}i'.format(len(s2i(self.Title))), *s2i(self.Title)))  # Title
        fid.write(struct.pack('<i', len(self.Variables)))  # Number of variables
        for var in self.Variables:
            fid.write(struct.pack('<{}i'.format(len(s2i(var))), *s2i(var)))  # Variable names
        
        if self.Echo_Mode[2]:
            buf = '     VAR:'
            for var in self.Variables:
                buf += f' <{var}>'
            self.last_log['Echo_Text'].append(buf)
            print(buf)
        
        # iv. Zones
        for zone in self.Zones:
            zone._write_plt_head(fid)
        
        # ix. Dataset Auxiliary data
        if self.Auxiliary:
            for au in self.Auxiliary:
                fid.write(struct.pack('<f', 799.0))  # DataSetAux Marker
                fid.write(struct.pack('<{}i'.format(len(s2i(au[0]))), *s2i(au[0])))  # Name
                fid.write(struct.pack('<i', 0))  # Value Format
                fid.write(struct.pack('<{}i'.format(len(s2i(au[1]))), *s2i(au[1])))  # Value
    
    def _write_plt_data(self, fid):
        """Write data section to file."""
        # II. DATA SECTION
        for zone_n, zone in enumerate(self.Zones, 1):
            if zone.Echo_Mode[0]:
                buf = f'--   write zone {zone_n}: {zone.ZoneName}   --'
                self.last_log['Echo_Text'].append(buf)
                print(buf)
            
            zone._write_plt_data(fid, self)
            
            self.last_log['Echo_Text'].append('#ZONE#')
            
            if zone.Echo_Mode[1]:
                buf = f'--   write zone {zone_n}: {zone.ZoneName}   --'
                self.last_log['Echo_Text'].append(buf)
                print(buf)


class TEC_ZONE(TEC_ZONE_BASE):
    """
    Main TEC_ZONE class for managing data zones in Tecplot files.
    
    This class manages individual data zones within a Tecplot file, supporting
    various data types, dimensions (1D, 2D, 3D), and data
    subsampling options (Skip, Begin, EEnd).
    
    Echo_Mode flags: [zone_head, zone_end, variable, max_real, max_org, 
                      skip, begin_end, stdid_soltime, size]
    
    Examples:
        >>> import numpy as np
        >>> from tecplot import TEC_ZONE
        >>>
        >>> # Create a zone with 2D data
        >>> zone = TEC_ZONE()
        >>> zone.ZoneName = 'FlowField'
        >>> x = np.linspace(0, 1, 50).reshape(50, 1)
        >>> y = np.linspace(0, 1, 50).reshape(1, 50)
        >>> X = x + np.zeros_like(y)
        >>> Y = y + np.zeros_like(x)
        >>> U = np.sin(2 * np.pi * X)
        >>> V = np.cos(2 * np.pi * Y)
        >>>
        >>> zone.Data = [X, Y, U, V]
        >>>
        >>> # Use Skip and Begin for subsampling
        >>> zone.Skip = [2, 2, 1]  # Skip every other point
        >>> zone.Begin = [5, 5, 0]  # Skip first 5 points
    """
    
    def __init__(self, *args):
        super().__init__()
        self.Data: List[np.ndarray] = []
        # Initialize _echo_mode before setting Echo_Mode property
        self._echo_mode: List[bool] = [True, False, False, True, False, False, False, False, False]
        # Now set the Echo_Mode property (will use the setter)
        self.Echo_Mode: Union[str, List[bool]] = 'brief'
        
        if len(args) == 0:
            # Default values from base class
            pass
        elif len(args) == 1:
            n = args[0]
            if isinstance(n, int) and n == n:
                pass  # Would create array in MATLAB, but Python uses list
            else:
                raise RuntimeError('input of TEC_ZONE constructor must be a positive integer')
        else:
            raise RuntimeError('TEC_ZONE constructor too many input arguments')
    
    @property
    def Echo_Mode(self) -> List[bool]:
        return self._echo_mode
    
    @Echo_Mode.setter
    def Echo_Mode(self, mode: Union[str, List[bool]]):
        if isinstance(mode, list):
            if all(isinstance(x, bool) for x in mode):
                if len(mode) == 9:
                    self._echo_mode = mode
                else:
                    raise RuntimeError('echo_mode code size wrong')
            else:
                raise RuntimeError(f'echo_mode type wrong ({type(mode)})')
        elif isinstance(mode, str):
            if mode == 'brief':
                self._echo_mode = [True, False, False, True, False, False, False, False, False]
            elif mode == 'full':
                self._echo_mode = [True] * 9
            elif mode == 'simple':
                self._echo_mode = [True, False, False, False, False, False, False, False, False]
            elif mode == 'none':
                self._echo_mode = [False] * 9
            elif mode == 'leave':
                pass  # Don't change
            else:
                raise RuntimeError(f'echo_mode code string wrong ("{mode}")')
        else:
            raise RuntimeError(f'echo_mode type wrong ({type(mode)})')
    
    def gather_real_size(self, n: int = 1) -> Tuple[List[int], int, bool, bool]:
        """
        Calculate real size of zone data.
        
        Args:
            n: Data index
            
        Returns:
            Tuple of (Real_Max, Real_Dim, noskip, noexc)
        """
        if len(self.Data) < n:
            raise RuntimeError(f'numel(self.Data):{len(self.Data)} < n:{n}')
        
        # Validate Skip, Begin, EEnd
        for i, (val, name) in enumerate(zip([self.Skip, self.Begin, self.EEnd], 
                                            ['Skip', 'Begin', 'EEnd'])):
            for v in val:
                if v < 0 or v != int(v):
                    raise RuntimeError(f'the {name} of zone is not positive integer: {val}')
        
        Real_Max = real_ijk(list(self.Data[n-1].shape), self.Skip, self.Begin, self.EEnd)
        
        if any(m <= 0 for m in Real_Max):
            raise RuntimeError(
                f'sum of Begin and EEnd is not smaller than Max: '
                f'{self.Begin}+{self.EEnd}>={list(self.Data[n-1].shape)}'
            )
        
        Real_Dim = max(i for i, m in enumerate(Real_Max, 1) if m != 1)
        noskip = self.Skip == [1, 1, 1]
        noexc = self.Begin == [0, 0, 0] and self.EEnd == [0, 0, 0]
        
        return Real_Max, Real_Dim, noskip, noexc
    
    def _write_plt_pre(self, file: TEC_FILE):
        """Pre-write validation and preparation."""
        if not self.Data:
            raise RuntimeError(
                f'FILE[{file.FileName}]--ZONE[{self.ZoneName}]: TEC_ZONE.Data is empty'
            )
        
        if len(self.Data) != len(file.Variables):
            raise RuntimeError(
                f'FILE[{file.FileName}]--ZONE[{self.ZoneName}]: '
                f'TEC_ZONE.Data is not correspond to TEC_FILE.Variables'
            )
        
        data_size = self.Data[0].shape
        
        for kk, da in enumerate(self.Data):
            if da.size == 0:
                raise RuntimeError(
                    f'FILE[{file.FileName}]--ZONE[{self.ZoneName}]: '
                    f'data[{file.Variables[kk]}] is empty'
                )
            
            if da.ndim > 3:
                raise RuntimeError(
                    f'FILE[{file.FileName}]--ZONE[{self.ZoneName}]: '
                    f'the dimension of data[{file.Variables[kk]}] is bigger than 3'
                )
            
            # Check for NaN or inf
            if np.any(np.isinf(da)) or np.any(np.isnan(da)):
                print(
                    f'Warning: FILE[{file.FileName}]--ZONE[{self.ZoneName}]: '
                    f'data[{file.Variables[kk]}] has nan or inf'
                )
            
            if da.shape != data_size:
                raise RuntimeError(
                    f'FILE[{file.FileName}]--ZONE[{self.ZoneName}]: data size is not equal'
                )
    
    def _write_plt_head(self, fid):
        """Write zone header to file."""
        # iv. Zones
        fid.write(struct.pack('<f', 299.0))  # Zone marker
        fid.write(struct.pack('<{}i'.format(len(s2i(self.ZoneName))), *s2i(self.ZoneName)))  # Zone name
        fid.write(struct.pack('<i', -1))  # ParentZone
        fid.write(struct.pack('<i', self.StrandId))  # StrandID
        fid.write(struct.pack('<d', self.SolutionTime))  # Solution time
        fid.write(struct.pack('<i', -1))  # Not used. Set to -1
        fid.write(struct.pack('<i', 0))  # ZoneType 0=ORDERED
        fid.write(struct.pack('<i', 0))  # Specify Var Location
        fid.write(struct.pack('<i', 0))  # Face neighbors supplied? 0=FALSE
        fid.write(struct.pack('<i', 0))  # Number of miscellaneous face neighbor connections
        
        # IMax, JMax, KMax
        real_dims = real_ijk(list(self.Data[0].shape), self.Skip, self.Begin, self.EEnd)
        fid.write(struct.pack('<3i', *real_dims))
        
        # Auxiliary data
        if self.Auxiliary:
            for au in self.Auxiliary:
                fid.write(struct.pack('<i', 1))  # Auxiliary name/value pair to follow
                fid.write(struct.pack('<{}i'.format(len(s2i(au[0]))), *s2i(au[0])))  # name
                fid.write(struct.pack('<i', 0))  # Value Format
                fid.write(struct.pack('<{}i'.format(len(s2i(au[1]))), *s2i(au[1])))  # value
        
        fid.write(struct.pack('<i', 0))  # No more Auxiliary name/value pairs
    
    def _write_plt_data(self, fid, file: TEC_FILE):
        """Write zone data to file."""
        pos_b = fid.tell()
        fid.write(struct.pack('<f', 299.0))  # Zone marker
        
        # Write variable data formats
        for val_n, da in enumerate(self.Data):
            try:
                ty, _ = gettype(da)
                fid.write(struct.pack('<i', ty))
            except Exception as e:
                raise RuntimeError(
                    f'zone [{self.ZoneName}] var[{file.Variables[val_n]}]: {str(e)}'
                )
        
        fid.write(struct.pack('<i', 0))  # Has passive variables: 0 = no
        fid.write(struct.pack('<i', 0))  # Has variable sharing: 0 = no
        fid.write(struct.pack('<i', -1))  # Zone number to share connectivity with (-1 = no sharing)
        
        # Make buffers and write min/max values
        buf = [makebuf(da, self.Skip, self.Begin, self.EEnd) for da in self.Data]
        
        for kk, val in enumerate(buf):
            min_buf = float(np.min(val))
            max_buf = float(np.max(val))
            fid.write(struct.pack('<d', min_buf))  # Min value
            fid.write(struct.pack('<d', max_buf))  # Max value
        
        # Echo information
        if self.Echo_Mode[3]:
            Real_Max, Real_Dim, _, _ = self.gather_real_size()
            print(_format_dims('Dim', 'Real_Max', Real_Max, Real_Dim))
        
        if self.Echo_Mode[4]:
            orig_dims = list(self.Data[0].shape)
            orig_dim = max(i for i, m in enumerate(orig_dims, 1) if m != 1)
            print(_format_dims('Org_Dim', 'Org_Max', orig_dims, orig_dim))
        
        if self.Echo_Mode[5]:
            print(_format_vector('Skip', self.Skip))
        
        if self.Echo_Mode[6]:
            print(_format_vector('Begin', self.Begin) + '   ' + _format_vector('EEnd', self.EEnd))
        
        if self.Echo_Mode[7] and self.StrandId != -1:
            echobuf = f'     StrandId = {self.StrandId} SolutionTime = {self.SolutionTime:e}'
            print(echobuf)
        
        # Write variable data
        if self.Echo_Mode[2]:
            print('     write variables:', end='')
        
        for val_n, val in enumerate(buf):
            if self.Echo_Mode[2]:
                print(f' <{file.Variables[val_n]}>', end='')
            
            # Write data based on type
            ty, _ = gettype(self.Data[val_n])
            if ty == 1:  # float32
                fid.write(val.astype(np.float32).tobytes())
            elif ty == 2:  # float64
                fid.write(val.astype(np.float64).tobytes())
            elif ty == 3:  # int32
                fid.write(val.astype(np.int32).tobytes())
            elif ty == 4:  # int16
                fid.write(val.astype(np.int16).tobytes())
            elif ty == 5:  # int8
                fid.write(val.astype(np.int8).tobytes())
        
        if self.Echo_Mode[2]:
            print()
        
        pos_e = fid.tell()
        if self.Echo_Mode[8]:
            size_mb = (pos_e - pos_b) / 1024 / 1024
            echobuf = f'     zone size: {size_mb:.1f} MB'
            print(echobuf)
