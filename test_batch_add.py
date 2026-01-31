"""
Batch add demonstration for tecplot.py

This script demonstrates the add_datas() method (DataFrame-like API)
"""

import numpy as np
from tecplot import TEC_FILE

def test_batch_add():
    """Test batch add_datas API."""
    print("=" * 60)
    print("BATCH ADD (add_datas) DEMONSTRATION")
    print("=" * 60)
    
    # Create TEC_FILE object
    tec_file = TEC_FILE()
    tec_file.FileName = 'batch_add_demo'
    tec_file.Variables = ['X', 'Y', 'Z', 'u', 'v', 'w']
    
    # Generate test data
    Nr, Nt, Nz = 50, 30, 20
    
    r = np.linspace(0, 1, Nr)
    t = np.linspace(0, 2 * np.pi, Nt)
    z = np.linspace(0, 2 * np.pi, Nz)
    
    Z, T, R = np.meshgrid(z, t, r, indexing='ij')
    
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z_grid = Z
    
    # Create velocity field
    u = np.sin(2 * np.pi * T)
    v = np.cos(2 * np.pi * T)
    w = np.ones_like(T)
    
    print("\n1. Batch add with automatic variable names:")
    print("   (Uses Variables list in order: 'X', 'Y', 'Z', 'u', 'v', 'w')")
    print("\n   tec_file.add_datas([X, Y, Z_grid, u, v, w])")
    tec_file.add_datas([X, Y, Z_grid, u, v, w])
    
    # Create a second file with custom names
    tec_file2 = TEC_FILE()
    tec_file2.FileName = 'batch_add_custom'
    tec_file2.Variables = ['Xcoord', 'Ycoord', 'Zcoord', 'U', 'V', 'W']
    
    print("\n2. Batch add with custom variable names:")
    print("   (Uses provided name list)")
    print("\n   custom_names = ['Xcoord', 'Ycoord', 'Zcoord', 'U', 'V', 'W']")
    print("   tec_file.add_datas([X, Y, Z_grid, u, v, w], custom_names)")
    tec_file2.add_datas([X, Y, Z_grid, u, v, w], ['Xcoord', 'Ycoord', 'Zcoord', 'U', 'V', 'W'])
    
    # Write both files
    print("\nWriting files...")
    tec_file.write_plt()
    tec_file2.write_plt()
    
    print(f"\nSuccessfully created:")
    print(f"  - {tec_file.FileName}.plt")
    print(f"  - {tec_file2.FileName}.plt")
    print(f"\nData points per file: {X.size}")
    print(f"Variables: {len(tec_file.Variables)}")
    print(f"Data arrays: {len(tec_file.Zones[0].Data)}")

if __name__ == '__main__':
    test_batch_add()
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
