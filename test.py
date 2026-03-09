import numpy as np
import os
from tecplot import TEC_FILE, TEC_ZONE


def test_simple_write():
    print("=" * 60)
    print("Test 1: Simple PLT file write")
    print("=" * 60)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 't_all'
    tec_file.Variables = ['x', 'y', 'z', 'p1', 'p2', 'p3', 'p4']
    tec_file.Zones = [TEC_ZONE()]
    
    nx, ny = 100, 100
    x = np.linspace(0, 10, nx).reshape(nx, 1)
    y = np.linspace(0, 10, ny).reshape(1, ny)
    X = x + np.zeros_like(y)
    Y = y + np.zeros_like(x)
    Z = np.sin(X) * np.cos(Y)
    
    p1 = Z * 1000
    p2 = Z * 2000
    p3 = Z * 3000
    p4 = Z * 4000
    
    tec_file.Zones[0].Data = [X, Y, Z, p1, p2, p3, p4]
    tec_file.Zones[0].ZoneName = 'Zone1'
    tec_file.write_plt()
    print(f"\nSuccessfully created: {tec_file.FileName}.plt")
    return True


def test_multiple_zones():
    print("\n" + "=" * 60)
    print("Test 2: Multiple zones")
    print("=" * 60)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'multi_zone'
    tec_file.Variables = ['x', 'y', 'pressure']
    tec_file.Zones = [TEC_ZONE() for _ in range(3)]
    
    for i, zone in enumerate(tec_file.Zones):
        n = 50 + i * 10
        x = np.linspace(i * 10, (i + 1) * 10, n).reshape(n, 1)
        y = np.linspace(0, 10, n).reshape(1, n)
        X = x + np.zeros_like(y)
        Y = y + np.zeros_like(x)
        P = (X + Y) * (i + 1)
        
        zone.Data = [X, Y, P]
        zone.ZoneName = f'Zone_{i}'
        zone.SolutionTime = i * 0.5
    
    tec_file.write_plt()
    print(f"\nSuccessfully created: {tec_file.FileName}.plt with {len(tec_file.Zones)} zones")
    return True


def test_skip_and_begin():
    print("\n" + "=" * 60)
    print("Test 3: Skip and Begin parameters")
    print("=" * 60)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'skip_test'
    tec_file.Variables = ['x', 'y', 'z']
    tec_file.Zones = [TEC_ZONE()]
    zone = tec_file.Zones[0]
    
    n = 200
    x = np.linspace(0, 10, n).reshape(n, 1)
    y = np.linspace(0, 10, n).reshape(1, n)
    X = x + np.zeros_like(y)
    Y = y + np.zeros_like(x)
    Z = np.sin(X) * np.cos(Y)
    
    zone.Data = [X, Y, Z]
    zone.ZoneName = 'HighResZone'
    zone.Skip = [2, 2, 1]
    zone.Begin = [10, 10, 0]
    zone.EEnd = [10, 10, 0]
    
    tec_file.write_plt()
    print(f"\nSuccessfully created: {tec_file.FileName}.plt with Skip={zone.Skip}")
    return True


def test_3d_data():
    print("\n" + "=" * 60)
    print("Test 4: 3D data")
    print("=" * 60)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'test_3d'
    tec_file.Variables = ['x', 'y', 'z', 'temperature']
    tec_file.Zones = [TEC_ZONE()]
    zone = tec_file.Zones[0]
    
    nx, ny, nz = 30, 30, 30
    x = np.linspace(0, 10, nx).reshape(nx, 1, 1)
    y = np.linspace(0, 10, ny).reshape(1, ny, 1)
    z = np.linspace(0, 10, nz).reshape(1, 1, nz)
    X = x + np.zeros_like(y) + np.zeros_like(z)
    Y = y + np.zeros_like(x) + np.zeros_like(z)
    Z = z + np.zeros_like(x) + np.zeros_like(y)
    T = np.sin(X) * np.cos(Y) * np.sin(Z) + 300
    
    zone.Data = [X, Y, Z, T]
    zone.ZoneName = 'Volume3D'
    tec_file.write_plt()
    print(f"\nSuccessfully created: {tec_file.FileName}.plt with 3D data")
    return True


def test_different_data_types():
    print("\n" + "=" * 60)
    print("Test 5: Different data types")
    print("=" * 60)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'data_types'
    tec_file.Variables = ['x', 'y', 'id_int32', 'id_int16', 'id_int8']
    tec_file.Zones = [TEC_ZONE()]
    zone = tec_file.Zones[0]
    
    n = 50
    x = np.linspace(0, 10, n, dtype=np.float64).reshape(n, 1)
    y = np.linspace(0, 10, n, dtype=np.float32).reshape(1, n)
    X = x + np.zeros_like(y)
    Y = y + np.zeros_like(X)
    id32 = np.arange(n * n, dtype=np.int32).reshape(n, n)
    id16 = np.arange(n * n, dtype=np.int16).reshape(n, n)
    total = n * n
    values = np.arange(total)
    id8 = (values % 127).astype(np.int8).reshape(n, n)
    
    zone.Data = [X, Y, id32, id16, id8]
    zone.ZoneName = 'MixedTypes'
    tec_file.write_plt()
    print(f"\nSuccessfully created: {tec_file.FileName}.plt with mixed data types")
    return True


def test_echo_modes():
    print("\n" + "=" * 60)
    print("Test 6: Echo modes")
    print("=" * 60)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'echo_full'
    tec_file.Variables = ['x', 'y', 'value']
    tec_file.set_echo_mode('full', 'full')
    tec_file.Zones = [TEC_ZONE()]
    
    n = 20
    x = np.linspace(0, 1, n).reshape(n, 1)
    y = np.linspace(0, 1, n).reshape(1, n)
    X = x + np.zeros_like(y)
    Y = y + np.zeros_like(x)
    V = X * Y
    
    tec_file.Zones[0].Data = [X, Y, V]
    tec_file.Zones[0].ZoneName = 'TestZone'
    tec_file.write_plt()
    print("\n--- Full echo mode completed ---")
    
    tec_file2 = TEC_FILE()
    tec_file2.FileName = 'echo_none'
    tec_file2.Variables = ['x', 'y', 'value']
    tec_file2.set_echo_mode('none', 'none')
    tec_file2.Zones = [TEC_ZONE()]
    tec_file2.Zones[0].Data = [X, Y, V]
    tec_file2.Zones[0].ZoneName = 'TestZone'
    tec_file2.write_plt()
    print("(No output from 'none' echo mode - this is expected)")
    print("\n--- None echo mode completed ---")
    return True


def test_auto_infer():
    print("\n" + "=" * 60)
    print("Test 7: Auto-infer variable names")
    print("=" * 60)
    
    X = np.linspace(0, 10, 50).reshape(50, 1)
    Y = np.linspace(0, 10, 50).reshape(1, 50)
    X_grid = X + np.zeros_like(Y)
    Y_grid = Y + np.zeros_like(X)
    u = np.sin(X_grid) * np.cos(Y_grid)
    v = np.cos(X_grid) * np.sin(Y_grid)
    w = np.ones_like(X_grid)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'auto_infer_test'
    tec_file.add_datas([X_grid, Y_grid, u, v, w])
    
    assert tec_file.Variables == ['X_grid', 'Y_grid', 'u', 'v', 'w'], \
        f"Expected ['X_grid', 'Y_grid', 'u', 'v', 'w'], got {tec_file.Variables}"
    
    tec_file.write_plt()
    print(f"Variables: {tec_file.Variables}")
    print(f"\nSuccessfully created: {tec_file.FileName}.plt")
    return True


def test_auto_infer_explicit():
    print("\n" + "=" * 60)
    print("Test 8: Auto-infer with explicit names")
    print("=" * 60)
    
    X = np.linspace(0, 10, 50).reshape(50, 1)
    Y = np.linspace(0, 10, 50).reshape(1, 50)
    X_grid = X + np.zeros_like(Y)
    Y_grid = Y + np.zeros_like(X)
    u = np.sin(X_grid)
    v = np.cos(Y_grid)
    w = np.ones_like(X_grid)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'explicit_names_test'
    tec_file.add_datas([X_grid, Y_grid, u, v, w], 
                       ['Xcoord', 'Ycoord', 'Uvel', 'Vvel', 'Wcomp'])
    
    assert tec_file.Variables == ['Xcoord', 'Ycoord', 'Uvel', 'Vvel', 'Wcomp'], \
        f"Expected explicit names, got {tec_file.Variables}"
    
    tec_file.write_plt()
    print(f"Variables: {tec_file.Variables}")
    print(f"\nSuccessfully created: {tec_file.FileName}.plt")
    return True


def test_auto_infer_user_example():
    print("\n" + "=" * 60)
    print("Test 9: User's original example")
    print("=" * 60)
    
    Nr, Nt, Nz = 20, 15, 10
    r = np.linspace(0, 1, Nr)
    t = np.linspace(0, 2 * np.pi, Nt)
    z = np.linspace(0, 2 * np.pi, Nz)
    
    Z, T, R = np.meshgrid(z, t, r, indexing='ij')
    
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z_coord = Z.copy()
    
    u = np.sin(2 * np.pi * T)
    v = np.cos(2 * np.pi * T)
    w = np.ones_like(T)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'user_example_auto'
    tec_file.add_datas([X, Y, Z_coord, u, v, w])
    
    assert tec_file.Variables == ['X', 'Y', 'Z_coord', 'u', 'v', 'w'], \
        f"Expected ['X', 'Y', 'Z_coord', 'u', 'v', 'w'], got {tec_file.Variables}"
    
    tec_file.write_plt()
    print(f"Variables: {tec_file.Variables}")
    print(f"\nSuccessfully created: {tec_file.FileName}.plt")
    
    tec_file2 = TEC_FILE()
    tec_file2.FileName = 'user_example_explicit'
    tec_file2.add_datas([X, Y, Z_coord, u, v, w], 
                        ['Xcoord', 'Ycoord', 'Zcoord', 'U', 'V', 'W'])
    
    assert tec_file2.Variables == ['Xcoord', 'Ycoord', 'Zcoord', 'U', 'V', 'W'], \
        f"Expected ['Xcoord', 'Ycoord', 'Zcoord', 'U', 'V', 'W'], got {tec_file2.Variables}"
    
    tec_file2.write_plt()
    print(f"Variables (explicit): {tec_file2.Variables}")
    print(f"\nSuccessfully created: {tec_file2.FileName}.plt")
    return True


def test_add_data_single():
    print("\n" + "=" * 60)
    print("Test 10: AddData single")
    print("=" * 60)
    
    n = 50
    x = np.linspace(0, 10, n).reshape(n, 1)
    y = np.linspace(0, 10, n).reshape(1, n)
    X = x + np.zeros_like(y)
    Y = y + np.zeros_like(X)
    Z = np.sin(X) * np.cos(Y)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'add_data_single'
    tec_file.Variables = ['X', 'Y', 'Z']
    tec_file.AddData(X, name='X').AddData(Y, name='Y').AddData(Z, name='Z')
    
    assert len(tec_file.Zones[0].Data) == 3
    tec_file.write_plt()
    print(f"\nSuccessfully created: {tec_file.FileName}.plt")
    return True


def cleanup():
    files = [
        't_all.plt', 'multi_zone.plt', 'skip_test.plt', 'test_3d.plt',
        'data_types.plt', 'echo_full.plt', 'echo_none.plt',
        'auto_infer_test.plt', 'explicit_names_test.plt',
        'user_example_auto.plt', 'user_example_explicit.plt',
        'add_data_single.plt',
    ]
    for f in files:
        if os.path.exists(f):
            os.remove(f)


def main():
    print("\n" + "=" * 60)
    print("TECPLOT TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_simple_write,
        test_multiple_zones,
        test_skip_and_begin,
        test_3d_data,
        test_different_data_types,
        test_echo_modes,
        test_auto_infer,
        test_auto_infer_explicit,
        test_auto_infer_user_example,
        test_add_data_single,
    ]
    
    passed = failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n\nFAILED: {test.__name__}")
            print(f"Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    cleanup()
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
