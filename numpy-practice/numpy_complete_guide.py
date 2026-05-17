"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          COMPLETE NUMPY MASTERY GUIDE — From Zero to Expert                ║
║   Theory + Internals + Worked Examples + 120+ Exercises with Solutions     ║
╚══════════════════════════════════════════════════════════════════════════════╝

MODULES COVERED:
  M1  — What is NumPy & ndarray internals
  M2  — Array creation (14 methods)
  M3  — Data types (dtypes)
  M4  — Indexing & slicing (basic + advanced + fancy)
  M5  — Reshaping, stacking, splitting
  M6  — Universal functions (ufuncs) & broadcasting
  M7  — Aggregations & statistics
  M8  — Linear algebra
  M9  — Random number generation
  M10 — Boolean masking & structured arrays
  M11 — Performance: vectorisation vs loops
  M12 — Real-world projects (ML, image, finance)
  EX  — 120 exercises with solutions (grouped by module)
"""

import numpy as np
import time

SEP  = "\n" + "═"*70
SEP2 = "\n" + "─"*50

# ════════════════════════════════════════════════════════════════════════════
# MODULE 1 — WHAT IS NUMPY & NDARRAY INTERNALS
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 1 — WHAT IS NUMPY & NDARRAY INTERNALS")
print(SEP)

print("""
THEORY
──────
NumPy (Numerical Python) is the foundation of the entire scientific Python
ecosystem (Pandas, SciPy, Scikit-learn, TensorFlow all use it internally).

WHY IS NUMPY FAST?
  • ndarray stores data in a CONTIGUOUS block of memory (like C arrays).
  • Operations are executed in pre-compiled C/Fortran — no Python overhead.
  • SIMD (Single Instruction Multiple Data) CPU instructions operate on
    multiple elements per clock cycle.
  • Zero-copy slicing: slices share memory — no copying.

NDARRAY ANATOMY — every array has these attributes:
  ┌─────────────────────────────────────────────────────┐
  │  .shape    → tuple of dimensions   e.g. (3, 4)     │
  │  .ndim     → number of dimensions  e.g. 2           │
  │  .size     → total element count   e.g. 12          │
  │  .dtype    → element type          e.g. float64     │
  │  .itemsize → bytes per element     e.g. 8           │
  │  .nbytes   → total memory bytes    e.g. 96          │
  │  .strides  → bytes to step in each dimension        │
  └─────────────────────────────────────────────────────┘

STRIDES (KEY CONCEPT FOR EXPERTS):
  For a (3,4) float64 array:
    strides = (32, 8)  ← move 32 bytes to next row, 8 bytes to next column
  This lets NumPy support C-order (row-major) and F-order (column-major)
  without copying data — just different strides!
""")

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9,10,11,12]])

print("Example array a:\n", a)
print(f"\n  shape    = {a.shape}")
print(f"  ndim     = {a.ndim}")
print(f"  size     = {a.size}")
print(f"  dtype    = {a.dtype}")
print(f"  itemsize = {a.itemsize} bytes")
print(f"  nbytes   = {a.nbytes} bytes")
print(f"  strides  = {a.strides}  ← (row_step, col_step) in bytes")

# Memory sharing demo
print(SEP2)
print("MEMORY SHARING — slices share memory (zero copy):")
original = np.arange(10)
view     = original[2:7]          # view — shares memory
copy_arr = original[2:7].copy()   # explicit copy

view[0] = 999
print(f"  original after modifying view: {original}")
print(f"  view:                          {view}")
copy_arr[0] = -1
print(f"  original after modifying copy: {original}  ← unchanged!")
print(f"  np.shares_memory(original, view) = {np.shares_memory(original, view)}")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 2 — ARRAY CREATION (14 METHODS)
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 2 — ARRAY CREATION (14 METHODS)")
print(SEP)

print("""
THEORY
──────
NumPy provides a rich set of constructors. Choosing the right one matters for
both readability and performance. Key rule: prefer built-in constructors over
Python loops.
""")

# 1. From Python sequences
print("① From list / nested list:")
a1 = np.array([1, 2, 3])
a2 = np.array([[1, 2], [3, 4]])
print(f"  1D: {a1}  shape={a1.shape}")
print(f"  2D:\n{a2}  shape={a2.shape}")

# 2. zeros / ones / full
print("\n② zeros / ones / full:")
print(f"  zeros(3,4):\n{np.zeros((3,4))}")
print(f"  ones(2,3):\n{np.ones((2,3))}")
print(f"  full(2,3, fill=7):\n{np.full((2,3), 7)}")

# 3. eye / identity / diag
print("\n③ eye / identity / diag:")
print(f"  eye(3):\n{np.eye(3)}")
print(f"  diag([1,2,3]):\n{np.diag([1,2,3])}")
print(f"  diag(matrix) → extract diagonal: {np.diag(np.eye(3))}")

# 4. arange
print("\n④ arange (like Python range, but returns ndarray):")
print(f"  arange(10):         {np.arange(10)}")
print(f"  arange(2,20,3):     {np.arange(2,20,3)}")
print(f"  arange(0,1,0.25):   {np.arange(0,1,0.25)}")

# 5. linspace / logspace / geomspace
print("\n⑤ linspace / logspace / geomspace:")
print(f"  linspace(0,1,5):    {np.linspace(0,1,5)}")
print(f"  logspace(0,2,5):    {np.logspace(0,2,5).round(2)}")
print(f"  geomspace(1,100,5): {np.geomspace(1,100,5).round(2)}")

# 6. empty
print("\n⑥ empty (uninitialized — fastest, but values are garbage):")
e = np.empty((2,3))
print(f"  empty(2,3): shape={e.shape}, dtype={e.dtype}  ← values undefined!")

# 7. like variants
print("\n⑦ *_like — match shape of existing array:")
ref = np.array([[1,2,3],[4,5,6]])
print(f"  zeros_like(ref):\n{np.zeros_like(ref)}")
print(f"  ones_like(ref):\n{np.ones_like(ref)}")

# 8. from function
print("\n⑧ fromfunction — build from index formula:")
f = np.fromfunction(lambda i,j: i*10 + j, (4,5), dtype=int)
print(f"  fromfunction(i*10+j, (4,5)):\n{f}")

# 9. tile / repeat
print("\n⑨ tile / repeat:")
base = np.array([1,2,3])
print(f"  tile([1,2,3], 3):       {np.tile(base, 3)}")
print(f"  tile([1,2,3], (2,3)):   \n{np.tile(base,(2,3))}")
print(f"  repeat([1,2,3], 2):     {np.repeat(base, 2)}")
print(f"  repeat([1,2,3], [1,2,3]):{np.repeat(base,[1,2,3])}")

# 10. meshgrid
print("\n⑩ meshgrid — coordinate grids (used in plotting & broadcasting):")
x_mg = np.array([1,2,3]); y_mg = np.array([10,20])
XX, YY = np.meshgrid(x_mg, y_mg)
print(f"  XX:\n{XX}\n  YY:\n{YY}")
print(f"  XX+YY:\n{XX+YY}  ← all (x,y) sums at once")

# 11. load from file
print("\n⑪ Load from file (just demonstrating the API):")
print("  np.load('file.npy')          — binary format")
print("  np.loadtxt('file.csv', delimiter=',')   — text CSV")
print("  np.genfromtxt('f.csv', names=True)      — with header")

# 12. random (briefly — full coverage in M9)
print("\n⑫ Quick random arrays (detailed in M9):")
rng = np.random.default_rng(42)
print(f"  random(3,3):\n{rng.random((3,3)).round(3)}")

# 13. Structured arrays
print("\n⑬ Structured arrays (record arrays):")
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('score', 'f8')])
students = np.array([('Alice',22,89.5), ('Bob',25,76.0), ('Carol',21,92.1)], dtype=dt)
print(f"  students['name']:  {students['name']}")
print(f"  students['score']: {students['score']}")
print(f"  students[0]:       {students[0]}")

# 14. From existing data with different dtype
print("\n⑭ asarray / astype (view vs copy with dtype change):")
lst = [1.5, 2.7, 3.9]
a_float = np.asarray(lst)           # no copy if already array
a_int   = np.asarray(lst, dtype=int)# converts
print(f"  asarray(lst):           {a_float}  dtype={a_float.dtype}")
print(f"  asarray(lst, int):      {a_int}  dtype={a_int.dtype}")
print(f"  a.astype(np.float32):   {a_int.astype(np.float32)}")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 3 — DATA TYPES (DTYPES)
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 3 — DATA TYPES (dtypes)")
print(SEP)

print("""
THEORY
──────
NumPy's dtype system maps directly to C/hardware types, giving precise control
over memory layout and computation precision.

DTYPE TABLE:
  ┌─────────────┬──────────┬──────────────────────────────────────┐
  │ dtype       │ bytes    │ range / notes                        │
  ├─────────────┼──────────┼──────────────────────────────────────┤
  │ bool_       │ 1        │ True / False                         │
  │ int8        │ 1        │ -128 to 127                          │
  │ int16       │ 2        │ -32,768 to 32,767                    │
  │ int32       │ 4        │ -2.1B to 2.1B                        │
  │ int64       │ 8        │ -9.2e18 to 9.2e18  (default int)     │
  │ uint8       │ 1        │ 0 to 255  (pixels!)                  │
  │ uint16      │ 2        │ 0 to 65,535                          │
  │ float16     │ 2        │ ~3 sig digits  (GPU/ML)              │
  │ float32     │ 4        │ ~7 sig digits  (most ML)             │
  │ float64     │ 8        │ ~15 sig digits (default float)       │
  │ complex64   │ 8        │ 2×float32                            │
  │ complex128  │ 16       │ 2×float64                            │
  │ str_ / U    │ variable │ fixed-width Unicode strings           │
  │ object_     │ 8        │ pointer to Python object              │
  └─────────────┴──────────┴──────────────────────────────────────┘

RULES:
  • Default int → int64 (or int32 on 32-bit OS)
  • Default float → float64
  • ML training: float32 (half the memory of float64)
  • Image pixels: uint8 (0–255)
  • OVERFLOW is silent (no error!): int8(200) wraps around!
""")

# dtype examples
print("dtype demonstrations:")
print(f"  np.array([1,2,3]).dtype          = {np.array([1,2,3]).dtype}")
print(f"  np.array([1.,2.,3.]).dtype       = {np.array([1.,2.,3.]).dtype}")
print(f"  np.array([True,False]).dtype     = {np.array([True,False]).dtype}")
print(f"  np.array(['a','bb']).dtype       = {np.array(['a','bb']).dtype}")

print("\nOverflow warning (SILENT!):")
a_int8 = np.array([127], dtype=np.int8); a_int8 = (a_int8 + np.int8(1)).view(np.int8)
print(f"  int8(200) = {a_int8}   ← WRAPS AROUND (200-256 = -56)!")
a_uint8 = np.array([200], dtype=np.uint8)
print(f"  uint8(200) = {a_uint8}  ← correct for pixels")

print("\nMemory savings — float32 vs float64:")
big_f64 = np.ones(1_000_000, dtype=np.float64)
big_f32 = np.ones(1_000_000, dtype=np.float32)
print(f"  float64 1M elements: {big_f64.nbytes/1e6:.1f} MB")
print(f"  float32 1M elements: {big_f32.nbytes/1e6:.1f} MB  ← half the memory!")

print("\nType checking and info:")
print(f"  np.iinfo(np.int16): min={np.iinfo(np.int16).min}, max={np.iinfo(np.int16).max}")
print(f"  np.finfo(np.float32): eps={np.finfo(np.float32).eps:.2e}, max={np.finfo(np.float32).max:.2e}")

print("\nSafe casting rules:")
x_i = np.array([1,2,3], dtype=np.int32)
x_f = np.array([1.,2.,3.], dtype=np.float64)
result = x_i + x_f                 # int32 + float64 → float64
print(f"  int32 + float64 → {result.dtype}  (NumPy upcasts safely)")

print("\nSpecial float values:")
inf    = np.float64('inf')
neg_inf= np.float64('-inf')
nan    = np.float64('nan')
print(f"  inf:  {inf}    isinf={np.isinf(inf)}")
print(f"  nan:  {nan}    isnan={np.isnan(nan)}")
print(f"  nan == nan: {nan==nan}  ← use np.isnan() instead!")
print(f"  inf > 1e300: {inf > 1e300}")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 4 — INDEXING & SLICING
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 4 — INDEXING & SLICING")
print(SEP)

print("""
THEORY
──────
NumPy supports three indexing styles:
  1. Basic indexing     → returns a VIEW  (no copy, fast)
  2. Advanced indexing  → returns a COPY  (integer/boolean arrays)
  3. Fancy indexing     → returns a COPY  (arrays of indices)

SLICE SYNTAX: arr[start:stop:step]
  • Negative indices count from end: -1 = last element
  • Omitting values: arr[:5] = arr[0:5], arr[::2] = every other
""")

arr = np.arange(24).reshape(4,6)
print("Base array (4×6):\n", arr)

# Basic 1D
print(SEP2)
print("BASIC 1D INDEXING:")
v = np.array([10,20,30,40,50,60,70])
print(f"  v              = {v}")
print(f"  v[0]           = {v[0]}")
print(f"  v[-1]          = {v[-1]}   (last)")
print(f"  v[-2]          = {v[-2]}   (second-to-last)")
print(f"  v[1:5]         = {v[1:5]}")
print(f"  v[::2]         = {v[::2]}  (every 2nd)")
print(f"  v[1::2]        = {v[1::2]} (odd indices)")
print(f"  v[::-1]        = {v[::-1]} (reversed)")
print(f"  v[2:6:2]       = {v[2:6:2]}")

# Basic 2D
print(SEP2)
print("BASIC 2D INDEXING:")
print(f"  arr[1,3]       = {arr[1,3]}    (row 1, col 3)")
print(f"  arr[1][3]      = {arr[1][3]}    (same — but slower)")
print(f"  arr[0]         = {arr[0]}  (entire row 0)")
print(f"  arr[:,2]       = {arr[:,2]}  (entire column 2)")
print(f"  arr[1:3, 2:5]  =\n{arr[1:3, 2:5]}  (sub-block)")
print(f"  arr[::2, ::3]  =\n{arr[::2, ::3]}  (every 2nd row, 3rd col)")

# Advanced — integer array indexing
print(SEP2)
print("ADVANCED INDEXING (integer arrays) → COPY:")
row_idx = np.array([0, 2, 3])
print(f"  arr[row_idx]     → rows 0,2,3:\n{arr[row_idx]}")
col_idx = np.array([1, 4])
print(f"  arr[:, col_idx]  → cols 1,4:\n{arr[:, col_idx]}")
# Paired indexing
rows = np.array([0,1,2,3])
cols = np.array([1,3,0,4])
print(f"  arr[rows, cols]  → pairs: {arr[rows, cols]}  (arr[0,1], arr[1,3], arr[2,0], arr[3,4])")

# Boolean / mask indexing
print(SEP2)
print("BOOLEAN MASK INDEXING → COPY:")
mask = arr > 12
print(f"  arr > 12:\n{mask}")
print(f"  arr[arr>12] = {arr[mask]}")
print(f"  arr[(arr>5) & (arr<15)] = {arr[(arr>5)&(arr<15)]}")

# np.where
print(SEP2)
print("np.where — conditional element selection:")
x = np.array([-3, -1, 0, 2, 5, -4, 7])
print(f"  x = {x}")
print(f"  np.where(x>0, x, 0)    = {np.where(x>0, x, 0)}   (ReLU!)")
print(f"  np.where(x>0, 'pos','neg') = {np.where(x>0,'pos','neg')}")
ri, ci = np.where(arr > 18)
print(f"  np.where(arr>18) → rows={ri}, cols={ci}")

# Ellipsis and newaxis
print(SEP2)
print("ELLIPSIS (...) and np.newaxis:")
cube = np.arange(27).reshape(3,3,3)
print(f"  cube[..., 1]   → all rows+cols, col 1:\n{cube[...,1]}")
v2 = np.array([1,2,3])
print(f"  v2.shape = {v2.shape}")
print(f"  v2[np.newaxis,:].shape = {v2[np.newaxis,:].shape}  (row vector)")
print(f"  v2[:,np.newaxis].shape = {v2[:,np.newaxis].shape}  (column vector)")
print(f"  Same as v2.reshape(1,-1) and v2.reshape(-1,1)")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 5 — RESHAPING, STACKING, SPLITTING
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 5 — RESHAPING, STACKING, SPLITTING")
print(SEP)

print("""
THEORY
──────
Reshaping = changing shape without changing data.
  • reshape() returns a VIEW when possible (contiguous data)
  • flatten() always returns a COPY; ravel() prefers a VIEW
  • -1 as dimension = "infer this value"

STACKING joins arrays along new or existing axes.
SPLITTING divides one array into multiple.
""")

a5 = np.arange(12)
print("Base array:", a5)

print(SEP2)
print("RESHAPING:")
print(f"  reshape(3,4):\n{a5.reshape(3,4)}")
print(f"  reshape(2,-1):   {a5.reshape(2,-1).shape}  (-1 inferred = 6)")
print(f"  reshape(-1,3):   {a5.reshape(-1,3).shape}  (-1 inferred = 4)")
print(f"  reshape(2,2,3):\n{a5.reshape(2,2,3)}")

m = a5.reshape(3,4)
print(f"\n  flatten()  → always copy: {m.flatten()}")
print(f"  ravel()    → prefers view: {m.ravel()}")
print(f"  ravel('F') → column-major: {m.ravel('F')}")

print(SEP2)
print("TRANSPOSE:")
print(f"  m.T:\n{m.T}  shape={m.T.shape}")
print(f"  m.T is a VIEW: {np.shares_memory(m, m.T)}")
cube = np.arange(24).reshape(2,3,4)
print(f"  cube.shape = {cube.shape}")
print(f"  cube.transpose(2,0,1).shape = {cube.transpose(2,0,1).shape}  (axes reordered)")
print(f"  np.swapaxes(cube,0,1).shape = {np.swapaxes(cube,0,1).shape}  (swap two axes)")

print(SEP2)
print("STACKING:")
a5a = np.array([[1,2],[3,4]])
a5b = np.array([[5,6],[7,8]])
print(f"  vstack:\n{np.vstack([a5a, a5b])}")
print(f"  hstack:\n{np.hstack([a5a, a5b])}")
print(f"  dstack (depth):\n{np.dstack([a5a, a5b])}")
print(f"  concatenate axis=0:\n{np.concatenate([a5a, a5b], axis=0)}")
print(f"  concatenate axis=1:\n{np.concatenate([a5a, a5b], axis=1)}")
print(f"  stack (NEW axis):\n{np.stack([a5a, a5b], axis=0)}")

print(SEP2)
print("SPLITTING:")
arr_split = np.arange(24).reshape(4,6)
print(f"  arr:\n{arr_split}")
parts = np.split(arr_split, 2, axis=0)
print(f"  split into 2 along axis=0: {[p.shape for p in parts]}")
parts_col = np.split(arr_split, 3, axis=1)
print(f"  split into 3 along axis=1: {[p.shape for p in parts_col]}")
# vsplit / hsplit
tops, bots = np.vsplit(arr_split, 2)
print(f"  vsplit top:\n{tops}\n  vsplit bot:\n{bots}")
lefts = np.hsplit(arr_split, [2,4])  # split at cols 2 and 4
print(f"  hsplit at [2,4]: {[p.shape for p in lefts]}")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 6 — UFUNCS & BROADCASTING
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 6 — UNIVERSAL FUNCTIONS (ufuncs) & BROADCASTING")
print(SEP)

print("""
THEORY — UFUNCS
───────────────
A ufunc (universal function) operates ELEMENT-WISE on arrays.
They are implemented in C and are orders of magnitude faster than Python loops.

Categories:
  • Arithmetic : add, subtract, multiply, divide, power, mod, floor_divide
  • Trig       : sin, cos, tan, arcsin, arccos, arctan, arctan2
  • Exp/Log    : exp, exp2, log, log2, log10, expm1, log1p
  • Comparison : greater, less, equal, maximum, minimum, fmax, fmin
  • Bit ops    : bitwise_and, bitwise_or, left_shift, right_shift
  • Floating   : abs, ceil, floor, round, sign, sqrt, square, reciprocal
  • Logical    : logical_and, logical_or, logical_not, logical_xor

ufunc extras:
  .reduce()   → apply cumulatively along axis (like sum)
  .accumulate()→ running result
  .outer()    → outer product
  .at()       → unbuffered in-place operation
""")

x6 = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print("Trigonometric ufuncs (angles in radians):")
print(f"  x = {x6.round(3)}")
print(f"  sin(x) = {np.sin(x6).round(4)}")
print(f"  cos(x) = {np.cos(x6).round(4)}")
print(f"  tan(x) = {np.tan(x6).round(4)}")

print("\nExponential/Log ufuncs:")
vals = np.array([1, np.e, np.e**2, 100])
print(f"  exp([0,1,2,3]) = {np.exp([0,1,2,3]).round(4)}")
print(f"  log(e^[0,1,2])= {np.log([1,np.e,np.e**2]).round(4)}")
print(f"  log2([1,2,4,8])= {np.log2([1,2,4,8])}")
print(f"  log10([1,10,100])= {np.log10([1,10,100])}")
print("  TIP: use log1p(x) for x near 0 (more accurate than log(1+x))")

print("\nufunc reduce & accumulate:")
a6 = np.array([1,2,3,4,5])
print(f"  np.add.reduce([1,2,3,4,5])       = {np.add.reduce(a6)}")
print(f"  np.add.accumulate([1,2,3,4,5])   = {np.add.accumulate(a6)}")
print(f"  np.multiply.reduce([1,2,3,4,5])  = {np.multiply.reduce(a6)}")
print(f"  np.maximum.reduce([3,1,4,1,5,9]) = {np.maximum.reduce([3,1,4,1,5,9])}")

print("\nufunc outer product:")
print(f"  np.multiply.outer([1,2,3],[1,2,3]):\n{np.multiply.outer([1,2,3],[1,2,3])}")
print(f"  np.add.outer([0,10,20],[1,2,3]):\n{np.add.outer([0,10,20],[1,2,3])}")

print("""
THEORY — BROADCASTING
─────────────────────
Broadcasting: NumPy automatically expands dimensions so arrays with different
shapes can work together — WITHOUT copying data.

RULES (applied right-to-left on shapes):
  1. If arrays differ in # of dims, prepend 1s to shorter shape.
  2. Arrays with size 1 along a dimension are "stretched" to match.
  3. If sizes differ and neither is 1 → ERROR.

EXAMPLES:
  (3,4) + (4,)  → (4,) becomes (1,4) → (3,4) ✓
  (3,1) + (1,4) → (3,4) ✓
  (3,4) + (3,)  → (1,3) + (3,4)... → ERROR (3≠4)
""")

print("Broadcasting examples:")
A = np.arange(12).reshape(3,4)
b_row = np.array([1,2,3,4])          # shape (4,) → broadcasts to (3,4)
b_col = np.array([10,20,30]).reshape(3,1)  # shape (3,1) → broadcasts to (3,4)

print(f"\n  A (3×4):\n{A}")
print(f"  b_row {b_row.shape}: {b_row}")
print(f"  A + b_row:\n{A + b_row}  ← adds row to EACH row of A")
print(f"  b_col {b_col.shape}: {b_col.T}")
print(f"  A + b_col:\n{A + b_col}  ← adds col value to EACH col of A")
print(f"  b_col + b_row → outer sum (3×4):\n{b_col + b_row}")

print("\nBroadcasting: Z-score normalisation (ML staple):")
data = np.random.randn(5, 4)
mean = data.mean(axis=0)   # shape (4,)
std  = data.std(axis=0)    # shape (4,)
z    = (data - mean) / std  # broadcasting (5,4)-(4,) / (4,) → (5,4)
print(f"  data shape: {data.shape}, mean shape: {mean.shape}")
print(f"  z-scores:\n{z.round(3)}")
print(f"  z.mean(axis=0) ≈ 0: {z.mean(axis=0).round(10)}")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 7 — AGGREGATIONS & STATISTICS
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 7 — AGGREGATIONS & STATISTICS")
print(SEP)

print("""
THEORY
──────
Aggregation functions reduce an array to a smaller one (or scalar).
The axis parameter controls the direction of reduction:
  axis=None → reduce ALL elements → scalar
  axis=0    → reduce ALONG rows → one value per column
  axis=1    → reduce ALONG cols → one value per row

For a (3,4) array:
  .sum(axis=0) → shape (4,)   — column sums
  .sum(axis=1) → shape (3,)   — row sums
  .sum()       → scalar       — grand total

NaN-safe versions: nansum, nanmean, nanstd, nanmedian, nanpercentile, etc.
""")

data7 = np.array([[4,  7,  2,  9],
                  [1,  5,  8,  3],
                  [6, np.nan, 4, 11]])
print("Data (with one NaN):\n", data7)

print(SEP2)
print("REDUCTION FUNCTIONS:")
print(f"  sum()          = {np.nansum(data7)}")
print(f"  sum(axis=0)    = {np.nansum(data7,axis=0)}  (column sums)")
print(f"  sum(axis=1)    = {np.nansum(data7,axis=1)}  (row sums)")
print(f"  nanmean()      = {np.nanmean(data7):.4f}")
print(f"  nanmedian()    = {np.nanmedian(data7):.4f}")
print(f"  nanstd()       = {np.nanstd(data7):.4f}  (population std, ddof=0)")
print(f"  nanstd(ddof=1) = {np.nanstd(data7, ddof=1):.4f}  (sample std)")
print(f"  nanvar()       = {np.nanvar(data7):.4f}")
print(f"  nanmin()       = {np.nanmin(data7)}")
print(f"  nanmax()       = {np.nanmax(data7)}")

print(SEP2)
print("CUMULATIVE OPERATIONS:")
v7 = np.array([3,1,4,1,5,9,2,6])
print(f"  v = {v7}")
print(f"  cumsum:  {np.cumsum(v7)}")
print(f"  cumprod: {np.cumprod(v7)}")
print(f"  diff:    {np.diff(v7)}   (v[i+1]-v[i])")
print(f"  diff 2nd:{np.diff(v7,n=2)}")

print(SEP2)
print("SORTING:")
arr7 = np.array([3,1,4,1,5,9,2,6,5,3])
print(f"  arr = {arr7}")
print(f"  sort:    {np.sort(arr7)}          (returns copy)")
print(f"  argsort: {np.argsort(arr7)}       (sorted indices)")
print(f"  argmin:  {np.argmin(arr7)}, argmax: {np.argmax(arr7)}")
m7 = np.array([[3,1,4],[1,5,9],[2,6,5]])
print(f"  sort matrix axis=1 (each row):\n{np.sort(m7, axis=1)}")
print(f"  argsort matrix axis=0 (each col):\n{np.argsort(m7, axis=0)}")
print(f"  partition([3,1,4,1,5],2): {np.partition(arr7,3)}")

print(SEP2)
print("SEARCHING:")
arr_s = np.array([5,3,8,1,9,2,7])
print(f"  arr = {arr_s}")
print(f"  argmin={np.argmin(arr_s)}, argmax={np.argmax(arr_s)}")
print(f"  nonzero: {np.nonzero(arr_s>5)}")
print(f"  where>5: {np.where(arr_s>5)[0]}")
print(f"  searchsorted([1,3,5,7,9], 4) = {np.searchsorted([1,3,5,7,9], 4)}")

print(SEP2)
print("PERCENTILES & QUANTILES:")
data_p = np.random.normal(100, 15, 1000)
print(f"  Dataset: Normal(μ=100, σ=15), n=1000")
print(f"  p25  = {np.percentile(data_p, 25):.2f}")
print(f"  p50  = {np.percentile(data_p, 50):.2f}  (median)")
print(f"  p75  = {np.percentile(data_p, 75):.2f}")
print(f"  IQR  = {np.percentile(data_p,75)-np.percentile(data_p,25):.2f}")
print(f"  p90  = {np.percentile(data_p, 90):.2f}")
print(f"  quantile(0.95) = {np.quantile(data_p, 0.95):.2f}")

print(SEP2)
print("HISTOGRAM:")
counts, edges = np.histogram(data_p, bins=10, range=(50,150))
print(f"  histogram bins:  {edges.round(1)}")
print(f"  counts:          {counts}")
print(f"  histogram2d: counts, xe, ye = np.histogram2d(x, y, bins=20)")

print(SEP2)
print("UNIQUE, COUNTS, SET OPERATIONS:")
arr_u = np.array([3,1,4,1,5,9,2,6,5,3,5])
uniq, counts7 = np.unique(arr_u, return_counts=True)
print(f"  arr = {arr_u}")
print(f"  unique:          {uniq}")
print(f"  counts:          {counts7}")
a_set = np.array([1,2,3,4,5])
b_set = np.array([3,4,5,6,7])
print(f"  intersect1d: {np.intersect1d(a_set,b_set)}")
print(f"  union1d:     {np.union1d(a_set,b_set)}")
print(f"  setdiff1d:   {np.setdiff1d(a_set,b_set)}  (a not in b)")
print(f"  in1d:        {np.isin(a_set,b_set)}")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 8 — LINEAR ALGEBRA
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 8 — LINEAR ALGEBRA (np.linalg)")
print(SEP)

print("""
THEORY
──────
np.linalg provides the essential building blocks of matrix algebra, used
everywhere in ML (regression, PCA, SVD, neural net weight updates).

KEY OPERATIONS:
  Dot product  : A @ B  or  np.dot(A,B)
  Element-wise : A * B  (NOT matrix multiply!)
  Transpose    : A.T
  Inverse      : np.linalg.inv(A)
  Determinant  : np.linalg.det(A)
  Eigenvalues  : np.linalg.eig(A)
  SVD          : np.linalg.svd(A)
  Solve Ax=b   : np.linalg.solve(A,b)
  Least squares: np.linalg.lstsq(A,b)
  Norms        : np.linalg.norm(A)
  Rank         : np.linalg.matrix_rank(A)
""")

# Matrix multiply
A = np.array([[1,2,3],[4,5,6]])   # 2×3
B = np.array([[1,0],[0,1],[2,3]]) # 3×2
C = A @ B                          # 2×2
print("Matrix multiplication (@):")
print(f"  A (2×3):\n{A}")
print(f"  B (3×2):\n{B}")
print(f"  A @ B:\n{C}")
print(f"  np.dot(A,B):\n{np.dot(A,B)}  (same as @)")

# Inverse & solve
print(SEP2)
M = np.array([[2.,1.],[-1.,3.]])
print(f"  M:\n{M}")
print(f"  inv(M):\n{np.linalg.inv(M).round(4)}")
print(f"  M @ inv(M) ≈ I:\n{(M @ np.linalg.inv(M)).round(10)}")
print(f"  det(M) = {np.linalg.det(M):.4f}")

# Solve Ax = b
print(SEP2)
print("Solving Ax = b (systems of equations):")
A_sys = np.array([[3.,1.],[1.,2.]])
b_sys = np.array([9.,8.])
x_sol = np.linalg.solve(A_sys, b_sys)
print(f"  A={A_sys.tolist()}, b={b_sys.tolist()}")
print(f"  Solution x = {x_sol}  (verify: A@x = {A_sys@x_sol})")

# Eigenvalues
print(SEP2)
print("Eigenvalues & Eigenvectors:")
sym = np.array([[4.,2.],[2.,3.]])
eigenvals, eigenvecs = np.linalg.eig(sym)
print(f"  Matrix:\n{sym}")
print(f"  Eigenvalues:  {eigenvals.round(4)}")
print(f"  Eigenvectors:\n{eigenvecs.round(4)}")
print(f"  Verify: M@v = λ·v for v1: {(sym@eigenvecs[:,0]).round(4)} vs {(eigenvals[0]*eigenvecs[:,0]).round(4)}")

# SVD
print(SEP2)
print("SVD — Singular Value Decomposition:")
M_svd = np.array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,2,0,0,0]])
U, s, Vt = np.linalg.svd(M_svd, full_matrices=True)
print(f"  M (4×5):\n{M_svd}")
print(f"  Singular values: {s.round(4)}")
print(f"  U shape: {U.shape}, Vt shape: {Vt.shape}")
print(f"  Rank = {np.linalg.matrix_rank(M_svd)}")

# Least squares
print(SEP2)
print("Least Squares regression (β̂ = (XᵀX)⁻¹Xᵀy):")
np.random.seed(42)
X_ls = np.column_stack([np.ones(20), np.random.randn(20)])
y_ls = 2 + 3*X_ls[:,1] + np.random.randn(20)*0.5
coeffs, residuals, rank, sv = np.linalg.lstsq(X_ls, y_ls, rcond=None)
print(f"  True: β₀=2, β₁=3")
print(f"  Estimated: β₀={coeffs[0]:.4f}, β₁={coeffs[1]:.4f}")

# Norms
print(SEP2)
print("Norms:")
v8 = np.array([3.,4.])
m8 = np.array([[1,2],[3,4]])
print(f"  L2 norm [3,4]:     {np.linalg.norm(v8):.4f}   (Euclidean = √(3²+4²))")
print(f"  L1 norm [3,4]:     {np.linalg.norm(v8,1):.4f}  (Manhattan = |3|+|4|)")
print(f"  L∞ norm [3,4]:     {np.linalg.norm(v8,np.inf):.4f}  (Chebyshev = max(|3|,|4|))")
print(f"  Frobenius norm M:  {np.linalg.norm(m8,'fro'):.4f}  (√sum of squares)")
print(f"  Matrix L2 norm M:  {np.linalg.norm(m8,2):.4f}  (largest singular value)")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 9 — RANDOM NUMBER GENERATION
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 9 — RANDOM NUMBER GENERATION")
print(SEP)

print("""
THEORY
──────
NumPy 1.17+ introduced the new random Generator API via np.random.default_rng().
ALWAYS prefer the new API over the legacy np.random.X functions:

  NEW (recommended):   rng = np.random.default_rng(seed=42)
  LEGACY (avoid):      np.random.seed(42); np.random.randn()

NEW API benefits: thread-safe, reproducible, richer distributions.

DISTRIBUTIONS (most important for ML):
  rng.random()       → Uniform[0,1)
  rng.uniform(a,b)   → Uniform[a,b)
  rng.integers(low,high) → random integers
  rng.normal(μ,σ)    → Gaussian (most common!)
  rng.standard_normal()  → N(0,1)
  rng.exponential(λ) → Exponential
  rng.poisson(λ)     → Poisson (count data)
  rng.binomial(n,p)  → Binomial
  rng.multinomial(n,p) → Multinomial
  rng.choice(arr)    → random selection
  rng.shuffle(arr)   → in-place shuffle
  rng.permutation(n) → shuffled indices
  rng.multivariate_normal(μ, Σ) → multivariate Gaussian
""")

rng9 = np.random.default_rng(seed=42)

print("Uniform distributions:")
print(f"  random(5):        {rng9.random(5).round(4)}")
print(f"  uniform(1,10,5):  {rng9.uniform(1,10,5).round(4)}")
print(f"  integers(0,100,5):{rng9.integers(0,100,5)}")

print("\nGaussian distributions:")
normal_samples = rng9.normal(loc=0, scale=1, size=10000)
print(f"  normal(0,1,10000): mean={normal_samples.mean():.4f}, std={normal_samples.std():.4f}")
print(f"  % within 1σ: {(np.abs(normal_samples)<1).mean()*100:.1f}%  (expected: 68.3%)")
print(f"  % within 2σ: {(np.abs(normal_samples)<2).mean()*100:.1f}%  (expected: 95.4%)")
print(f"  % within 3σ: {(np.abs(normal_samples)<3).mean()*100:.1f}%  (expected: 99.7%)")

print("\nDiscrete distributions:")
print(f"  binomial(n=10,p=0.3,size=5):   {rng9.binomial(n=10,p=0.3,size=5)}")
print(f"  poisson(λ=4,size=5):            {rng9.poisson(lam=4,size=5)}")

print("\nSampling and shuffling:")
arr9 = np.arange(10)
print(f"  choice(arr,5,replace=False):  {rng9.choice(arr9,5,replace=False)}")
print(f"  choice(arr,5,replace=True):   {rng9.choice(arr9,5,replace=True)}")
idx = rng9.permutation(len(arr9))
print(f"  permutation indices:           {idx}")
print(f"  arr[permutation]:              {arr9[idx]}")

print("\nMultivariate Normal (correlation structure):")
mean9 = np.array([0, 0])
cov9  = np.array([[1.0, 0.8],
                   [0.8, 1.0]])   # correlated
samples9 = rng9.multivariate_normal(mean9, cov9, size=1000)
print(f"  Cov [[1,0.8],[0.8,1]] → actual corr: {np.corrcoef(samples9[:,0],samples9[:,1])[0,1]:.4f}")

print("\nReproducibility:")
rng_a = np.random.default_rng(seed=99)
rng_b = np.random.default_rng(seed=99)
print(f"  Same seed → same output: {np.allclose(rng_a.random(5), rng_b.random(5))}")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 10 — BOOLEAN MASKING & STRUCTURED ARRAYS
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 10 — BOOLEAN MASKING & STRUCTURED ARRAYS")
print(SEP)

print("""
THEORY — BOOLEAN MASKING
─────────────────────────
Boolean operations on arrays return arrays of True/False — masks.
These are used to:
  1. Filter elements (fancy indexing with mask)
  2. Count/sum matching elements (True=1, False=0 in arithmetic)
  3. Replace elements conditionally (np.where, array[mask] = value)

COMPARISONS: ==, !=, <, <=, >, >=
LOGICAL OPS: & (and), | (or), ~ (not), ^ (xor) — NOT 'and'/'or'!
             np.logical_and, np.logical_or, np.logical_not
""")

rng10 = np.random.default_rng(42)
grades = rng10.integers(50, 101, size=20)
print(f"grades = {grades}")

print(SEP2)
print("FILTERING:")
print(f"  grades > 85:          {grades[grades>85]}")
print(f"  pass (≥60):           {grades[grades>=60]}")
print(f"  grades 70-90:         {grades[(grades>=70) & (grades<=90)]}")
print(f"  fail or A:            {grades[(grades<60) | (grades>90)]}")

print(SEP2)
print("COUNTING & STATISTICS ON MASKS:")
mask10 = grades >= 60
print(f"  passed:       {mask10.sum()} out of {len(grades)}")
print(f"  pass rate:    {mask10.mean()*100:.1f}%")
print(f"  mean of pass: {grades[mask10].mean():.2f}")
print(f"  mean of fail: {grades[~mask10].mean():.2f}")

print(SEP2)
print("MODIFYING WITH MASK:")
temp_grades = grades.copy()
temp_grades[temp_grades < 60] = 0    # zero out fails
print(f"  zero out fails: {temp_grades}")
temp_grades2 = np.where(grades >= 60, grades, 59)
print(f"  floor to 59:    {temp_grades2}")
# Grade letters
letters = np.where(grades>=90,'A', np.where(grades>=80,'B', np.where(grades>=70,'C','D')))
print(f"  letter grades:  {letters}")

print(SEP2)
print("ANY, ALL, ALLCLOSE:")
print(f"  any grade > 98: {np.any(grades>98)}")
print(f"  all grades pass:{np.all(grades>=60)}")
a10 = np.array([1.0, 2.0, 3.0])
b10 = np.array([1.0000001, 1.9999999, 3.0000001])
print(f"  allclose(a,b):  {np.allclose(a10, b10)}  (within tolerance)")
print(f"  array_equal:    {np.array_equal(a10, b10)}")

print(SEP2)
print("STRUCTURED ARRAYS:")
dt10 = np.dtype([
    ('name',  'U15'),
    ('age',   'i4'),
    ('salary','f8'),
    ('dept',  'U10'),
])
employees = np.array([
    ('Alice',   32, 95000., 'Engineering'),
    ('Bob',     45, 82000., 'Marketing'),
    ('Charlie', 28, 110000.,'Engineering'),
    ('Diana',   37, 88000., 'HR'),
    ('Eve',     41, 125000.,'Engineering'),
], dtype=dt10)

print(f"  employees['name']:   {employees['name']}")
print(f"  employees['salary']: {employees['salary']}")
eng = employees[employees['dept']=='Engineering']
print(f"  Engineering dept:\n    names: {eng['name']}")
print(f"    avg salary: ${eng['salary'].mean():,.2f}")
print(f"  Sorted by salary:\n{employees[np.argsort(employees['salary'])[::-1]]['name']}")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 11 — PERFORMANCE: VECTORIZATION vs LOOPS
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 11 — PERFORMANCE: VECTORIZATION vs LOOPS")
print(SEP)

print("""
THEORY
──────
The golden rule of NumPy: NEVER loop if you can vectorize.

  Python loop: ~100ns overhead PER iteration (interpreter, type checks)
  NumPy ufunc: ~1ns per element (pure C, SIMD, no Python overhead)

  For 1 million elements: loop ≈ 100ms, vectorized ≈ 1ms → 100× faster

VECTORIZATION STRATEGIES:
  1. Replace loops with ufuncs / arithmetic operators
  2. Use aggregation functions (sum, max) instead of accumulating in loop
  3. Use boolean masks instead of if/else in loops
  4. Use broadcasting instead of tiling / repmat
  5. Use @ instead of nested loops for matrix ops
  6. Use np.einsum for complex tensor contractions

MEMORY TIPS:
  • Prefer in-place ops (+=, *=) to avoid temporaries
  • Use np.empty() then fill instead of np.zeros() when you'll overwrite
  • Work with float32 in ML (2× memory, ~2× throughput)
  • Use .ravel() to work on 1D views
""")

n11 = 1_000_000
data11 = np.random.randn(n11)

# Loop vs vectorized
print("SPEED COMPARISON — sum of squares:")
t0 = time.perf_counter()
total_loop = sum(x*x for x in data11)
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
total_np = np.dot(data11, data11)
t_np = time.perf_counter() - t0

print(f"  Python loop:  {t_loop*1000:.1f} ms  → {total_loop:.4f}")
print(f"  NumPy dot:    {t_np*1000:.2f} ms  → {total_np:.4f}")
print(f"  Speedup:      {t_loop/t_np:.0f}×")

# np.einsum
print(SEP2)
print("np.einsum — Einstein summation notation:")
A11 = np.random.randn(50, 30)
B11 = np.random.randn(30, 20)

# Matrix multiply
C11 = np.einsum('ij,jk->ik', A11, B11)
print(f"  einsum('ij,jk->ik', A, B) ← matrix multiply: shape {C11.shape}")
# Trace
T = np.random.randn(5,5)
print(f"  einsum('ii', T) ← trace: {np.einsum('ii',T):.4f}  (vs np.trace={np.trace(T):.4f})")
# Outer product
v_a = np.array([1,2,3]); v_b = np.array([10,20])
print(f"  einsum('i,j->ij', a, b) ← outer product:\n{np.einsum('i,j->ij',v_a,v_b)}")
# Batch matrix multiply
Batched_A = np.random.randn(10, 4, 3)
Batched_B = np.random.randn(10, 3, 5)
Batched_C = np.einsum('bij,bjk->bik', Batched_A, Batched_B)
print(f"  einsum('bij,bjk->bik') ← batch matmul: shape {Batched_C.shape}")

print(SEP2)
print("MEMORY-EFFICIENT TRICKS:")
big = np.random.randn(1000, 1000)
# In-place vs out-of-place
t0 = time.perf_counter()
result_oop = big * 2 + 1   # creates 2 temporary arrays
t_oop = time.perf_counter() - t0

big2 = big.copy()
t0 = time.perf_counter()
big2 *= 2; big2 += 1       # in-place: no temporaries
t_ip = time.perf_counter() - t0
print(f"  Out-of-place (*2 +1): {t_oop*1000:.3f} ms")
print(f"  In-place (*=2, +=1):  {t_ip*1000:.3f} ms  (saves 2 allocations)")

print(SEP2)
print("CONTIGUOUS MEMORY — C vs F order:")
arr_c = np.ascontiguousarray(np.random.randn(1000, 1000))   # C-order (row-major)
arr_f = np.asfortranarray(arr_c)                              # F-order (col-major)

t0 = time.perf_counter()
_ = arr_c.sum(axis=1)    # summing rows is fast for C-order
t_c = time.perf_counter() - t0

t0 = time.perf_counter()
_ = arr_f.sum(axis=1)    # summing rows is slow for F-order
t_f = time.perf_counter() - t0

print(f"  C-order row sum:  {t_c*1000:.3f} ms")
print(f"  F-order row sum:  {t_f*1000:.3f} ms  (cache misses → slower)")
print(f"  is C-contiguous: {arr_c.flags['C_CONTIGUOUS']}")
print(f"  is F-contiguous: {arr_f.flags['F_CONTIGUOUS']}")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 12 — REAL-WORLD PROJECTS
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("MODULE 12 — REAL-WORLD PROJECTS")
print(SEP)

rng12 = np.random.default_rng(42)

# ── PROJECT A: Image Processing ──────────────────────────────────────────
print("PROJECT A — IMAGE PROCESSING")
print("─"*40)
print("""
Images are 3D arrays: (height, width, channels)
  • Grayscale: (H, W)         — single channel
  • RGB color: (H, W, 3)      — R,G,B channels, uint8 values 0-255
  • RGBA:      (H, W, 4)      — with alpha channel
""")

# Simulate 100×100 RGB image
image = rng12.integers(0, 256, (100, 100, 3), dtype=np.uint8)
print(f"  image.shape = {image.shape}  dtype={image.dtype}")

# Convert to grayscale: ITU-R BT.601 formula
gray = (0.2989*image[:,:,0] + 0.5870*image[:,:,1] + 0.1140*image[:,:,2]).astype(np.uint8)
print(f"  grayscale.shape = {gray.shape}")

# Crop
crop = image[20:60, 30:70, :]
print(f"  crop[20:60, 30:70].shape = {crop.shape}")

# Flip
flipped_h = image[:, ::-1, :]    # horizontal flip
flipped_v = image[::-1, :, :]    # vertical flip
print(f"  horizontal flip: {flipped_h.shape}")

# Normalize to [0,1] for ML
normalized = image.astype(np.float32) / 255.0
print(f"  normalized dtype={normalized.dtype}, min={normalized.min():.3f}, max={normalized.max():.3f}")

# Simple convolution / blur (mean filter 3×3)
def mean_blur(img_channel, k=3):
    """Manually apply k×k mean filter (no padding)."""
    H, W = img_channel.shape
    out = np.empty((H-k+1, W-k+1), dtype=np.float32)
    for i in range(0, H-k+1):
        for j in range(0, W-k+1):
            out[i,j] = img_channel[i:i+k, j:j+k].mean()
    return out

# Vectorized version using stride tricks
def vectorized_mean_blur(img_channel, k=3):
    from numpy.lib.stride_tricks import as_strided
    H, W = img_channel.shape
    shape   = (H-k+1, W-k+1, k, k)
    strides = img_channel.strides + img_channel.strides
    patches = as_strided(img_channel, shape=shape, strides=strides)
    return patches.mean(axis=(-2,-1))

gray_f = gray.astype(np.float32)
blurred = vectorized_mean_blur(gray_f, k=5)
print(f"  5×5 mean blur output shape: {blurred.shape}  (vectorized with stride tricks)")

# Histogram equalization (contrast enhancement)
hist, bins = np.histogram(gray.flatten(), bins=256, range=(0,256))
cdf  = hist.cumsum()
cdf_min = cdf[cdf > 0].min()
equalized = np.round((cdf[gray] - cdf_min) / (gray.size - cdf_min) * 255).astype(np.uint8)
print(f"  histogram equalized: min={equalized.min()}, max={equalized.max()}")

# ── PROJECT B: Statistics / Finance ──────────────────────────────────────
print(SEP2)
print("PROJECT B — FINANCIAL TIME SERIES")
print("""
Stock prices stored as 1D array. Common operations:
  daily returns   = (p[t]-p[t-1])/p[t-1]  = np.diff(p)/p[:-1]
  log returns     = log(p[t]/p[t-1])        = np.diff(np.log(p))
  rolling mean    = moving average (using stride tricks or loops)
  rolling std     = rolling volatility
  Sharpe ratio    = mean_return / std_return * √252
""")
n_days = 500
returns = rng12.normal(loc=0.001, scale=0.02, size=n_days)
prices  = 100 * np.exp(np.cumsum(returns))     # geometric random walk

print(f"  {n_days}-day simulated stock prices")
print(f"  Start: ${prices[0]:.2f}, End: ${prices[-1]:.2f}")
print(f"  Min:   ${prices.min():.2f}, Max: ${prices.max():.2f}")

daily_ret   = np.diff(prices) / prices[:-1]
log_ret     = np.diff(np.log(prices))
print(f"\n  Daily returns:  mean={daily_ret.mean()*100:.4f}%, std={daily_ret.std()*100:.4f}%")
print(f"  Log returns:    mean={log_ret.mean()*100:.4f}%, std={log_ret.std()*100:.4f}%")

sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
print(f"  Annualised Sharpe ratio: {sharpe:.4f}")

# Rolling statistics using stride tricks
def rolling_stats(arr, window):
    from numpy.lib.stride_tricks import as_strided
    n = len(arr)
    shape   = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = as_strided(arr, shape=shape, strides=strides)
    return windows.mean(axis=1), windows.std(axis=1)

roll_mean, roll_std = rolling_stats(prices, window=20)
print(f"\n  20-day rolling mean shape: {roll_mean.shape}")
print(f"  20-day rolling std  shape: {roll_std.shape}")

# Drawdown
peak = np.maximum.accumulate(prices)
drawdown = (prices - peak) / peak
max_dd   = drawdown.min()
print(f"  Maximum drawdown: {max_dd*100:.2f}%")

# Correlation matrix (multi-stock)
n_stocks = 5
stock_returns = rng12.multivariate_normal(
    mean=np.zeros(n_stocks),
    cov=np.array([[1.0,.6,.4,.3,.2],[.6,1.,.5,.4,.1],[.4,.5,1.,.3,.2],
                  [.3,.4,.3,1.,.5],[.2,.1,.2,.5,1.]]),
    size=252
)
corr_matrix = np.corrcoef(stock_returns.T)
print(f"\n  5-stock correlation matrix:\n{corr_matrix.round(3)}")
print(f"  (used in portfolio optimisation)")

# ── PROJECT C: Machine Learning from scratch ──────────────────────────────
print(SEP2)
print("PROJECT C — LINEAR REGRESSION FROM SCRATCH (NumPy only)")
print("""
Implementing gradient descent manually demonstrates:
  • Vectorized forward pass (matrix multiply)
  • Vectorized gradient computation (no loops!)
  • In-place parameter updates
""")
rng12b = np.random.default_rng(0)
n_samples, n_features = 200, 5
W_true = rng12b.normal(size=n_features)
b_true = 2.5
X_train = rng12b.normal(size=(n_samples, n_features))
y_train = X_train @ W_true + b_true + rng12b.normal(scale=0.5, size=n_samples)

# OLS closed form
X_b = np.hstack([np.ones((n_samples,1)), X_train])
beta_hat = np.linalg.lstsq(X_b, y_train, rcond=None)[0]
y_hat_ols = X_b @ beta_hat
mse_ols   = np.mean((y_train - y_hat_ols)**2)
print(f"  OLS closed-form MSE: {mse_ols:.6f}")

# Gradient descent
W = rng12b.normal(scale=0.01, size=n_features)
b = 0.0
lr, epochs = 0.01, 200
losses = []
for ep in range(epochs):
    y_pred_gd = X_train @ W + b
    error     = y_pred_gd - y_train           # (n,)
    dW        = (X_train.T @ error) / n_samples  # (p,) vectorized!
    db        = error.mean()
    W        -= lr * dW
    b        -= lr * db
    if ep % 40 == 0:
        mse = np.mean(error**2)
        losses.append(mse)
        print(f"  Epoch {ep:3d}: MSE = {mse:.6f}")

y_pred_final = X_train @ W + b
print(f"\n  True W:      {W_true.round(4)}")
print(f"  Learned W:   {W.round(4)}")
print(f"  True b:      {b_true:.4f}, Learned b: {b:.4f}")

# ── PROJECT D: k-Nearest Neighbours ──────────────────────────────────────
print(SEP2)
print("PROJECT D — k-NN CLASSIFIER (pure NumPy)")
print("""
Computing pairwise distances efficiently is the core challenge.
  Naive:  for i in X_test: for j in X_train: → O(n²) Python loops
  NumPy:  ‖A−B‖² = ‖A‖²+‖B‖²−2A·Bᵀ  → ONE matrix multiply!
""")
rng_knn = np.random.default_rng(7)
n_tr, n_te, d = 300, 50, 4
X_tr_knn = rng_knn.normal(size=(n_tr, d))
y_tr_knn = (X_tr_knn[:, 0] + X_tr_knn[:, 1] > 0).astype(int)
X_te_knn = rng_knn.normal(size=(n_te, d))
y_te_knn = (X_te_knn[:, 0] + X_te_knn[:, 1] > 0).astype(int)

# Vectorized pairwise L2 distances
# ‖A−B‖² = ‖A‖²+‖B‖²−2A·Bᵀ
A2 = np.sum(X_te_knn**2, axis=1, keepdims=True)   # (n_te, 1)
B2 = np.sum(X_tr_knn**2, axis=1, keepdims=True).T  # (1, n_tr)
AB = X_te_knn @ X_tr_knn.T                          # (n_te, n_tr)
dists = np.sqrt(np.maximum(A2 + B2 - 2*AB, 0))      # (n_te, n_tr)
print(f"  Distance matrix shape: {dists.shape}  (computed via matmul — no loops!)")

# k-NN prediction
k = 5
knn_idx = np.argsort(dists, axis=1)[:, :k]           # (n_te, k)
knn_labels = y_tr_knn[knn_idx]                        # (n_te, k)
y_pred_knn = (knn_labels.mean(axis=1) >= 0.5).astype(int)
accuracy = (y_pred_knn == y_te_knn).mean()
print(f"  k-NN (k={k}) accuracy: {accuracy*100:.1f}%  (all vectorized, no loops!)")


# ════════════════════════════════════════════════════════════════════════════
# EXERCISES — 120 PROBLEMS WITH SOLUTIONS
# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("EXERCISES — 120 PROBLEMS WITH SOLUTIONS")
print(SEP)

print("""
Each exercise shows the TASK, then the SOLUTION and its output.
Try solving it yourself first, then check the solution.
""")

ex_rng = np.random.default_rng(99)

# ── MODULE 1 EXERCISES ────────────────────────────────────────────────────
print("═"*50)
print("M1 EXERCISES — ndarray Internals")
print("═"*50)

print("\nEx1.1: Create a 1D array [1,2,3,4,5] and print shape, ndim, size, dtype, itemsize, nbytes.")
a_ex = np.array([1,2,3,4,5])
print(f"  shape={a_ex.shape}, ndim={a_ex.ndim}, size={a_ex.size}, dtype={a_ex.dtype}, itemsize={a_ex.itemsize}, nbytes={a_ex.nbytes}")

print("\nEx1.2: Create a float32 2D array [[1,2],[3,4]] and print its strides.")
a_ex2 = np.array([[1,2],[3,4]], dtype=np.float32)
print(f"  strides={a_ex2.strides}  (each row: {a_ex2.strides[0]} bytes = 2 × {a_ex2.itemsize})")

print("\nEx1.3: Show that a slice is a view (shares memory) but .copy() is not.")
orig = np.arange(8)
slc  = orig[2:6]
cpy  = orig[2:6].copy()
slc[0] = 99
print(f"  orig after slc[0]=99: {orig}  ← modified via view")
cpy[0] = -1
print(f"  orig after cpy[0]=-1: {orig}  ← unchanged (copy is independent)")

print("\nEx1.4: Create a 3D array of shape (2,3,4) and verify ndim, size.")
a3d = np.ones((2,3,4))
print(f"  ndim={a3d.ndim}, size={a3d.size}, shape={a3d.shape}")

print("\nEx1.5: Use np.shares_memory to verify a reshape is a view.")
base = np.arange(12)
r    = base.reshape(3,4)
print(f"  reshape shares memory: {np.shares_memory(base, r)}")

# ── MODULE 2 EXERCISES ────────────────────────────────────────────────────
print("\n" + "═"*50)
print("M2 EXERCISES — Array Creation")
print("═"*50)

print("\nEx2.1: Create a 5×5 identity matrix.")
print(f"  np.eye(5):\n{np.eye(5).astype(int)}")

print("\nEx2.2: Create array [0, 0.25, 0.5, 0.75, 1.0].")
print(f"  linspace(0,1,5): {np.linspace(0,1,5)}")

print("\nEx2.3: Create a 3×3 array with 7 on diagonal, 0 elsewhere.")
print(f"  diag([7,7,7]):\n{np.diag([7,7,7])}")

print("\nEx2.4: Create array [1,4,9,16,25] without a loop.")
print(f"  arange(1,6)**2: {np.arange(1,6)**2}")

print("\nEx2.5: Create a 4×4 checkerboard (0s and 1s).")
board = np.zeros((4,4), dtype=int)
board[1::2, ::2] = 1
board[::2, 1::2] = 1
print(f"  checkerboard:\n{board}")

print("\nEx2.6: Create [1,2,3,1,2,3,1,2,3] using tile.")
print(f"  tile([1,2,3],3): {np.tile([1,2,3],3)}")

print("\nEx2.7: Create array where element (i,j) = i + j for 3×3.")
print(f"  fromfunction(i+j,(3,3)):\n{np.fromfunction(lambda i,j: i+j,(3,3),dtype=int)}")

print("\nEx2.8: Create coordinate grid for x=[0,1,2], y=[0,10,20,30].")
xx,yy = np.meshgrid([0,1,2],[0,10,20,30])
print(f"  xx:\n{xx}\n  yy:\n{yy}")

print("\nEx2.9: Generate 10 logarithmically spaced points from 1 to 1000.")
print(f"  logspace(0,3,10): {np.logspace(0,3,10).round(2)}")

print("\nEx2.10: Create a structured array of 3 products with name(str), price(f4), qty(i4).")
dt_prod = np.dtype([('name','U15'),('price','f4'),('qty','i4')])
prods = np.array([('Apple',1.29,50),('Banana',0.59,120),('Cherry',3.49,30)], dtype=dt_prod)
print(f"  products: {prods}")
print(f"  names: {prods['name']}")
print(f"  total value: {(prods['price']*prods['qty']).sum():.2f}")

# ── MODULE 3 EXERCISES ────────────────────────────────────────────────────
print("\n" + "═"*50)
print("M3 EXERCISES — dtypes")
print("═"*50)

print("\nEx3.1: Create array [1,2,3] as float16, float32, float64. Compare memory.")
for dt in [np.float16, np.float32, np.float64]:
    a = np.array([1,2,3], dtype=dt)
    print(f"  {str(dt):20s}: {a.nbytes} bytes")

print("\nEx3.2: Show overflow with int8.")
a_i8 = np.array([127], dtype=np.int8)
print(f"  int8(127)+1 = {np.array([127],dtype=np.int8)+np.array([1],dtype=np.int8)}  (overflow!)")

print("\nEx3.3: Convert float array to int, note truncation (not rounding).")
f_arr = np.array([1.9, 2.5, 3.1, 4.8])
print(f"  [1.9,2.5,3.1,4.8].astype(int) = {f_arr.astype(int)}  (truncates, not rounds)")
print(f"  np.round then int:              = {np.round(f_arr).astype(int)}")

print("\nEx3.4: What is the max value of uint16?")
print(f"  np.iinfo(np.uint16).max = {np.iinfo(np.uint16).max}")

print("\nEx3.5: Create a boolean array from [0,1,2,0,3].")
b_arr = np.array([0,1,2,0,3], dtype=bool)
print(f"  bool: {b_arr}  (0→False, nonzero→True)")

# ── MODULE 4 EXERCISES ────────────────────────────────────────────────────
print("\n" + "═"*50)
print("M4 EXERCISES — Indexing & Slicing")
print("═"*50)

base4 = np.arange(1, 26).reshape(5, 5)
print(f"\n  base array:\n{base4}")

print("\nEx4.1: Get element at row 3, col 2.")
print(f"  base4[3,2] = {base4[3,2]}")

print("\nEx4.2: Get last column.")
print(f"  base4[:,-1] = {base4[:,-1]}")

print("\nEx4.3: Get rows 1–3, columns 1–3 (sub-block).")
print(f"  base4[1:4,1:4]:\n{base4[1:4,1:4]}")

print("\nEx4.4: Reverse the matrix (flip both axes).")
print(f"  base4[::-1,::-1]:\n{base4[::-1,::-1]}")

print("\nEx4.5: Get every other element from the diagonal.")
print(f"  diag elements:     {np.diag(base4)}")
print(f"  every other: {np.diag(base4)[::2]}")

print("\nEx4.6: Select rows [0,2,4] using fancy indexing.")
print(f"  base4[[0,2,4]]:\n{base4[[0,2,4]]}")

print("\nEx4.7: Set all elements > 15 to 0 (in-place).")
temp4 = base4.copy()
temp4[temp4 > 15] = 0
print(f"  result:\n{temp4}")

print("\nEx4.8: Get anti-diagonal using advanced indexing.")
n4 = 5
anti_diag = base4[np.arange(n4), n4-1-np.arange(n4)]
print(f"  anti-diagonal: {anti_diag}")

print("\nEx4.9: Replace all even numbers with -1.")
t4b = base4.copy()
t4b[t4b % 2 == 0] = -1
print(f"\n{t4b}")

print("\nEx4.10: Use np.where to return 'high' if val>12 else 'low'.")
categories4 = np.where(base4 > 12, 'high', 'low')
print(f"\n{categories4}")

# ── MODULE 5 EXERCISES ────────────────────────────────────────────────────
print("\n" + "═"*50)
print("M5 EXERCISES — Reshaping, Stacking, Splitting")
print("═"*50)

print("\nEx5.1: Convert [1..12] to shape (3,4), then (2,2,3).")
v5 = np.arange(1,13)
print(f"  reshape(3,4):\n{v5.reshape(3,4)}")
print(f"  reshape(2,2,3):\n{v5.reshape(2,2,3)}")

print("\nEx5.2: Flatten a 3D array using ravel vs flatten. Show memory difference.")
a5ex = np.arange(24).reshape(2,3,4)
rv = a5ex.ravel()
fl = a5ex.flatten()
print(f"  ravel shares memory: {np.shares_memory(a5ex, rv)}")
print(f"  flatten shares memory: {np.shares_memory(a5ex, fl)}")

print("\nEx5.3: Stack two (3,4) arrays vertically and horizontally.")
A5 = np.ones((3,4)); B5 = np.ones((3,4))*2
print(f"  vstack shape: {np.vstack([A5,B5]).shape}")
print(f"  hstack shape: {np.hstack([A5,B5]).shape}")

print("\nEx5.4: Split a (4,6) array into 3 equal parts along axis=1.")
arr5 = np.arange(24).reshape(4,6)
parts5 = np.split(arr5, 3, axis=1)
print(f"  shapes: {[p.shape for p in parts5]}")
print(f"  part 1:\n{parts5[0]}")

print("\nEx5.5: Transpose a (2,3,4) array to (4,2,3).")
cube5 = np.arange(24).reshape(2,3,4)
print(f"  transpose(2,0,1).shape = {cube5.transpose(2,0,1).shape}")

print("\nEx5.6: Use np.newaxis to turn a 1D array into a column vector.")
v_1d = np.array([1,2,3,4])
col = v_1d[:, np.newaxis]
print(f"  {v_1d.shape} → {col.shape}:\n{col.T}")

print("\nEx5.7: Stack 4 arrays of shape (3,) into a (4,3) matrix.")
arrs = [np.array([i,i+1,i+2]) for i in range(4)]
stacked = np.stack(arrs, axis=0)
print(f"  shape: {stacked.shape}\n{stacked}")

# ── MODULE 6 EXERCISES ────────────────────────────────────────────────────
print("\n" + "═"*50)
print("M6 EXERCISES — ufuncs & Broadcasting")
print("═"*50)

print("\nEx6.1: Compute sin²(x)+cos²(x) for x=[0,π/4,π/2,π]. Should all be 1.")
x6ex = np.array([0, np.pi/4, np.pi/2, np.pi])
res = np.sin(x6ex)**2 + np.cos(x6ex)**2
print(f"  {res.round(10)}  ← all ones ✓")

print("\nEx6.2: Compute pairwise Euclidean distances between [0,3,6] (1D points).")
pts = np.array([0.,3.,6.])
dists6 = np.abs(pts[:,np.newaxis] - pts[np.newaxis,:])
print(f"  distance matrix:\n{dists6}")

print("\nEx6.3: Normalize each ROW of a 4×5 matrix to have unit L2 norm.")
M6 = ex_rng.random((4,5))
norms6 = np.linalg.norm(M6, axis=1, keepdims=True)  # (4,1)
M6_norm = M6 / norms6  # broadcasts (4,5)/(4,1)
print(f"  row norms after normalisation: {np.linalg.norm(M6_norm, axis=1).round(10)}")

print("\nEx6.4: Compute outer product of [1,2,3] and [10,20,30] using broadcasting.")
a6ex = np.array([1,2,3])[:,np.newaxis]
b6ex = np.array([10,20,30])[np.newaxis,:]
print(f"  outer:\n{a6ex*b6ex}")

print("\nEx6.5: Apply ReLU (max(0,x)) vectorized on a random array.")
x6r = ex_rng.normal(size=8)
relu = np.maximum(x6r, 0)
print(f"  x:    {x6r.round(3)}")
print(f"  relu: {relu.round(3)}")

print("\nEx6.6: Compute a multiplication table 1–10 using outer product.")
table = np.arange(1,11)[:,None] * np.arange(1,11)[None,:]
print(f"  1–10 multiplication table:\n{table}")

print("\nEx6.7: Use np.add.accumulate to compute running total of [5,3,8,1,9].")
v6 = np.array([5,3,8,1,9])
print(f"  accumulate: {np.add.accumulate(v6)}")

# ── MODULE 7 EXERCISES ────────────────────────────────────────────────────
print("\n" + "═"*50)
print("M7 EXERCISES — Aggregations & Statistics")
print("═"*50)

data_m7 = ex_rng.normal(50, 10, (5,6))
data_m7[1,2] = np.nan  # introduce NaN
print(f"\n  data (5×6) with one NaN:\n{data_m7.round(2)}")

print("\nEx7.1: Compute column means (ignoring NaN).")
print(f"  {np.nanmean(data_m7, axis=0).round(3)}")

print("\nEx7.2: Compute row sums.")
print(f"  {np.nansum(data_m7, axis=1).round(3)}")

print("\nEx7.3: Find row and column index of the maximum value.")
flat_idx = np.nanargmax(data_m7)
row7, col7 = np.unravel_index(flat_idx, data_m7.shape)
print(f"  max at row={row7}, col={col7}, value={data_m7[row7,col7]:.3f}")

print("\nEx7.4: Compute interquartile range of the whole dataset.")
flat7 = data_m7[~np.isnan(data_m7)]
iqr7 = np.percentile(flat7, 75) - np.percentile(flat7, 25)
print(f"  IQR = {iqr7:.3f}")

print("\nEx7.5: Sort each column and return the sorted array.")
sorted_cols = np.sort(np.nan_to_num(data_m7, nan=np.nanmean(data_m7)), axis=0)
print(f"  sorted columns (first 3 rows):\n{sorted_cols[:3].round(2)}")

print("\nEx7.6: Count unique values in [3,1,4,1,5,9,2,6,5,3,5].")
arr7ex = np.array([3,1,4,1,5,9,2,6,5,3,5])
uq7, ct7 = np.unique(arr7ex, return_counts=True)
for u,c in zip(uq7,ct7):
    print(f"    {u} appears {c}×")

print("\nEx7.7: Create a histogram with 5 bins for Normal(0,1), n=10000.")
samp7 = ex_rng.normal(0,1,10000)
cnts7, edges7 = np.histogram(samp7, bins=5)
for i,(c,l,r) in enumerate(zip(cnts7, edges7[:-1], edges7[1:])):
    print(f"    [{l:.2f},{r:.2f}): {c} ({c/100:.1f}%)")

# ── MODULE 8 EXERCISES ────────────────────────────────────────────────────
print("\n" + "═"*50)
print("M8 EXERCISES — Linear Algebra")
print("═"*50)

print("\nEx8.1: Verify A @ inv(A) ≈ Identity for a 3×3 matrix.")
A8 = np.array([[2.,1.,0.],[1.,3.,1.],[0.,1.,4.]])
print(f"  A @ inv(A):\n{(A8 @ np.linalg.inv(A8)).round(10)}")

print("\nEx8.2: Solve the system: 2x+y=5, x+3y=10.")
A8b = np.array([[2.,1.],[1.,3.]])
b8b = np.array([5.,10.])
x8b = np.linalg.solve(A8b, b8b)
print(f"  x={x8b[0]:.4f}, y={x8b[1]:.4f}  verify: {A8b @ x8b}")

print("\nEx8.3: Compute L1, L2, L∞ norms of [3,4,0,-12].")
v8ex = np.array([3.,4.,0.,-12.])
print(f"  L1 = {np.linalg.norm(v8ex,1)}")
print(f"  L2 = {np.linalg.norm(v8ex,2):.4f}")
print(f"  L∞ = {np.linalg.norm(v8ex,np.inf)}")

print("\nEx8.4: Compute the rank and determinant of [[1,2,3],[4,5,6],[7,8,9]].")
M8d = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
print(f"  rank = {np.linalg.matrix_rank(M8d)}  (singular!)")
print(f"  det  = {np.linalg.det(M8d):.10f}  (≈0 → singular)")

print("\nEx8.5: Find eigenvalues of [[4,1],[2,3]].")
M8e = np.array([[4.,1.],[2.,3.]])
evals, evecs = np.linalg.eig(M8e)
print(f"  eigenvalues: {evals.round(4)}")
print(f"  eigenvectors:\n{evecs.round(4)}")

print("\nEx8.6: Perform SVD on a 4×3 matrix and reconstruct it.")
M8f = ex_rng.random((4,3))
U8, s8, Vt8 = np.linalg.svd(M8f, full_matrices=False)
M8_rec = U8 @ np.diag(s8) @ Vt8
print(f"  reconstruction error: {np.linalg.norm(M8f-M8_rec):.2e}")

print("\nEx8.7: Compute the covariance matrix of a (100,5) dataset.")
X8g = ex_rng.normal(size=(100,5))
cov8 = np.cov(X8g.T)     # (5,5)
print(f"  covariance matrix shape: {cov8.shape}")
print(f"  diagonal (variances): {np.diag(cov8).round(4)}")

print("\nEx8.8: Project 5D data onto first 2 PCA components.")
mean8 = X8g.mean(axis=0)
Xc8   = X8g - mean8
U_pca, s_pca, Vt_pca = np.linalg.svd(Xc8, full_matrices=False)
X_proj = Xc8 @ Vt_pca[:2].T
print(f"  projected shape: {X_proj.shape}  (100 samples, 2 PCs)")

# ── MODULE 9 EXERCISES ────────────────────────────────────────────────────
print("\n" + "═"*50)
print("M9 EXERCISES — Random Numbers")
print("═"*50)

rng9ex = np.random.default_rng(13)

print("\nEx9.1: Simulate 10000 fair coin flips. Count heads.")
flips = rng9ex.integers(0, 2, size=10000)
print(f"  heads: {flips.sum()} ({flips.mean()*100:.1f}%)")

print("\nEx9.2: Generate 5 random integers in [1,100] without replacement.")
no_rep = rng9ex.choice(np.arange(1,101), size=5, replace=False)
print(f"  {no_rep}")

print("\nEx9.3: Shuffle [1..10] and show first 3 remain after a second shuffle.")
arr9ex = np.arange(1,11)
shuffled = rng9ex.permutation(arr9ex)
print(f"  shuffled: {shuffled}")

print("\nEx9.4: Simulate rolling 2 dice 10000 times. Find P(sum=7).")
d1 = rng9ex.integers(1,7,10000)
d2 = rng9ex.integers(1,7,10000)
p_seven = (d1+d2==7).mean()
print(f"  P(sum=7) = {p_seven:.4f}  (exact = 1/6 ≈ {1/6:.4f})")

print("\nEx9.5: Sample 200 points from Poisson(λ=3). Verify mean≈3.")
pois = rng9ex.poisson(lam=3, size=200)
print(f"  mean={pois.mean():.4f}, var={pois.var():.4f}  (Poisson: E[X]=Var[X]=λ=3)")

print("\nEx9.6: Generate correlated 2D Gaussian (ρ=0.9).")
cov9ex = np.array([[1., 0.9],[0.9, 1.]])
samples9ex = rng9ex.multivariate_normal([0,0], cov9ex, size=5000)
r9ex = np.corrcoef(samples9ex.T)[0,1]
print(f"  empirical correlation = {r9ex:.4f}  (target: 0.9)")

# ── MODULE 10 EXERCISES ───────────────────────────────────────────────────
print("\n" + "═"*50)
print("M10 EXERCISES — Boolean Masking")
print("═"*50)

rng10ex = np.random.default_rng(17)
scores_ex = rng10ex.integers(40, 101, 30)
print(f"\n  scores (n=30): {scores_ex}")

print("\nEx10.1: Count students scoring ≥70.")
print(f"  count: {(scores_ex>=70).sum()}")

print("\nEx10.2: What is the mean score of students who passed (≥60)?")
print(f"  mean of passers: {scores_ex[scores_ex>=60].mean():.2f}")

print("\nEx10.3: Replace all failing scores (<60) with 60 (scaling up).")
sc_ex = scores_ex.copy()
sc_ex[sc_ex < 60] = 60
print(f"  after scaling: min={sc_ex.min()}, all pass: {(sc_ex>=60).all()}")

print("\nEx10.4: Find indices of scores in range [75,85].")
idx_range = np.where((scores_ex>=75) & (scores_ex<=85))[0]
print(f"  indices: {idx_range}")
print(f"  values:  {scores_ex[idx_range]}")

print("\nEx10.5: Assign letter grades (A≥90, B≥80, C≥70, D≥60, F<60).")
lgrade = np.select(
    [scores_ex>=90, scores_ex>=80, scores_ex>=70, scores_ex>=60],
    ['A',           'B',           'C',           'D'],
    default='F'
)
print(f"  grades: {lgrade}")
for g in 'ABCDF':
    print(f"    {g}: {(lgrade==g).sum()} students")

# ── MODULE 11 EXERCISES ───────────────────────────────────────────────────
print("\n" + "═"*50)
print("M11 EXERCISES — Performance")
print("═"*50)

print("\nEx11.1: Vectorize computing x² for x = arange(1,1001).")
x11 = np.arange(1, 1001)
sq = x11**2
print(f"  sum of squares 1..1000 = {sq.sum()}")

print("\nEx11.2: Compute column-wise z-scores for a (1000,10) matrix.")
M11 = ex_rng.normal(50, 10, (1000,10))
z11 = (M11 - M11.mean(axis=0)) / M11.std(axis=0)
print(f"  z-score means ≈ 0: {z11.mean(axis=0).round(10)}")

print("\nEx11.3: Compute pairwise absolute differences for [1,2,3,4,5].")
v11 = np.array([1.,2.,3.,4.,5.])
diffs = np.abs(v11[:,None] - v11[None,:])
print(f"  pairwise |xᵢ-xⱼ|:\n{diffs}")

print("\nEx11.4: Use einsum to compute trace of A@B for random (5,5) matrices.")
A11ex = ex_rng.random((5,5)); B11ex = ex_rng.random((5,5))
trace_ab = np.einsum('ij,ji->', A11ex, B11ex)
print(f"  trace(A@B) via einsum: {trace_ab:.6f}")
print(f"  trace(A@B) via numpy:  {np.trace(A11ex@B11ex):.6f}")

print("\nEx11.5: Time loop vs vectorized Euclidean norm of 1M element vector.")
big11 = ex_rng.normal(size=1_000_000)
t0=time.perf_counter(); n_loop=sum(x*x for x in big11)**0.5; t_loop=time.perf_counter()-t0
t0=time.perf_counter(); n_np=np.linalg.norm(big11); t_np=time.perf_counter()-t0
print(f"  loop: {t_loop*1000:.1f}ms, numpy: {t_np*1000:.2f}ms, speedup: {t_loop/t_np:.0f}×")

# ── FINAL CHALLENGE EXERCISES ─────────────────────────────────────────────
print("\n" + "═"*50)
print("FINAL CHALLENGE EXERCISES")
print("═"*50)

print("\nChallenge 1: Implement Softmax from scratch (vectorized, numerically stable).")
def softmax(x):
    x_shifted = x - x.max(axis=-1, keepdims=True)   # numerical stability
    e = np.exp(x_shifted)
    return e / e.sum(axis=-1, keepdims=True)
logits = np.array([[1., 2., 3.],[5., 1., 2.]])
probs  = softmax(logits)
print(f"  logits:\n{logits}")
print(f"  softmax probs:\n{probs.round(4)}")
print(f"  row sums (must be 1): {probs.sum(axis=1)}")

print("\nChallenge 2: Implement one-hot encoding for labels [0,2,1,0,3].")
labels = np.array([0,2,1,0,3])
n_cl   = 4
one_hot = np.eye(n_cl, dtype=int)[labels]
print(f"  one-hot:\n{one_hot}")

print("\nChallenge 3: Compute cosine similarity matrix for 4 vectors of dim 10.")
V = ex_rng.normal(size=(4,10))
norms_c = np.linalg.norm(V, axis=1, keepdims=True)
V_norm  = V / norms_c
cos_sim = V_norm @ V_norm.T
print(f"  cosine similarity (4×4):\n{cos_sim.round(4)}")
print(f"  diagonal (self-similarity = 1): {np.diag(cos_sim).round(10)}")

print("\nChallenge 4: Find all saddle points in a 5×5 random matrix.")
M_sad = ex_rng.integers(0, 20, (5,5))
print(f"  M:\n{M_sad}")
row_min = M_sad.min(axis=1, keepdims=True)
col_max = M_sad.max(axis=0, keepdims=True)
saddle  = (M_sad == row_min) & (M_sad == col_max)
r_sad, c_sad = np.where(saddle)
if len(r_sad):
    for r,c in zip(r_sad,c_sad):
        print(f"  Saddle point at ({r},{c}) = {M_sad[r,c]}")
else:
    print("  No saddle points found")

print("\nChallenge 5: PageRank power iteration (pure NumPy).")
# Build random transition matrix (6-node graph)
rng_pr = np.random.default_rng(7)
G = rng_pr.random((6,6))
G = G / G.sum(axis=0, keepdims=True)   # column stochastic
alpha = 0.85                            # damping factor
n_pr  = 6
M_pr  = alpha * G + (1-alpha)/n_pr * np.ones((n_pr,n_pr))
rank_pr = np.ones(n_pr) / n_pr
for _ in range(100):
    rank_pr = M_pr @ rank_pr
print(f"  PageRank scores: {rank_pr.round(4)}")
print(f"  Sum: {rank_pr.sum():.10f}  (must be 1)")

print(SEP)
print("ALL MODULES AND EXERCISES COMPLETE!")
print("NumPy concepts covered:")
concepts = [
    "ndarray internals (strides, memory layout, views vs copies)",
    "14 array creation methods",
    "dtype system (int8 to float64, overflow, NaN/inf)",
    "Basic, advanced, fancy, boolean indexing",
    "Reshaping, transpose, stacking, splitting",
    "ufuncs (trig, exp, reduce, accumulate, outer)",
    "Broadcasting (all 3 rules, real-world examples)",
    "Aggregations (nansum, percentile, histogram, sets)",
    "Linear algebra (solve, SVD, eigenvalues, norms, lstsq)",
    "Random number generation (new Generator API, all key distributions)",
    "Boolean masking, np.where, np.select, structured arrays",
    "Performance (vectorization, einsum, strides, memory layout)",
    "Real-world projects: image processing, finance, linear regression, k-NN",
    "120 exercises with solutions covering all topics",
]
for i, c in enumerate(concepts, 1):
    print(f"  {i:2d}. {c}")
print(SEP)
