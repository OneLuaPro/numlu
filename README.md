# numlu (Numerical Lua)

A high-performance numerical library for **Lua 5.5**, powered by the **Intel Math Kernel Library (MKL)**.

## Goal
The goal of `numlu` is to provide a NumPy-like experience within the Lua ecosystem. It combines the elegance of Lua's syntax with the raw computational power of Intel MKL. Unlike standard Lua tables, `numlu` arrays are memory-aligned (64-byte), type-strict, and support multi-dimensional indexing with efficient strides.

## Core Features
- **Engine:** Intel MKL for all memory and mathematical operations.
- **Memory:** 64-byte aligned allocations (`mkl_malloc`) for AVX-512 optimization.
- **Advanced Slicing Engine:** Supports NumPy-style slicing (e.g., `"1:5:2"`) with zero-copy views.
- **Efficient Reshaping:** Change array dimensions with zero-copy views, including NumPy-style auto-calculation (`-1`).
- **Partial Indexing:** Intelligently handles missing dimensions (e.g., `A(1)` returns a view of the first row).
- **Lua 5.5 Optimized:** Full use of User Value slots for metadata anchoring and the new Userdata API.
- **Interoperability:** Seamless integration with the `lcomplex` library.
- **Indexing:** 1-based indexing (Lua/Fortran style) for consistency with the Lua ecosystem.

## API Comparison: NumPy vs. numlu

| Feature | NumPy (Python) | numlu (Lua 5.5) |
| :--- | :--- | :--- |
| 1D Creation | `np.zeros(100)` | `numlu.zeros(100)` |
| nD Creation | `np.zeros((3, 4))` | `numlu.zeros({3, 4})` |
| Data Types | `dtype=np.float64` | `numlu.float64` |
| Shape Access | `a.shape` (tuple) | `a.shape` (table) |
| Dimensions | `a.ndim` | `a.ndims` |
| Flat Access | `a[i]` (0-based) | `a[i]` (1-based) |
| Multi-Dim Access | `a[i, j]` | `a(i, j)` |
| Multi-Dim Setter | `a[i, j] = 1.0` | `a(i, j, 1.0)` |
| Slicing | `a[1:5, :]` | `a("1:5", ":")` |
| Reshape | `a.reshape((2, 5))` | `a:reshape({2, 5})` |
| Auto-Reshape | `a.reshape((2, -1))` | `a:reshape({2, -1})` |
| Squeeze (all) | `a.squeeze()` | `a:squeeze()` |
| Squeeze (axis) | `a.squeeze(axis=0)` | `a:squeeze(1)` |
| Length Operator | `len(a)` | `#a` |

> [!NOTE]
> While NumPy uses square brackets `[]` for multi-dimensional access, Lua's syntax limits `[]` to a single argument. To ensure maximum performance and flexible slicing without creating temporary objects, `numlu` uses the function call syntax `()` for multi-index and slicing operations.

## Usage Example

```lua
local numlu = require("numlu")

-- Create a 4x5 matrix (20 elements)
local mat = numlu.zeros({4, 5}, "float64")

-- Reshape: Change to 2x10 without copying data
local b = mat:reshape({2, 10})

-- Auto-Dimension: Let numlu calculate the missing dimension
-- Here: 20 / (2 * 2) = 5. Result is a 2x2x5 tensor.
local c = mat:reshape({2, 2, -1})
print(table.concat(c.shape, "x")) -- Output: 2x2x5

-- Slicing: Get a sub-matrix. This creates a VIEW.
local sub = mat("2:3", "2:3")

-- Safety: Reshape only works on contiguous memory.
-- Slices are often non-contiguous, so numlu protects you from corruption:
local status, err = pcall(function() sub:reshape(4) end)
print(err) -- "numlu: reshape without copy only possible for contiguous arrays"

-- Squeeze: Remove dimensions of size 1
local T = numlu.zeros({1, 3, 1}, "float64")
local s = T:squeeze()
print(s.ndims) -- 1
```

## Current Status
- N-Dimensional Array Container
- Strided Memory Layout (Row-Major)
- Memory Management (GC + MKL Alignment)
- Basic Indexing (Flat, Multi-Dim, Property)
- Multi-Dimensional Getter and Setter via __call
- Slicing and Zero-Copy Views
- Shape Manipulation: Support for reshape() with -1 logic and squeeze().
- Vectorized Math Operations (MKL VML) - In Progress
