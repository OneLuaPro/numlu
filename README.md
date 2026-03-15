# numlu (Numerical Lua)

A high-performance numerical library for **Lua 5.5**, powered by the **Intel Math Kernel Library (MKL)**.

## Goal
The goal of `numlu` is to provide a NumPy-like experience within the Lua ecosystem. It combines the elegance of Lua's syntax with the raw computational power of Intel MKL. Unlike standard Lua tables, `numlu` arrays are memory-aligned (64-byte), type-strict, and support multi-dimensional indexing with efficient strides.

## Core Features
- **Engine:** Intel MKL for all memory and mathematical operations.
- **Memory:** 64-byte aligned allocations (`mkl_malloc`) for AVX-512 optimization.
- **Advanced Slicing Engine:** Supports NumPy-style slicing (e.g., `"1:5:2"`) with zero-copy views.
- **Partial Indexing:** Intelligently handles missing dimensions (e.g., `A(1)` returns a view of the first row).
- **Lua 5.5 Optimized:** Full use of User Value slots for metadata anchoring and the new Userdata API.
- **Interoperability:** Seamless integration with the `lcomplex` library.
- **Indexing:** 1-based indexing (Lua/Fortran style) for consistency with the Lua ecosystem.

## API Comparison: NumPy vs. numlu


| Feature | NumPy (Python) | numlu (Lua 5.5) |
| :--- | :--- | :--- |
| **1D Creation** | `np.zeros(100)` | `numlu.zeros(100)` |
| **nD Creation** | `np.zeros((3, 4))` | `numlu.zeros({3, 4})` |
| **Data Types** | `dtype=np.float64` | `numlu.float64` |
| **Shape Access** | `a.shape` (tuple) | `a.shape` (table) |
| **Dimensions** | `a.ndim` | `a.ndims` |
| **Flat Access** | `a[i]` (0-based) | `a[i]` (1-based) |
| **Multi-Dim Access** | `a[i, j]` | `a(i, j)` |
| **Multi-Dim Setter** | `a[i, j] = 1.0` | `a(i, j, 1.0)` |
| **Slicing** | `a[1:5, :]` | `a("1:5", ":")` |
| **Partial Indexing** | `a[1]` | `a(1)` |
| **Length Operator** | `len(a)` | `#a` |
| **Flat Setter** | `a[i] = 1.0` | `a[i] = 1.0` |

> [!NOTE]
> While NumPy uses square brackets `[]` for multi-dimensional access, Lua's syntax limits `[]` to a single argument. To ensure maximum performance and flexible slicing without creating temporary objects, `numlu` uses the function call syntax `()` for multi-index and slicing operations.

## Usage Example

```lua
local numlu = require("numlu")

-- Create a 4x4 matrix of 64-bit floats
local mat = numlu.zeros({4, 4}, numlu.float64)

-- Multi-dimensional access
mat(2, 2, 100.5)

-- Slicing: Get a 2x2 sub-matrix (rows 2-3, cols 2-3)
-- This creates a VIEW, not a copy. Original memory is shared.
local sub = mat("2:3", "2:3")
print(sub(1, 1)) -- Output: 100.5

-- Partial Indexing: Get the entire first row
local first_row = mat(1)
print(first_row.ndims) -- 1 (Dimension collapsed)

-- Negative indexing (last element of the matrix)
mat(4, 4, 99.0)
print(mat("-1", "-1")) -- 99.0

```

## Current Status
- N-Dimensional Array Container
- Strided Memory Layout (Row-Major)
- Memory Management (GC + MKL)
- Basic Indexing (Flat, Multi-Dim, Property)
- Multi-Dimensional Getter abd Setter via __call
- Vectorized Math Operations (MKL VML)
- Slicing and Views
