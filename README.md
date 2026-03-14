# numlu (Numerical Lua)

A high-performance numerical library for **Lua 5.5**, powered by the **Intel Math Kernel Library (MKL)**.

## Goal
The goal of `numlu` is to provide a NumPy-like experience within the Lua ecosystem. It combines the elegance of Lua's syntax with the raw computational power of Intel MKL. Unlike standard Lua tables, `numlu` arrays are memory-aligned (64-byte), type-strict, and support multi-dimensional indexing with efficient strides.

## Core Features
- **Engine:** Intel MKL for all memory and mathematical operations.
- **Memory:** 64-byte aligned allocations (`mkl_malloc`) for AVX-512 optimization.
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
| **Length Operator** | `len(a)` | `#a` |
| **Flat Setter** | `a[i] = 1.0` | `a[i] = 1.0` |

> [!NOTE]
>
> While NumPy uses square brackets `[]` for multi-dimensional access, Lua's syntax limits `[]` to a single argument. To ensure maximum performance without creating temporary objects, `numlu` uses the function call syntax `()` for multi-index operations.

## Usage Example

```lua
local numlu = require("numlu")

-- Create a 3x3 matrix of 64-bit floats
local mat = numlu.zeros({3, 3}, numlu.float64)

-- Access properties
print("Dimensions:", mat.ndims) -- 2
print("Total Size:", mat.size)  -- 9
print("Rows:", #mat)            -- 3 (Length operator)

-- Different indexing styles
mat[1] = 10.5        -- Flat indexing (1-based)
mat(2, 2) = 20.0     -- Multi-dimensional access (High performance)

local s = mat.shape
print("Shape:", s[1], "x", s[2]) -- 3 x 3
```

## Current Status
- N-Dimensional Array Container
- Strided Memory Layout (Row-Major)
- Memory Management (GC + MKL)
- Basic Indexing (Flat, Multi-Dim, Property)
- Multi-Dimensional Getter abd Setter via __call
- Vectorized Math Operations (MKL VML)
- Slicing and Views
