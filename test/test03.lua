local numlu = require("numlu")
local complex = require("lcomplex")

-- Helper function for consistent [PASS] / [FAIL] output
local function assert_eq(actual, expected, msg)
   if actual ~= expected then
      error(string.format("\n[FAIL] %s\nExpected: %s\nActual:   %s",
			  msg, tostring(expected), tostring(actual)))
   else
      print("[PASS] " .. msg)
   end
end

-- Helper to ensure a specific call fails (NumPy-conformity check)
local function assert_error(fn, msg)
   local ok, err = pcall(fn)
   if ok then
      error(string.format("\n[FAIL] %s (Should have thrown an error but didn't)", msg))
   else
      -- Optional: print(string.format("[DEBUG] Caught expected error: %s", err))
      print("[PASS] " .. msg .. " (Caught expected error)")
   end
end

print("--- Test 1: Multi-Dimensional Creation ---")
local matrix = numlu.zeros({3, 4}, numlu.float64)
assert_eq(matrix.ndims, 2, "Matrix ndims is 2")
assert_eq(matrix.size, 12, "Matrix size is 12")
local s = matrix.shape
assert_eq(s[1], 3, "Shape dimension 1 is 3")
assert_eq(s[2], 4, "Shape dimension 2 is 4")

print("\n--- Test 2: Methods vs Operators (Flat) ---")
matrix:at(1, 10.5)
matrix[12] = 99.9
assert_eq(matrix[1], 10.5, "Flat access at [1] via operator")
assert_eq(matrix[12], 99.9, "Flat access at [12] via operator")

print("\n--- Test 3: GC Stability ---")
for i = 1, 5000 do
   local tensor = numlu.zeros({10, 10, 10}, "float32") -- luacheck: ignore
   if i % 2500 == 0 then
      collectgarbage()
      print(string.format("[PASS] Progress: %d tensors processed and collected", i))
   end
end

print("\n--- Test 4: Length Operator (#) ---")
local mat = numlu.zeros({50, 20}, numlu.float64)
assert_eq(#mat, 50, "Length operator (#mat) returns first dimension")

print("\n--- Test 5: Multi-Dimensional Call (Get/Set) ---")
local m2d = numlu.zeros({3, 3}, numlu.float64)
m2d(1, 1, 1.1)
m2d(2, 1, 2.1)
m2d(3, 3, 3.3)

assert_eq(m2d(1, 1), 1.1, "Multi-dim access (1,1)")
assert_eq(m2d(2, 1), 2.1, "Multi-dim access (2,1)")
assert_eq(m2d(3, 3), 3.3, "Multi-dim access (3,3)")

-- Bounds Check Test
local ok = pcall(function() m2d(4, 1, 0.0) end)
assert_eq(ok, false, "Bounds check: Accessing (4,1) on 3x3 fails as expected")

print("\n--- Test 6: Complex Multi-Dim Call ---")
local cmat = numlu.zeros({2, 2}, "complex128")
local z = complex.new(5, -2)
cmat(1, 2, z)
cmat(2, 2, 10.5) -- Set real number to complex array

local val12 = cmat(1, 2)
local val22 = cmat(2, 2)
assert_eq(val12:real(), 5.0, "Complex (1,2) real part")
assert_eq(val12:imag(), -2.0, "Complex (1,2) imag part")
assert_eq(val22:real(), 10.5, "Complex (2,2) auto-conversion from real")
assert_eq(val22:imag(), 0.0, "Complex (2,2) imag part is zero")

print("\n--- Test 7: Scalar (0-D) Arrays (NumPy-Conformity) ---")
local scalar = numlu.zeros({}, "float64")
assert_eq(scalar.ndims, 0, "Scalar ndims is 0")
assert_eq(scalar.size, 1, "Scalar size is 1")
assert_eq(#scalar.shape, 0, "Scalar shape is an empty table")

-- 1. Test: Length operator must fail on 0-D (NumPy len() behavior)
assert_error(function() return #scalar end, "Length operator (#) on 0-D array must fail")

-- 2. Test: Indexing must fail on 0-D (NumPy indexing behavior)
assert_error(function() return scalar[1] end, "Indexing scalar via [] must fail")

-- 3. Test: Call indexing must fail on 0-D
assert_error(function() return scalar(1) end, "Calling scalar(1) must fail")

-- 4. Test: Correct value access (Scalars can be accessed via :at(1) for testing)
scalar:at(1, 42.5)
assert_eq(scalar:at(1), 42.5, "Scalar access via :at(1)")

-- 5. ToString representation
local s_str = tostring(scalar)
assert_eq(s_str, "numlu.ndarray<float64>({})", "Scalar tostring representation")

-- 6. Complex Scalar (NumPy-conform access via method)
local c_scalar = numlu.zeros({}, "complex128")
c_scalar:at(1, complex.new(1, 2))
local c_val = c_scalar:at(1)
assert_eq(c_val:real(), 1.0, "Complex scalar real part")
assert_eq(c_val:imag(), 2.0, "Complex scalar imag part")

-- 7. Verify indexing error for complex scalar as well
assert_error(function() return c_scalar[1] end, "Indexing complex scalar via [] must fail")

print("\n--- ALL BASE TESTS PASSED ---")
