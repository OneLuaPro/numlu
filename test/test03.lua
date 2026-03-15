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

print("\n--- ALL BASE TESTS PASSED ---")
