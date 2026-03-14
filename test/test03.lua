local numlu = require("numlu")
local complex = require("lcomplex")

print("--- Test 1: Multi-Dimensional Creation ---")
local matrix = numlu.zeros({3, 4}, numlu.float64)
print("Matrix properties:")
print("  ndims:  " .. matrix.ndims)
print("  size:   " .. matrix.size)
local s = matrix.shape
print("  shape:  {" .. s[1] .. ", " .. s[2] .. "}")
assert(s[1] == 3 and s[2] == 4, "Shape mismatch!")

print("\n--- Test 2: Methods vs Operators (Flat) ---")
matrix:at(1, 10.5)
matrix[12] = 99.9
print("  Value at [1]:  " .. matrix[1])
print("  Value at [12]: " .. matrix[12])
assert(matrix[1] == 10.5 and matrix[12] == 99.9, "Index operator failed")

print("\n--- Test 3: GC Stability ---")
for i = 1, 5000 do
   local tensor = numlu.zeros({10, 10, 10}, "float32")
   if i % 2500 == 0 then
      collectgarbage()
      print("  Progress: " .. i .. " tensors processed.")
   end
end

print("\n--- Test 4: Length Operator (#) ---")
local mat = numlu.zeros({50, 20}, numlu.float64)
print("  2D Matrix (50x20) #mat: " .. #mat)
assert(#mat == 50, "Length operator failed")

print("\n--- Test 5: Multi-Dimensional Call (Get/Set) ---")
local m2d = numlu.zeros({3, 3}, numlu.float64)

-- New: Set values via multi-dim call: mat(r, c, val)
m2d(1, 1, 1.1)
m2d(1, 2, 1.2)
m2d(2, 1, 2.1)
m2d(3, 3, 3.3)

print("  Access (1,1): " .. m2d(1, 1))
print("  Access (2,1): " .. m2d(2, 1))
print("  Access (3,3): " .. m2d(3, 3))

assert(m2d(2, 1) == 2.1, "Multi-dim call setter/getter failed")

-- Bounds Check Test
local ok, err = pcall(function() m2d(4, 1, 0) end)
print("  Bounds Check (expected error): " .. (ok and "Failed" or "OK (" .. err:match("out of bounds") .. ")"))

print("\n--- Test 6: Complex Multi-Dim Call ---")
local cmat = numlu.zeros({2, 2}, "complex128")
local z = complex.new(5, -2)

cmat(1, 2, z)
cmat(2, 2, 10.5) -- Set real number to complex array

print("  Complex (1,2): " .. tostring(cmat(1, 2)))
print("  Complex (2,2): " .. tostring(cmat(2, 2)))

assert(cmat(1, 2):real() == 5 and cmat(1, 2):imag() == -2, "Complex multi-dim call failed")

print("\n--- All Tests Passed! ---")
