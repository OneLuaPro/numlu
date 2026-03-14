local numlu = require("numlu")

print("--- Test 1: Multi-Dimensional Creation ---")
local matrix = numlu.zeros({3, 4}, numlu.float64)
print("Matrix properties:")
print("  ndims:  " .. matrix.ndims)
print("  size:   " .. matrix.size)
local s = matrix.shape
print("  shape:  {" .. s[1] .. ", " .. s[2] .. "}")
assert(s[1] == 3 and s[2] == 4, "Shape mismatch!")

print("\n--- Test 2: Methods vs Operators ---")
-- Classic method
matrix:at(1, 10.5)
-- New: Unified __newindex (flat)
matrix[12] = 99.9

-- New: Unified __index (flat)
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

print("\n--- Test 5: Multi-Dimensional Access (Call) ---")
local m2d = numlu.zeros({3, 3}, numlu.float64)

-- Set values via flat index for setup
m2d[1] = 1.1 -- Position (1,1)
m2d[2] = 1.2 -- Position (1,2)
m2d[4] = 2.1 -- Position (2,1)
m2d[9] = 3.3 -- Position (3,3)

-- Test multi-dim getter via __call
print("  Access (1,1): " .. m2d(1, 1))
print("  Access (1,2): " .. m2d(1, 2))
print("  Access (2,1): " .. m2d(2, 1))
print("  Access (3,3): " .. m2d(3, 3))

assert(m2d(2, 1) == 2.1, "Multi-dim call access failed")

print("\n--- All Tests Passed! ---")
