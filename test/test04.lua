local numlu = require("numlu")

-- Helper function for simple assertions
local function assert_eq(actual, expected, msg)
   if actual ~= expected then
      error(string.format("\n[FAIL] %s\nExpected: %s\nActual:   %s",
			  msg, tostring(expected), tostring(actual)))
   else
      print("[PASS] " .. msg)
   end
end

print("--- Initializing 4x4 double matrix (1.0 to 16.0) ---")
local M = numlu.zeros({4, 4}, "float64")
local count = 1
for i = 1, 4 do
   for j = 1, 4 do
      M(i, j, count)
      count = count + 1
   end
end

-- TEST 1: Basic Slicing (Sub-matrix)
-- Selecting rows 2-3 and columns 2-3 -> Result should be a 2x2 matrix
local sub = M("2:3", "2:3")
assert_eq(sub.ndims, 2, "Sub-matrix ndims remains 2")
assert_eq(sub.size, 4, "Sub-matrix total size is 4")
assert_eq(sub(1, 1), 6, "Sub-matrix element (1,1) is M(2,2) = 6")
assert_eq(sub(2, 2), 11, "Sub-matrix element (2,2) is M(3,3) = 11")

-- TEST 2: Dimensionality Collapse (NumPy-style)
-- Selecting a single row with ":" for columns -> Result should be a 1D vector
local row2 = M(2, ":")
assert_eq(row2.ndims, 1, "Row slice collapses to 1D (NumPy-style)")
assert_eq(row2.size, 4, "Row slice has 4 elements")
assert_eq(row2(1), 5, "Row 2, element 1 is 5")
assert_eq(row2(4), 8, "Row 2, element 4 is 8")

-- TEST 3: Negative Indices (Lua-stack style)
-- Using "-1" for the last row and "-2:" for the last two columns
local last_part = M("-1", "-2:")
assert_eq(last_part.ndims, 1, "Last row part collapses to 1D")
assert_eq(last_part.size, 2, "Last row part has 2 elements")
assert_eq(last_part(1), 15, "M(-1, -2) is 15")
assert_eq(last_part(2), 16, "M(-1, -1) is 16")

-- TEST 4: Slicing with Steps
-- Every second row, first two columns
local stepped = M("1:4:2", "1:2") -- Rows 1 and 3
assert_eq(stepped.ndims, 2, "Stepped slice is 2D")
assert_eq(stepped(1, 1), 1, "Stepped (1,1) is M(1,1) = 1")
assert_eq(stepped(2, 1), 9, "Stepped (2,1) is M(3,1) = 9")

-- TEST 5: Recursive Slicing (Slice of a Slice)
-- 1. Get rows 2-4 -> 2. Get the first row of that view -> 3. Get last 2 elements
local nested = M("2:4", ":")("1", "3:4")
assert_eq(nested.ndims, 1, "Nested slice dimensionality check")
assert_eq(nested.size, 2, "Nested slice size check")
assert_eq(nested(1), 7, "Nested value check: M(2,3) = 7")

-- TEST 6: Memory Safety (GC Protection)
print("--- Testing GC Protection (Views keep original alive) ---")
local function stress_gc()
   local Big = numlu.zeros({100, 100}, "float64")
   Big(1, 1, 999.0)
   local view = Big(1, ":") -- View of the first row
   Big = nil -- luacheck: ignore
   collectgarbage() -- Force GC
   assert_eq(view(1), 999.0, "View data persists after original is nil")
end
stress_gc()

-- TEST 7: Bounds Checking on Views
local ok = pcall(function() sub(3, 1) end) -- sub is 2x2
assert_eq(ok, false, "Accessing sub(3,1) should fail (out of bounds)")

local ok2 = pcall(function() M("1:10", ":") end)
assert_eq(ok2, false, "Creating slice 1:10 on 4x4 matrix should fail")

-- TEST 8: Full Slice with ":"
local full = M(":", ":")
assert_eq(full.size, 16, "Full slice size check")
assert_eq(full(4, 4), 16, "Full slice last element check")

-- TEST 9: Writing to a View (Reflects in Original)
local view = M(1, "1:2") -- First row, first two columns
view(1, 99.0) -- This is M(1, 1)
view(2, 88.0) -- This is M(1, 2)
assert_eq(M(1, 1), 99.0, "Writing to view affects M(1,1)")
assert_eq(M(1, 2), 88.0, "Writing to view affects M(1,2)")

-- TEST 10: Partial Indexing (Implied ":" for missing dimensions)
print("--- Testing Partial Indexing ---")
-- M(2) should be equivalent to M(2, ":")
local partial = M(2)
assert_eq(partial.ndims, 1, "Partial indexing collapses indexed dim")
assert_eq(partial.size, 4, "Partial indexing keeps full remaining dim")
assert_eq(partial(1), 5, "M(2)(1) is M(2,1) = 5")
assert_eq(partial(4), 8, "M(2)(4) is M(2,4) = 8")

-- TEST 11: N-Dimensionality (3D Array Test)
print("--- Testing 3D Array & Slicing ---")
local T = numlu.zeros({2, 3, 4}, "float32")
T(1, 1, 1, 100.0)
T(2, 1, 1, 200.0)

-- Partial indexing on 3D -> 2D view
local layer1 = T(1)
assert_eq(layer1.ndims, 2, "T(1) results in 2D view")
assert_eq(layer1(1, 1), 100.0, "Value in layer1(1,1) is T(1,1,1)")

-- Mixing scalar, slice and partial indexing: T(2, "1:2") -> T(2, "1:2", ":")
local sub_3d = T(2, "1:2")
assert_eq(sub_3d.ndims, 2, "T(2, '1:2') results in 2D view (dim 1 collapsed)")
assert_eq(sub_3d.size, 8, "Sub-view size is 2x4 = 8")
assert_eq(sub_3d(1, 1), 200.0, "Value in sub_3d(1,1) is T(2,1,1)")

-- TEST 12: Setter Guard (Should fail with partial indices)
print("--- Testing Setter Guards ---")
local ok_set = pcall(function() T(1, 1, 999.0) end)
assert_eq(ok_set, false, "Setter with missing indices must fail (expected 3, got 2)")

print("\n--- ALL SLICING TESTS PASSED ---")
