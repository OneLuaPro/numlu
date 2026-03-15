local numlu = require("numlu")

-- Helper function for consistent [PASS] / [FAIL] output
local function assert_eq(actual, expected, msg)
   if actual ~= expected then
      error(string.format("\n[FAIL] %s\nExpected: %s\nActual:   %s",
			  msg, tostring(expected), tostring(actual)))
   else
      print("[PASS] " .. msg)
   end
end

print("--- Testing Strided Flat Access (Paket 2) ---")

-- 1. Create a 3x3 matrix and fill it with numbers 1 to 9
-- Memory layout (Row-Major): 1, 2, 3, 4, 5, 6, 7, 8, 9
local M = numlu.zeros({3, 3}, "float64")
local count = 1
for i = 1, 3 do
   for j = 1, 3 do
      M(i, j, count)
      count = count + 1
   end
end

-- 2. Create a View of the second column (Column 2)
-- Logical elements should be: 2, 5, 8
-- Physical memory indices are: 1, 4, 7 (0-based)
local col2 = M(":", 2)
assert_eq(col2.ndims, 1, "Column view is 1D")
assert_eq(col2.size, 3, "Column view has 3 elements")

-- 3. TEST: Flat Getter on View
-- Before Paket 2: col2[1] would have returned 2, col2[2] would have returned 3 (WRONG)
-- After Paket 2: col2[2] must return 5
assert_eq(col2[1], 2.0, "col2[1] is M(1,2) = 2")
assert_eq(col2[2], 5.0, "col2[2] is M(2,2) = 5 (Strided access)")
assert_eq(col2[3], 8.0, "col2[3] is M(3,2) = 8 (Strided access)")

-- 4. TEST: Flat Setter on View
-- Changing the middle element of the column view should affect the original matrix
col2[2] = 500.0
assert_eq(M(2, 2), 500.0, "Setting col2[2] affects M(2,2) correctly")

-- 5. TEST: Complex Slicing with Steps
-- Matrix 4x4
local M4 = numlu.zeros({4, 4}, "float32")
for i = 1, 16 do M4[i] = i end

-- Every second row, every second column (2x2 View)
-- Elements: M4(1,1)=1, M4(1,3)=3, M4(3,1)=9, M4(3,3)=11
local stepped = M4("1:4:2", "1:4:2")
assert_eq(stepped.ndims, 2, "Stepped view is 2D")
assert_eq(stepped.size, 4, "Stepped view has 4 elements")

-- Logical flat access on 2D view: 1=top-left, 4=bottom-right
assert_eq(stepped[1], 1.0, "stepped[1] is 1")
assert_eq(stepped[2], 3.0, "stepped[2] is 3")
assert_eq(stepped[3], 9.0, "stepped[3] is 9")
assert_eq(stepped[4], 11.0, "stepped[4] is 11")

print("\n--- STRIDED ACCESS TESTS PASSED ---")
