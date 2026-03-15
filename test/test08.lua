local numlu = require("numlu")

local function assert_eq(actual, expected, msg)
   if actual ~= expected then
      error(string.format("\n[FAIL] %s\nExpected: %s\nActual:   %s",
			  msg, tostring(expected), tostring(actual)))
   else
      print("[PASS] " .. msg)
   end
end

print("--- Testing Contiguity Check ---")

-- 1. Full Matrix (Freshly allocated)
local A = numlu.zeros({10, 10}, "float64")
assert_eq(A.is_contiguous, true, "Full matrix is contiguous")

-- 2. Row Slice (Should be contiguous)
-- Row 2 contains elements 11-20, which lie together in memory.
local row2 = A(2, ":")
assert_eq(row2.is_contiguous, true, "Row slice remains contiguous")

-- 3. Column Slice (Should NOT be contiguous)
-- Column 2 contains elements 2, 12, 22... which have gaps of 10 in memory.
local col2 = A(":", 2)
assert_eq(col2.is_contiguous, false, "Column slice is NOT contiguous")

-- 4. Stepped Slice (Should NOT be contiguous)
-- Every second element of a row has gaps.
local stepped = A(1, "1:10:2")
assert_eq(stepped.is_contiguous, false, "Stepped slice is NOT contiguous")

print("\n--- CONTIGUITY TESTS PASSED ---")
