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

print("--- Testing Reshape & Auto-Dimension logic ---")

-- 1. Create a base array (4x5 = 20 elements)
local a = numlu.zeros({4, 5}, "float64")
assert_eq(a.size, 20, "Initial size calculation")
assert_eq(a.ndims, 2, "Initial dimension count")

-- 2. Test: Basic reshape to 2x10
local b = a:reshape({2, 10})
assert_eq(b.ndims, 2, "Reshape ndims (2x10)")
assert_eq(b.shape[1], 2, "Reshape shape[1]")
assert_eq(b.shape[2], 10, "Reshape shape[2]")
assert_eq(b.size, 20, "Size remains constant after reshape")

-- 3. Test: Auto-Dimension (-1) at the end
local c = a:reshape({2, 2, -1})
assert_eq(c.ndims, 3, "Auto-dim (-1) creates 3rd dimension")
assert_eq(c.shape[3], 5, "Auto-dim calculation (20 / 2 / 2 = 5)")

-- 4. Test: Auto-Dimension (-1) at the beginning
local d = a:reshape({-1, 10})
assert_eq(d.shape[1], 2, "Auto-dim calculation at start (20 / 10 = 2)")

-- 5. Test: Reshape to 1D (scalar argument)
local e = a:reshape(20)
assert_eq(e.ndims, 1, "Reshape to 1D via scalar")
assert_eq(e.shape[1], 20, "1D shape size")

-- 6. Test: Automatic Copy for Non-Contiguous Reshape
-- A column slice is NOT contiguous. Now, reshape should SUCCEED by copying.
local col_slice = a(":", 2)
assert_eq(col_slice.is_contiguous, false, "Slice contiguity check (Column)")

-- This used to fail, now it creates a fresh, contiguous 1D array
local reshaped_col = col_slice:reshape(4)
assert_eq(reshaped_col.size, 4, "Reshape succeeds on non-contiguous via copy")
assert_eq(reshaped_col.is_contiguous, true, "Reshaped copy is now contiguous")
assert_eq(reshaped_col.is_view, false, "Reshaped non-contiguous is an owner (copy)")

-- 7. Test: Size Mismatch Error
local status2, err2 = pcall(function() a:reshape({3, 3}) end)
local size_error = (not status2 and err2:find("mismatch") ~= nil)
assert_eq(size_error, true, "Safety check: Reshape fails on size mismatch")

-- 8. Test: Identity Reshape (Reshape to same shape)
local f = a:reshape({4, 5})
assert_eq(f.shape[1], 4, "Identity reshape dim 1")
assert_eq(f.shape[2], 5, "Identity reshape dim 2")
assert_eq(f.size, 20, "Identity size check")

-- 9. Test: Nested Reshaping (View-Chain Stability)
-- Test if reshaping a reshaped array works (anchoring check)
local g = a:reshape({2, 10}):reshape({5, 4})
assert_eq(g.ndims, 2, "Nested reshape dimension count")
assert_eq(g.shape[1], 5, "Nested reshape shape[1]")
assert_eq(g.shape[2], 4, "Nested reshape shape[2]")
assert_eq(g.size, 20, "Nested reshape size stability")

-- 10. Test: Safety check for multiple -1 placeholders
local status3, err3 = pcall(function() a:reshape({-1, -1}) end)
local multi_auto_error = (not status3 and err3:find("only one %-1 allowed") ~= nil)
assert_eq(multi_auto_error, true, "Safety check: Multiple -1 placeholders fail")

-- 11. Test: Data Integrity after Copy-Reshape
-- Fill column 2 with {1.1, 2.2, 3.3, 4.4}
for i = 1, 4 do a(i, 2, i * 1.1) end 
local col_view = a(":", 2)
local col_copy = col_view:reshape(4)
assert_eq(col_copy[1], 1.1, "Data integrity: Element 1")
assert_eq(col_copy[4], 4.4, "Data integrity: Element 4")

-- 12. Test: Independence of Copy
-- Changing the copy must NOT affect the original
col_copy[1] = 99.9
assert_eq(a(1, 2), 1.1, "Independence: Original remains unchanged after copy-write")

-- 13. Test: Stepped Slice Reshape (Complex Strides)
local vec = numlu.zeros(10, "float64")
for i = 1, 10 do vec[i] = i end
local stepped = vec("1:10:2") -- {1, 3, 5, 7, 9}
local final = stepped:reshape({5, 1})
assert_eq(final.size, 5, "Stepped reshape size")
assert_eq(final(5, 1), 9, "Stepped reshape data check")

print("\n--- RESHAPE & AUTO-DIM TESTS PASSED ---")
