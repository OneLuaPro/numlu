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

print("--- Testing Squeeze Functionality ---")

-- TEST 1: Squeeze All
local A = numlu.zeros({1, 5, 1, 2}, "float32")
A(1, 3, 1, 2, 99.5) -- Set a value in the 4D space

local B = A:squeeze()
assert_eq(B.ndims, 2, "Squeeze all: 1x5x1x2 becomes 2D (5x2)")
assert_eq(B.size, 10, "Squeeze all: size remains 10")
assert_eq(B(3, 2), 99.5, "Squeeze all: data integrity check")

-- TEST 2: Squeeze Specific Axis
local C = A:squeeze(1) -- Squeeze the first dimension
assert_eq(C.ndims, 3, "Squeeze axis 1: 1x5x1x2 becomes 3D (5x1x2)")
assert_eq(C(3, 1, 2), 99.5, "Squeeze axis: data integrity check")

-- TEST 3: Squeeze Axis Fail (Size > 1)
local ok = pcall(function() A:squeeze(2) end) -- Axis 2 has size 5
assert_eq(ok, false, "Squeezing axis with size > 1 should fail")

-- TEST 4: Squeeze Axis Fail (Out of Bounds)
local ok2 = pcall(function() A:squeeze(10) end)
assert_eq(ok2, false, "Squeezing non-existent axis should fail")

-- TEST 5: Total Squeeze (1x1x1 -> 1D)
-- Note: We collapse to 1D of size 1 as a safety measure
local D = numlu.zeros({1, 1, 1}, "float64")
D(1, 1, 1, 7.7)
local E = D:squeeze()
assert_eq(E.ndims, 1, "Total squeeze of 1x1x1 collapses to 1D")
assert_eq(E(1), 7.7, "Total squeeze: data integrity check")

-- TEST 6: Squeeze on a View (Explicit vs. Implicit Collapse)
local M = numlu.zeros({10, 10}, "float64")
M(5, 5, 55.5)
local implicit = M(5, ":")
assert_eq(implicit.ndims, 1, "M(5, ':') already collapses to 1D implicitly")
local sub = M("5:5", ":")
assert_eq(sub.ndims, 2, "Slice '5:5' keeps 2D shape {1, 10}")
local squeezed_sub = sub:squeeze()
assert_eq(squeezed_sub.ndims, 1, "Squeezing 1x10 view becomes 1D")
assert_eq(squeezed_sub(5), 55.5, "View squeeze data integrity check")

print("\n--- ALL SQUEEZE TESTS PASSED ---")
