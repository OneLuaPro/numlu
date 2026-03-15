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

print("--- Testing ndarray:__tostring() ---")

-- 1. Test standard 2D matrix
local A = numlu.zeros({2, 3}, "float64")
assert_eq(tostring(A), "numlu.ndarray<float64>({2, 3})", "Standard 2D Matrix string")

-- 2. Test 1D array
local v = numlu.zeros(10, "float32")
assert_eq(tostring(v), "numlu.ndarray<float32>({10})", "Standard 1D Array string")

-- 3. Test complex DType
local c = numlu.zeros({4, 4}, "complex128")
assert_eq(tostring(c), "numlu.ndarray<complex128>({4, 4})", "Complex 2D Matrix string")

-- 4. Test View/Slice (Automatic dimension collapse)
-- Selecting row 1 of a 2x3 matrix results in a 1D view of size 3
local row = A(1, ":")
assert_eq(tostring(row), "numlu.ndarray<float64>({3})", "1D View string (collapsed dim)")

-- 5. Test High-Dimensionality
local T = numlu.zeros({2, 3, 4, 5}, "float32")
assert_eq(tostring(T), "numlu.ndarray<float32>({2, 3, 4, 5})", "4D Tensor string")

-- 6. Test Squeezed View
local S = numlu.zeros({1, 5, 1}, "float64")
local squeezed = S:squeeze()
assert_eq(tostring(squeezed), "numlu.ndarray<float64>({5})", "Squeezed view string")

print("\n--- TOSTRING TESTS PASSED ---")
