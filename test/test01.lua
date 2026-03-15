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

print("--- Test 1: DType Inspection ---")
-- Test availability and string representation of all types
assert_eq(tostring(numlu.float32),    "float32",    "DType float32 available")
assert_eq(tostring(numlu.float64),    "float64",    "DType float64 available")
assert_eq(tostring(numlu.complex64),  "complex64",  "DType complex64 available")
assert_eq(tostring(numlu.complex128), "complex128", "DType complex128 available")

-- Test object identity (C-level pointer comparison)
assert_eq(numlu.float64 == numlu.float64, true, "DType identity check (self)")
assert_eq(numlu.float32 ~= numlu.float64, true, "DType identity check (distinct)")

print("\n--- Test 2: Array Creation & Properties ---")
-- Create via DType object
local arr1 = numlu.zeros(1000, numlu.float64)
assert_eq(arr1.size, 1000, "Array 1: Correct size via object creation")
assert_eq(arr1.dtype, numlu.float64, "Array 1: Correct DType via object creation")

-- Create via string lookup
local arr2 = numlu.zeros(2000, "complex128")
assert_eq(arr2.size, 2000, "Array 2: Correct size via string creation")
assert_eq(arr2.dtype, numlu.complex128, "Array 2: Correct DType via string creation")

print("\n--- Test 3: Memory Management (GC) ---")
print("Creating 10,000 temporary arrays (approx. 800 MB total workload)...")

for i = 1, 10000 do
   -- Create array with 10,000 doubles (~80 KB each)
   local temp = numlu.zeros(10000, "float64")	-- luacheck: ignore

   if i % 2500 == 0 then
      collectgarbage() -- Trigger mkl_free via __gc
      print(string.format("[PASS] Progress: %d arrays processed and cleared", i))
   end
end

print("\n--- ALL BASE TESTS PASSED ---")
