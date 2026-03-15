local complex = require("lcomplex")
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

print("--- Testing lcomplex Interoperability ---")

-- 1. Create a complex array (128-bit / double complex)
local a = numlu.zeros(10, "complex128")
local z = complex.new(1, 2)

-- 2. Test: Set lcomplex object into ndarray
a:at(1, z)
local val = a:at(1)

-- 3. Verify real and imaginary parts using lcomplex methods
assert_eq(val:real(), 1.0, "Complex real part (via lcomplex method)")
assert_eq(val:imag(), 2.0, "Complex imag part (via lcomplex method)")

-- 4. Test: Automatic real-to-complex conversion
a:at(2, 42.5)
local val2 = a:at(2)
assert_eq(val2:real(), 42.5, "Auto-conversion real part")
assert_eq(val2:imag(), 0.0, "Auto-conversion imag part is zero")

-- 5. Test: String representation (__tostring)
local str = tostring(val)
local is_valid = (str == "1+2i" or str == "1.0+2.0i")
assert_eq(is_valid, true, "lcomplex string representation (Actual: " .. str .. ")")

print("\n--- COMPLEX INTEROP TESTS PASSED ---")
