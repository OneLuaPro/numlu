local numlu = require("numlu")

print("--- Test 1: DType Inspection ---")
print("Available Types:")
print("  - " .. tostring(numlu.float32))
print("  - " .. tostring(numlu.float64))
print("  - " .. tostring(numlu.complex64))
print("  - " .. tostring(numlu.complex128))

-- Test object identity (Pointer comparison in C)
assert(numlu.float64 == numlu.float64, "DType identity check failed")
print("DType Identity: OK")


print("\n--- Test 2: Array Creation & Properties ---")
local size = 1000

-- Create via DType object
local arr1 = numlu.zeros(size, numlu.float64)
print("Array 1 (via object):  Size = " .. arr1.size .. ", Type = " .. tostring(arr1.dtype))

-- Create via string lookup
local arr2 = numlu.zeros(2000, "complex128")
print("Array 2 (via string):  Size = " .. arr2.size .. ", Type = " .. tostring(arr2.dtype))

assert(arr1.size == 1000, "Size mismatch for Array 1")
assert(arr1.dtype == numlu.float64, "DType mismatch for Array 1")
print("Properties Check: OK")


print("\n--- Test 3: Memory Management (GC) ---")
print("Creating 10,000 temporary arrays (approx. 800 MB total)...")

for i = 1, 10000 do
    -- Create array with 10,000 doubles (~80 KB each)
    local temp = numlu.zeros(10000, "float64")
    
    if i % 2500 == 0 then
        collectgarbage() -- Force mkl_free via __gc
        print("  Progress: " .. i .. " arrays processed...")
    end
end

print("Memory Test: OK (No crashes)")
print("\n--- All Base Tests Passed! ---")
