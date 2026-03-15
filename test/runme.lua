-- runme.lua: Dynamic LFS-based Test Runner for numlu
local lfs = require("lfs")

local function get_test_files()
   local files = {}
   -- Pattern matches any "test" followed by digits and ".lua"
   local pattern = "^test%d+%.lua$"

   for file in lfs.dir(".") do
      if file:match(pattern) then
	 table.insert(files, file)
      end
   end
   table.sort(files) -- Ensures test01 -> test02 sequence
   return files
end

local tests = get_test_files()

print("====================================================")
print("                numlu Test Suite                    ")
print(string.format("  Detected %d test files in current directory", #tests))
print("====================================================")

local passed, failed = 0, 0
local start_time = os.clock()

for _, test_file in ipairs(tests) do
   print("\n>> RUNNING: " .. test_file)
   print("----------------------------------------------------")

   local chunk, err = loadfile(test_file)
   if not chunk then
      print("[ERROR] Could not load " .. test_file .. ": " .. err)
      failed = failed + 1
   else
      local success = pcall(chunk)
      if success then
	 passed = passed + 1
      else
	 -- We don't need to print the error here if the test
	 -- already uses our assert_eq (which prints [FAIL])
	 failed = failed + 1
      end
   end
end

local duration = os.clock() - start_time

print("\n====================================================")
print("                  TEST SUMMARY                      ")
print("====================================================")
print(string.format("  Passed: %d | Failed: %d | Time: %.3fs", passed, failed, duration))
print("====================================================")

os.exit(failed == 0 and 0 or 1)
