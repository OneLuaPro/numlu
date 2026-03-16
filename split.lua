local argparse = require("argparse")
local lfs = require("lfs")

-- Configure Argument Parser
local parser = argparse("split", "Splits a file into chunks based on a line count.")
parser:argument("input", "The source file to be split.")

-- Add optional argument for line count
parser:option("-l --lines", "Number of lines per split file.")
   :default("220")
   :convert(tonumber)

local args = parser:parse()

local input_file = args.input
local lines_per_file = args.lines

-- Check if the file exists using LuaFileSystem
local attr = lfs.attributes(input_file)
if not attr or attr.mode ~= "file" then
   print("Error: File '" .. input_file .. "' not found or is not a regular file.")
   os.exit(1)
end

-- Extract base name (removes common extensions)
local basename = input_file:gsub("%.%w+$", "")

local f_in = io.open(input_file, "r")
local file_count = 1
local line_count = 0
local f_out = nil

-- Iterate through the file line by line
for line in f_in:lines() do
   -- Open a new file every X lines
   if line_count % lines_per_file == 0 then
      if f_out then f_out:close() end
      -- Create filename with format: basename-###.txt
      local filename = string.format("%s-%03d.txt", basename, file_count)
      f_out = io.open(filename, "w")
      file_count = file_count + 1
   end

   f_out:write(line .. "\n")
   line_count = line_count + 1
end

-- Cleanup resources
if f_out then f_out:close() end
f_in:close()

print(string.format("Done! Created %d files with %d lines each (last file may be shorter).",
		    file_count - 1, lines_per_file))
