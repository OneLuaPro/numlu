local lfs = require("lfs")

local output_md = "code_status.md"
local output_pdf = "code_status.pdf"
local base_path = lfs.currentdir()

-- Konfiguration der Kapitel und Suchpfade
local config = {
   { title = "## Sub-directory `src`",   folder = "src" },
   { title = "## Sub-directory `test`",  folder = "test" },
   { title = "## `CMakeLists.txt`", file = "CMakeLists.txt" }
}

-- Hilfsfunktion: Erkennt die Sprache für den Markdown-Codeblock
local function get_lang(filename)
   if filename:match("%.c$") or filename:match("%.h$") then return "c" end
   if filename:match("%.lua$") then return "lua" end
   if filename:match("CMakeLists%.txt") then return "cmake" end
   return ""
end

-- Hilfsfunktion: Liest den Dateiinhalt
local function read_file(path)
   local f = io.open(path, "r")
   if not f then return "[Fehler: Datei konnte nicht gelesen werden]" end
   local content = f:read("*all")
   f:close()
   return content
end

-- Markdown-Datei initialisieren
local timestamp = os.date("%d.%m.%Y %H:%M:%S")
local out = io.open(output_md, "w")
out:write("# numlu Code Status\n\n")
out:write("**Generated on:** " .. timestamp .. "\n\n")

-- Hauptlogik: Verzeichnisse und Dateien abarbeiten
for _, section in ipairs(config) do
   out:write(section.title .. "\n\n")

   if section.folder then
      -- Ordner durchsuchen (src, test)
      local path = base_path .. "\\" .. section.folder
      if lfs.attributes(path, "mode") == "directory" then
	 for file in lfs.dir(path) do
	    -- Filter: Keine Verzeichnisse, kein "." / "..", keine Tilde-Dateien
	    if file ~= "." and file ~= ".." and not file:match("~$") then
	       local full_path = path .. "\\" .. file
	       if lfs.attributes(full_path, "mode") == "file" then
		  out:write("### File: " .. file .. "\n\n")
		  out:write("```" .. get_lang(file) .. "\n")
		  out:write(read_file(full_path))
		  out:write("\n```\n\n")
	       end
	    end
	 end
      end
   elseif section.file then
      -- Einzeldatei (CMakeLists.txt)
      local full_path = base_path .. "\\" .. section.file
      if lfs.attributes(full_path, "mode") == "file" then
	 out:write("### Datei: " .. section.file .. "\n\n")
	 out:write("```" .. get_lang(section.file) .. "\n")
	 out:write(read_file(full_path))
	 out:write("\n```\n\n")
      end
   end
end

out:close()
print("Markdown erstellt: " .. output_md)

-- -- Pandoc Aufruf
-- print("Konvertiere zu PDF via Pandoc...")
-- local success = os.execute('pandoc "' .. output_md .. '" -o "' .. output_pdf .. '"')

-- if success then
--    print("Erfolgreich: " .. output_pdf)
-- else
--    print("Fehler bei der PDF-Erstellung. Ist Pandoc/LaTeX installiert?")
-- end
