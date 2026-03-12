local complex = require("lcomplex")
local numlu = require("numlu")

local a = numlu.zeros(10, "complex128")
local z = complex.new(1, 2)

a:at(1, z)        -- Setze lcomplex Objekt
local val = a:at(1)
print(val:real()) -- 1.0 (Methode von lcomplex!)
