#include "numlu_dtype.h"
#include "numlu_ndarray.h"

LUA_API int luaopen_numlu(lua_State* L) {
  numlu_dtype_register(L);
  numlu_ndarray_register(L);

  lua_newtable(L);

  // API: numlu.zeros(n, dtype)
  lua_pushcfunction(L, l_ndarray_new);
  lua_setfield(L, -2, "zeros");

  // DTypes: numlu.float64 etc.
  numlu_push_dtype(L, &NUMLU_DTYPE_F32);  lua_setfield(L, -2, "float32");
  numlu_push_dtype(L, &NUMLU_DTYPE_F64);  lua_setfield(L, -2, "float64");
  numlu_push_dtype(L, &NUMLU_DTYPE_C64);  lua_setfield(L, -2, "complex64");
  numlu_push_dtype(L, &NUMLU_DTYPE_C128); lua_setfield(L, -2, "complex128");

  return 1;
}
