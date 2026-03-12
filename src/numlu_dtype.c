#include "numlu_dtype.h"
#include <string.h>

const numlu_dtype_info NUMLU_DTYPE_F32  = { NUMLU_TYPE_F32,  4,  "float32" };
const numlu_dtype_info NUMLU_DTYPE_F64  = { NUMLU_TYPE_F64,  8,  "float64" };
const numlu_dtype_info NUMLU_DTYPE_C64  = { NUMLU_TYPE_C64,  8,  "complex64" };
const numlu_dtype_info NUMLU_DTYPE_C128 = { NUMLU_TYPE_C128, 16, "complex128" };

const numlu_dtype_info* numlu_dtype_from_string(const char* name) {
  if (strcmp(name, "float32") == 0)    return &NUMLU_DTYPE_F32;
  if (strcmp(name, "float64") == 0)    return &NUMLU_DTYPE_F64;
  if (strcmp(name, "complex64") == 0)  return &NUMLU_DTYPE_C64;
  if (strcmp(name, "complex128") == 0) return &NUMLU_DTYPE_C128;
  return NULL;
}

const numlu_dtype_info* numlu_dtype_check(lua_State* L, int arg) {
  if (lua_isstring(L, arg)) {
    const numlu_dtype_info* info = numlu_dtype_from_string(lua_tostring(L, arg));
    if (!info) luaL_error(L, "numlu: unbekannter dtype '%s'", lua_tostring(L, arg));
    return info;
  }
  const numlu_dtype_info** udata = luaL_checkudata(L, arg, "numlu.dtype");
  return *udata;
}

void numlu_push_dtype(lua_State* L, const numlu_dtype_info* info) {
  const numlu_dtype_info** udata = lua_newuserdata(L, sizeof(numlu_dtype_info*));
  *udata = info;
  luaL_setmetatable(L, "numlu.dtype");
}

static int l_dtype_tostring(lua_State* L) {
  const numlu_dtype_info* info = *(const numlu_dtype_info**)luaL_checkudata(L, 1, "numlu.dtype");
  lua_pushstring(L, info->name);
  return 1;
}

static int l_dtype_eq(lua_State* L) {
    const numlu_dtype_info* a = *(const numlu_dtype_info**)luaL_checkudata(L, 1, "numlu.dtype");
    const numlu_dtype_info* b = *(const numlu_dtype_info**)luaL_checkudata(L, 2, "numlu.dtype");
    lua_pushboolean(L, a == b);
    return 1;
}

void numlu_dtype_register(lua_State* L) {
  luaL_newmetatable(L, "numlu.dtype");
  lua_pushcfunction(L, l_dtype_tostring);
  lua_setfield(L, -2, "__tostring");
  lua_pushcfunction(L, l_dtype_eq);
  lua_setfield(L, -2, "__eq");
  lua_pop(L, 1);
}
