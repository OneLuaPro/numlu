#include "numlu_ndarray.h"
#include <mkl.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

/* Metatable name used by the lcomplex library */
#define LCOMPLEX_METATABLE "complex number"

/* Garbage collector: frees MKL-allocated memory */
static int l_ndarray_gc(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  if (arr->data) mkl_free(arr->data);
  return 0;
}

/* property getter for .dtype and .size */
static int l_ndarray_index(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  const char* key = luaL_checkstring(L, 2);

  if (strcmp(key, "dtype") == 0) {
    numlu_push_dtype(L, arr->dtype);
    return 1;
  }
  if (strcmp(key, "size") == 0) {
    lua_pushinteger(L, (lua_Integer)arr->size);
    return 1;
  }
  
  /* Fallback to methods in the metatable (like :at) */
  luaL_getmetatable(L, "numlu.ndarray");
  lua_pushvalue(L, 2);
  lua_gettable(L, -2);
  return 1;
}

/* Getter/Setter: arr:at(index, [value]) */
static int l_ndarray_at(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  lua_Integer idx = luaL_checkinteger(L, 2);

  /* Lua-style 1-based indexing check */
  if (idx < 1 || idx > (lua_Integer)arr->size) {
    return luaL_error(L, "numlu: index %d out of bounds (size %d)", (int)idx, (int)arr->size);
  }
  size_t i = (size_t)idx - 1;

  int n_args = lua_gettop(L);

  /* GETTER MODE: arr:at(index) */
  if (n_args == 2) {
    switch (arr->dtype->id) {
    case NUMLU_TYPE_F32:
      lua_pushnumber(L, ((float*)arr->data)[i]);
      return 1;
    case NUMLU_TYPE_F64:
      lua_pushnumber(L, ((double*)arr->data)[i]);
      return 1;
    case NUMLU_TYPE_C64:
    case NUMLU_TYPE_C128:
      {
	/* Create lcomplex-compatible userdata */
	/* We always return double complex (C128) to Lua for precision */
	double complex* res = lua_newuserdata(L, sizeof(double complex));
                
	luaL_getmetatable(L, LCOMPLEX_METATABLE);
	if (lua_isnil(L, -1)) {
	  return luaL_error(L, "numlu: lcomplex not loaded (meta '" LCOMPLEX_METATABLE "' missing)");
	}
	lua_setmetatable(L, -2);

	if (arr->dtype->id == NUMLU_TYPE_C64) {
	  float complex c = ((float complex*)arr->data)[i];
	  *res = (double)crealf(c) + (double)cimagf(c) * I;
	}
	else {
	  *res = ((double complex*)arr->data)[i];
	}
	return 1;
      }
    }
  }

  /* SETTER MODE: arr:at(index, value) */
  if (n_args == 3) {
    if (arr->dtype->id == NUMLU_TYPE_F32 || arr->dtype->id == NUMLU_TYPE_F64) {
      double val = luaL_checknumber(L, 3);
      if (arr->dtype->id == NUMLU_TYPE_F32)
	((float*)arr->data)[i] = (float)val;
      else
	((double*)arr->data)[i] = val;
    }
    else {
      /* Complex Setter: accepts Lua number or lcomplex userdata */
      double complex val;
      if (lua_isnumber(L, 3)) {
	val = lua_tonumber(L, 3) + 0.0 * I;
      }
      else {
	double complex* z = luaL_checkudata(L, 3, LCOMPLEX_METATABLE);
	val = *z;
      }

      if (arr->dtype->id == NUMLU_TYPE_C64) {
	((float complex*)arr->data)[i] = (float)creal(val) + (float)cimag(val) * I;
      }
      else {
	((double complex*)arr->data)[i] = val;
      }
    }
    return 0;
  }
  
  return 0;
}

/* Constructor: numlu.zeros(size, dtype) */
int l_ndarray_new(lua_State* L) {
  size_t size = (size_t)luaL_checkinteger(L, 1);
  const numlu_dtype_info* dtype = numlu_dtype_check(L, 2);

  numlu_ndarray* arr = lua_newuserdata(L, sizeof(numlu_ndarray));
  luaL_setmetatable(L, "numlu.ndarray");

  arr->size = size;
  arr->dtype = dtype;
  /* 64-byte alignment for optimal MKL/AVX performance */
  arr->data = mkl_malloc(size * dtype->itemsize, 64);
    
  if (!arr->data) return luaL_error(L, "numlu: mkl_malloc failed");
  memset(arr->data, 0, size * dtype->itemsize);

  return 1;
}

void numlu_ndarray_register(lua_State* L) {
  luaL_newmetatable(L, "numlu.ndarray");
  
  lua_pushcfunction(L, l_ndarray_gc);
  lua_setfield(L, -2, "__gc");
  
  lua_pushcfunction(L, l_ndarray_index);
  lua_setfield(L, -2, "__index");

  /* Register methods inside the metatable */
  lua_pushcfunction(L, l_ndarray_at);
  lua_setfield(L, -2, "at");
    
  lua_pop(L, 1);
}
