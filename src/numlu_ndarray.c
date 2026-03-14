#include "numlu_ndarray.h"
#include <mkl.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

/* Metatable name used by the lcomplex library */
#define LCOMPLEX_METATABLE "complex number"

/* Garbage collector: frees MKL-allocated memory and metadata */
static int l_ndarray_gc(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  
  /* 1. Free main data only if we are the owner (is_view == 0) */
  if (!arr->is_view && arr->data) {
    mkl_free(arr->data);
    arr->data = NULL;
  }
  
  /* 2. Always free metadata arrays (shape and strides) */
  if (arr->shape) {
    mkl_free(arr->shape);
    arr->shape = NULL;
  }
  if (arr->strides) {
    mkl_free(arr->strides);
    arr->strides = NULL;
  }
  
  return 0;
}

/* Unified __index: handles properties (arr.size) and numeric access (arr[i]) */
static int l_ndarray_index(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");

  /* CASE A: Numeric Index -> arr[i] */
  if (lua_isnumber(L, 2)) {
    lua_Integer idx = lua_tointeger(L, 2);
    
    /* Bounds check (1-based) */
    if (idx < 1 || idx > (lua_Integer)arr->size) {
      return luaL_error(L, "numlu: index %d out of bounds (size %d)", (int)idx, (int)arr->size);
    }
    size_t i = (size_t)idx - 1;

    switch (arr->dtype->id) {
    case NUMLU_TYPE_F32:
      {
	lua_pushnumber(L, ((float*)arr->data)[i]);
	return 1;
      }
    case NUMLU_TYPE_F64:
      {
        lua_pushnumber(L, ((double*)arr->data)[i]);
        return 1;
      }
    case NUMLU_TYPE_C64:
    case NUMLU_TYPE_C128:
      {
	double complex* res = lua_newuserdatauv(L, sizeof(double complex), 0);
	luaL_setmetatable(L, LCOMPLEX_METATABLE);
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

  /* CASE B: Property String -> arr.size, arr.shape, etc. */
  const char* key = luaL_checkstring(L, 2);
  if (strcmp(key, "dtype") == 0) {
    lua_getiuservalue(L, 1, 1);
    return 1;
  }
  if (strcmp(key, "size") == 0) {
    lua_pushinteger(L, (lua_Integer)arr->size);
    return 1;
  }
  if (strcmp(key, "ndims") == 0) {
    lua_pushinteger(L, (lua_Integer)arr->ndims);
    return 1;
  }
  if (strcmp(key, "shape") == 0) {
    lua_createtable(L, arr->ndims, 0);
    for (int i = 0; i < arr->ndims; i++) {
      lua_pushinteger(L, (lua_Integer)arr->shape[i]);
      lua_rawseti(L, -2, i + 1);
    }
    return 1;
  }

  /* CASE C: Method Fallback */
  if (luaL_getmetafield(L, 1, key)) {
    return 1;
  }

  return 0;
}

/* Unified __newindex: handles numeric assignment (arr[i] = val) */
static int l_ndarray_newindex(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");

  if (lua_isnumber(L, 2)) {
    lua_Integer idx = lua_tointeger(L, 2);
    if (idx < 1 || idx > (lua_Integer)arr->size) {
      return luaL_error(L, "numlu: index %d out of bounds", (int)idx);
    }
    size_t i = (size_t)idx - 1;

    if (arr->dtype->id == NUMLU_TYPE_F32 || arr->dtype->id == NUMLU_TYPE_F64) {
      double val = luaL_checknumber(L, 3);
      if (arr->dtype->id == NUMLU_TYPE_F32)
        ((float*)arr->data)[i] = (float)val;
      else
        ((double*)arr->data)[i] = val;
    }
    else {
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

  return luaL_error(L, "numlu: only numeric indexing is supported for assignment");
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
	/* Lua 5.5: 0 User Values for the complex result */
	double complex* res = lua_newuserdatauv(L, sizeof(double complex), 0);
	luaL_setmetatable(L, LCOMPLEX_METATABLE);

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

/* Length operator (#arr): returns the size of the first dimension */
static int l_ndarray_len(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  
  if (arr->ndims > 0) {
    lua_pushinteger(L, (lua_Integer)arr->shape[0]);
  }
  else {
    lua_pushinteger(L, 0);
  }
  return 1;
}

/* Constructor: numlu.zeros(size|{shape}, dtype) */
int l_ndarray_new(lua_State* L) {
  int ndims = 0;
  size_t total_size = 1;
  size_t* shape = NULL;
  size_t* strides = NULL;

  /* 1. Parse Shape: Accept number (1D) or table (nD) */
  if (lua_isnumber(L, 1)) {
    ndims = 1;
    lua_Integer s = luaL_checkinteger(L, 1);
    if (s <= 0) return luaL_error(L, "numlu: size must be positive");
    
    shape = (size_t*)mkl_malloc(sizeof(size_t), 64);
    strides = (size_t*)mkl_malloc(sizeof(size_t), 64);
    
    shape[0] = (size_t)s;
    strides[0] = 1;
    total_size = shape[0];
  } 
  else if (lua_istable(L, 1)) {
    ndims = (int)lua_rawlen(L, 1);
    if (ndims == 0) return luaL_error(L, "numlu: shape table cannot be empty");
    
    shape = (size_t*)mkl_malloc(ndims * sizeof(size_t), 64);
    strides = (size_t*)mkl_malloc(ndims * sizeof(size_t), 64);
    
    /* Read dimensions and calculate total flat size */
    for (int i = 1; i <= ndims; i++) {
      lua_rawgeti(L, 1, i);
      shape[i-1] = (size_t)luaL_checkinteger(L, -1);
      total_size *= shape[i-1];
      lua_pop(L, 1);
    }
    
    /* Calculate Strides (Row-Major / C-Style) */
    size_t current_stride = 1;
    for (int i = ndims - 1; i >= 0; i--) {
      strides[i] = current_stride;
      current_stride *= shape[i];
    }
  } 
  else {
    return luaL_typeerror(L, 1, "number or table");
  }

  /* 2. Get DType and create Userdata (Lua 5.5: 1 User Value slot) */
  const numlu_dtype_info* dtype = numlu_dtype_check(L, 2);
  numlu_ndarray* arr = lua_newuserdatauv(L, sizeof(numlu_ndarray), 1);
  luaL_setmetatable(L, "numlu.ndarray");

  /* Anchor DType to the ndarray */
  lua_pushvalue(L, 2);
  lua_setiuservalue(L, -2, 1);

  /* 3. Allocate and zero-initialize MKL memory */
  arr->data = mkl_malloc(total_size * dtype->itemsize, 64);
  if (!arr->data) {
    mkl_free(shape);
    mkl_free(strides);
    return luaL_error(L, "numlu: mkl_malloc failed for data");
  }
  memset(arr->data, 0, total_size * dtype->itemsize);

  /* 4. Fill struct fields */
  arr->size = total_size;
  arr->dtype = dtype;
  arr->ndims = ndims;
  arr->shape = shape;
  arr->strides = strides;
  arr->offset = 0;
  arr->is_view = 0; /* Owner of the memory */
  
  return 1;
}

/* Multi-dimensional access: arr(i, j, ...) */
static int l_ndarray_call(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  int n_args = lua_gettop(L) - 1; /* First arg is the array itself */

  if (n_args != arr->ndims) {
    return luaL_error(L, "numlu: expected %d indices, got %d", arr->ndims, n_args);
  }

  /* Calculate flat index using strides */
  size_t flat_idx = 0;
  for (int i = 0; i < arr->ndims; i++) {
    lua_Integer idx = luaL_checkinteger(L, i + 2);
    
    /* 1-based bounds check for each dimension */
    if (idx < 1 || idx > (lua_Integer)arr->shape[i]) {
      return luaL_error(L, "numlu: index %d out of bounds for dimension %d (size %d)", 
                        (int)idx, i + 1, (int)arr->shape[i]);
    }
    
    /* Accumulate offset: (index - 1) * stride */
    flat_idx += (size_t)(idx - 1) * arr->strides[i];
  }

  /* For now, we only implement the GETTER in __call */
  /* (Setter logic can be added by checking if an extra value is passed) */
  switch (arr->dtype->id) {
  case NUMLU_TYPE_F32:
    {
      lua_pushnumber(L, ((float*)arr->data)[flat_idx]);
      return 1;
    }
  case NUMLU_TYPE_F64:
    {
      lua_pushnumber(L, ((double*)arr->data)[flat_idx]);
      return 1;
    }
  case NUMLU_TYPE_C64:
  case NUMLU_TYPE_C128:
    {
      double complex* res = lua_newuserdatauv(L, sizeof(double complex), 0);
      luaL_setmetatable(L, LCOMPLEX_METATABLE);
      if (arr->dtype->id == NUMLU_TYPE_C64) {
	float complex c = ((float complex*)arr->data)[flat_idx];
	*res = (double)crealf(c) + (double)cimagf(c) * I;
        }
      else {
	*res = ((double complex*)arr->data)[flat_idx];
      }
      return 1;
    }
  }
  return 0;
}

void numlu_ndarray_register(lua_State* L) {
  luaL_newmetatable(L, "numlu.ndarray");
  
  lua_pushcfunction(L, l_ndarray_gc);
  lua_setfield(L, -2, "__gc");
  
  lua_pushcfunction(L, l_ndarray_index);
  lua_setfield(L, -2, "__index");

  /* Register the length operator */
  lua_pushcfunction(L, l_ndarray_len);
  lua_setfield(L, -2, "__len");

  /* Register methods inside the metatable */
  lua_pushcfunction(L, l_ndarray_at);
  lua_setfield(L, -2, "at");

  /* Register flat indexer */ 
  lua_pushcfunction(L, l_ndarray_newindex);
  lua_setfield(L, -2, "__newindex");

  /* Register multi-dimensional indexer */
  lua_pushcfunction(L, l_ndarray_call);
  lua_setfield(L, -2, "__call");
  
  lua_pop(L, 1);
}
