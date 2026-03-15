#include "numlu_ndarray.h"
#include <mkl.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <stdbool.h>

/* Forward declaration */
static int parse_slice_string(const char *s, long max_len, long *start, long *stop, long *step);

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
  
  /* 2. Always free metadata arrays (shape and strides). */
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

/* 
 * Helper function to parse a slicing string like "start:stop:step"
 * Returns SLICE_RANGE if it contains a colon, SLICE_SCALAR otherwise.
 */
static slice_type parse_slice_string(const char *s, long max_len, long *start, long *stop, long *step) {
  /* Default values: full range with step 1 */
  *start = 1;
  *stop = max_len;
  *step = 1;

  /* Check for range indicator (colon) before parsing */
  int is_range = (strchr(s, ':') != NULL);

  /* Special case: full dimension. Keep 1-based (start=1, stop=max_len). */
  if (strcmp(s, ":") == 0) {
    return SLICE_RANGE;
  }

  /* sscanf_s returns the number of successfully filled variables */
  int count = sscanf_s(s, "%ld:%ld:%ld", start, stop, step);
  if (count < 1) return SLICE_INVALID;

  /* Prevent Division by Zero */
  if (count >= 3 && *step == 0) {
    // We use SLICE_INVALID here, the caller l_ndarray_call will throw the luaL_error
    return SLICE_INVALID;
  }

  /* FIXME: Reverse slicing (negative steps) is not yet supported.
   * This requires negative strides and adjusted offset logic. */
  if (count >= 3 && *step < 0) {
    // Return a special marker or handle via error in the caller.
    // For now, let's keep it simple:
    return SLICE_INVALID;
  }

  /* Handle negative indices for 'start' (Lua-stack style: -1 is last) */
  if (*start < 0) {
    *start = max_len + *start + 1;
  }
    
  /* Handle negative indices for 'stop' */
  if (count >= 2 && *stop < 0) {
    *stop = max_len + *stop + 1;
  }

  /* Bounds checking to prevent memory access errors */
  if (*start < 1) *start = 1;
  if (*start > max_len) *start = max_len;
  if (*start < 1 || *start > max_len || *stop > max_len) {
    /* Indices must be within 1 and max_len
     * We could return SLICE_INVALID here, but a direct error message 
     * helps the user find the bug in their Lua code immediately. */
    return SLICE_INVALID; 
  }
  if (*stop < 0) *stop = 0; /* Resulting slice will be empty, which is okay */

  return is_range ? SLICE_RANGE : SLICE_SCALAR;
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

/* 
 * Multi-dimensional access: 
 * - GET: arr(i, j, ...) -> returns scalar or view (if slicing/partial indexing)
 * - SET: arr(i, j, ..., val) -> sets scalar value
 */
static int l_ndarray_call(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  int total_args = lua_gettop(L) - 1; /* Total arguments provided in Lua */

  /* 
   * Determine Mode:
   * A setter call must provide exactly ndims + 1 (the value) arguments.
   * Partial indexing is only allowed for getters (creating views).
   */
  bool is_setter = (total_args == (int)arr->ndims + 1);

  /* Guard against too many arguments */
  if (!is_setter && total_args > (int)arr->ndims) {
    return luaL_error(L, "numlu: too many indices (%d) for %d-D array", 
                      total_args, (int)arr->ndims);
  }

  /* Setters require all dimensions to be explicitly indexed */
  if (is_setter && total_args != (int)arr->ndims + 1) {
    return luaL_error(L, "numlu: setter requires exactly %d indices plus a value", (int)arr->ndims);
  }

  /* 
   * Check if we should enter Slicing Mode to create a View.
   * Triggered if: 
   * 1. At least one argument is a string (e.g., "1:5")
   * 2. Fewer arguments than dimensions are provided (Partial Indexing)
   */
  int is_slicing = 0;
  if (!is_setter) {
    if (total_args < (int)arr->ndims) {
      is_slicing = 1;
    }
    else {
      for (int i = 0; i < total_args; i++) {
        if (lua_type(L, i + 2) == LUA_TSTRING) {
          is_slicing = 1;
          break;
        }
      }
    }
  }

  if (is_slicing) {
    /* Step 1: Calculate resulting dimensionality (NumPy-style) */
    int view_ndims = 0;
    for (int i = 0; i < (int)arr->ndims; i++) {
      if (i < total_args) {
        if (lua_type(L, i + 2) == LUA_TSTRING) {
          long st, sp, sk;
          if (parse_slice_string(lua_tostring(L, i + 2), arr->shape[i], &st, &sp, &sk) == SLICE_RANGE) {
            view_ndims++;
          }
        }
        /* Scalar integers collapse the dimension */
      }
      else {
        /* Missing indices in partial indexing are treated as ":" (Full Range) */
        view_ndims++;
      }
    }

    /* Step 2: Initialize the new View object */
    numlu_ndarray *view = (numlu_ndarray *)lua_newuserdatauv(L, sizeof(numlu_ndarray), 1);
    luaL_getmetatable(L, "numlu.ndarray");
    lua_setmetatable(L, -2);

    /* Anchor original array in uservalue to prevent GC collection while view exists */
    lua_pushvalue(L, 1); 
    lua_setiuservalue(L, -2, 1);

    view->ndims = view_ndims;
    view->data = arr->data; 
    view->dtype = arr->dtype;
    view->is_view = 1;
    view->offset = arr->offset;
    view->size = 1;
    
    /* Allocate metadata for the view */
    view->shape = (size_t*)mkl_malloc(view->ndims * sizeof(size_t), 64);
    view->strides = (size_t*)mkl_malloc(view->ndims * sizeof(size_t), 64);

    if (!view->shape || !view->strides) {
      return luaL_error(L, "numlu: mkl_malloc failed for view metadata");
    }

    /* Step 3: Populate view metadata and calculate offset */
    int target_dim = 0;
    for (int i = 0; i < (int)arr->ndims; i++) {
      long start, stop, step;
      slice_type stype = SLICE_INVALID;

      if (i < total_args) {
        if (lua_type(L, i + 2) == LUA_TSTRING) {
          stype = parse_slice_string(lua_tostring(L, i + 2), arr->shape[i], &start, &stop, &step);
          if (stype == SLICE_INVALID) {
            return luaL_error(L, "numlu: slice out of bounds for dimension %d (max %d)", 
                              i + 1, (int)arr->shape[i]);
          }
        }
	else {
          start = (long)luaL_checkinteger(L, i + 2);
          stype = SLICE_SCALAR;
          stop = start;
          step = 1;
        }
      }
      else {
        /* Implied full range for partial indexing */
        start = 1;
        stop = (long)arr->shape[i];
        step = 1;
        stype = SLICE_RANGE;
      }

      long start_idx = start - 1; /* 0-based internal logic */
      
      if (stype == SLICE_RANGE) {
        view->offset += (size_t)start_idx * arr->strides[i];
        view->strides[target_dim] = arr->strides[i] * (size_t)step;

        long dim_size = (stop - start) / step + 1;
        view->shape[target_dim] = (size_t)(dim_size < 0 ? 0 : dim_size);
        view->size *= view->shape[target_dim];
        target_dim++;
      } 
      else { /* SLICE_SCALAR */
        if (start < 1 || start > (long)arr->shape[i]) {
          return luaL_error(L, "numlu: index %ld out of bounds for dim %d", start, i + 1);
        }
        view->offset += (size_t)start_idx * arr->strides[i];
      } 
    }
    return 1;
  }

  /* --- Scalar Access Mode (Standard Getter/Setter) --- */

  /* Calculate flat index using strides */
  size_t flat_idx = (size_t)arr->offset;
  for (int i = 0; i < (int)arr->ndims; i++) {
    lua_Integer idx = luaL_checkinteger(L, i + 2);
    
    if (idx < 1 || idx > (lua_Integer)arr->shape[i]) {
      return luaL_error(L, "numlu: index %d out of bounds for dimension %d (size %d)", 
                        (int)idx, i + 1, (int)arr->shape[i]);
    }
    flat_idx += (size_t)(idx - 1) * arr->strides[i];
  }

  if (is_setter) {
    int val_idx = lua_gettop(L); 

    if (arr->dtype->id == NUMLU_TYPE_F32 || arr->dtype->id == NUMLU_TYPE_F64) {
      double val = luaL_checknumber(L, val_idx);
      if (arr->dtype->id == NUMLU_TYPE_F32)
        ((float*)arr->data)[flat_idx] = (float)val;
      else
        ((double*)arr->data)[flat_idx] = val;
    } 
    else {
      double complex val;
      if (lua_isnumber(L, val_idx)) {
	/* Clean conversion from real to complex */
        val = (double)lua_tonumber(L, val_idx) + 0.0 * I;
      }
      else {
        double complex* z = luaL_checkudata(L, val_idx, LCOMPLEX_METATABLE);
        val = *z;
      }

      if (arr->dtype->id == NUMLU_TYPE_C64) {
	/* Explicitly cast to float complex for MKL C64 compatibility */
	float complex fz = (float)creal(val) + (float)cimag(val) * I;
        ((float complex*)arr->data)[flat_idx] = fz;
      }
      else {
	/* C128 (double complex) matches Lua/lcomplex directly */
        ((double complex*)arr->data)[flat_idx] = val;
      }
    }
    return 0;
  }

  /* GETTER MODE */
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
