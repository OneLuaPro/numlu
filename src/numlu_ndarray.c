#include "numlu_ndarray.h"
#include <mkl.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <stdbool.h>

/* Forward declaration */
static int parse_slice_string(const char *s, size_t max_len,
			      long long *start, long long *stop, long long *step);

/* Metatable name used by the lcomplex library */
#define LCOMPLEX_METATABLE "complex number"

/* 
 * Helper: Allocates a GC-managed memory block for metadata (shape or strides).
 * Supports 0-D arrays by ensuring at least 1 byte is allocated for Lua,
 * while returning NULL to the C-struct to signify a scalar/0-D state.
 */
static void* alloc_metadata_buffer(lua_State* L, int ndims, int uv_slot) {
  /* Ensure we always allocate at least a dummy byte to keep Lua's GC happy,
   * as 0-byte userdata behavior can be implementation-defined. */
  size_t alloc_size = (ndims > 0) ? (ndims * sizeof(size_t)) : 1;
  
  /* 1. Create a new "raw" userdata block for the metadata.
   * This is pushed onto the stack (at index -1). */
  void* ptr = lua_newuserdatauv(L, alloc_size, 0); 
  
  /* 2. Anchor this buffer to the ndarray (currently at index -2).
   * lua_setiuservalue pops the buffer from the stack and stores it in the slot. */
  lua_setiuservalue(L, -2, uv_slot); 
  
  /* For 0-D arrays, we return NULL so the C-code knows there is no 
   * shape/stride array to iterate over. */
  return (ndims > 0) ? ptr : NULL;
}

/*
 * Helper: Calculates C-contiguous (Row-Major) strides for a given shape.
 * Strides are calculated from the last dimension to the first.
 */
static void calc_strides_row_major(int ndims, const size_t* shape, size_t* strides) {
  if (ndims == 0) return;

  size_t current_stride = 1;
  /* Calculate from last dimension to first (Right-to-Left) */
  for (int i = ndims - 1; i >= 0; i--) {
    strides[i] = current_stride;
    current_stride *= shape[i];
  }
}

/* 
 * Helper: Checks if the array is C-contiguous (Row-Major).
 * Returns 1 if contiguous, 0 otherwise.
 */
static int numlu_is_contiguous(numlu_ndarray* arr) {
  if (arr->ndims == 0) return 1;
    
  size_t expected_stride = 1;
  /* Check from last dimension to first (Right-to-Left) */
  for (int i = arr->ndims - 1; i >= 0; i--) {
    if (arr->strides[i] != expected_stride) {
      return 0; /* Gap or non-standard layout detected */
    }
    expected_stride *= arr->shape[i];
  }
  return 1;
}

/* 
 * Helper: Converts a logical flat index (0-based) to the actual 
 * memory offset, taking strides and view-offsets into account.
 */
static size_t get_flat_offset(numlu_ndarray* arr, size_t logical_idx) {
  /* CASE 0-D: Scalars have only one element. The logical_idx must be 0.
     We return the base offset without accessing strides/shape. */
  if (arr->ndims == 0) {
    return arr->offset;
  }

  /* CASE 1-D: Simple linear calculation (Optimization) */
  if (arr->ndims == 1) {
    return arr->offset + (logical_idx * arr->strides[0]);
  }

  /* CASE N-D: Decompose logical_idx into coordinates based on shape */
  size_t offset = arr->offset;
  size_t remaining = logical_idx;
  for (int i = 0; i < arr->ndims; i++) {
    /* 
     * We need the logical strides (product of shapes) to decompose.
     * Note: These are NOT the memory strides (arr->strides).
     */
    size_t logical_stride = 1;
    for (int j = i + 1; j < arr->ndims; j++) {
      logical_stride *= arr->shape[j];
    }
    size_t coord = remaining / logical_stride;
    remaining %= logical_stride;        
    offset += coord * arr->strides[i];
  }
  return offset;
}

/* Garbage collector: frees ONLY MKL-allocated data block if owner */
static int l_ndarray_gc(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  
  /* 1. Free main data block only if this object is the owner (is_view == 0) */
  if (!arr->is_view && arr->data) {
    mkl_free(arr->data);
    arr->data = NULL;
  }
  
  /* 2. IMPORTANT: Do NOT free arr->shape or arr->strides. 
   * They are now managed by Lua's GC as User Values in slots 2 and 3. */
  
  return 0;
}

/* Unified __index: handles properties (arr.size) and numeric access (arr[i]) */
static int l_ndarray_index(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");

  /* CASE A: Numeric Index -> arr[i] */
  if (lua_isnumber(L, 2)) {
    /* NumPy-conformity: 0-D arrays (scalars) cannot be indexed */
    if (arr->ndims == 0) {
      return luaL_error(L, "IndexError: too many indices for array: array is 0nd-dimensional");
    }

    lua_Integer idx = lua_tointeger(L, 2);
    
    /* Bounds check (1-based) */
    if (idx < 1 || idx > (lua_Integer)arr->size) {
      return luaL_error(L, "numlu: index %d out of bounds (size %d)", (int)idx, (int)arr->size);
    }

    /* get_flat_offset is now 0-D aware and returns arr->offset if ndims == 0 */
    size_t i = get_flat_offset(arr, (size_t)idx - 1);

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
    /* Create a table of size ndims. For 0-D, this creates an empty table {}. */
    lua_createtable(L, arr->ndims, 0);
    if (arr->ndims > 0 && arr->shape != NULL) {
      for (int i = 0; i < arr->ndims; i++) {
        lua_pushinteger(L, (lua_Integer)arr->shape[i]);
        lua_rawseti(L, -2, i + 1);
      }
    }
    return 1;
  }
  if (strcmp(key, "is_view") == 0) {
    lua_pushboolean(L, arr->is_view);
    return 1;
  }
  if (strcmp(key, "is_contiguous") == 0) {
    lua_pushboolean(L, numlu_is_contiguous(arr));
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
    size_t i = get_flat_offset(arr, (size_t)idx - 1);

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
  size_t i = get_flat_offset(arr, (size_t)idx - 1);
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

/* Length operator (#arr): Returns the size of the first dimension */
static int l_ndarray_len(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  
  /* 
   * NumPy-conformity: 0-D arrays (scalars) do not have a length.
   * Calling len() on a scalar in Python raises a TypeError.
   */
  if (arr->ndims == 0) {
    return luaL_error(L, "TypeError: len() of unsized object (0-D array)");
  }

  /* 
   * Return the size of the first dimension. 
   * For N-D arrays where N > 0, shape[0] is the correct length.
   */
  if (arr->shape != NULL) {
    lua_pushinteger(L, (lua_Integer)arr->shape[0]);
  }
  else {
    /* Safety fallback: if ndims > 0 but shape is unexpectedly NULL */
    lua_pushinteger(L, 0);
  }
  
  return 1;
}

/* 
 * Helper function to parse a slicing string like "start:stop:step"
 * Returns SLICE_RANGE if it contains a colon, SLICE_SCALAR otherwise.
 */
static slice_type parse_slice_string(const char *s, size_t max_len, long long *start,
				     long long *stop, long long *step) {
  /* Default values: full range with step 1 */
  *start = 1;
  *stop = (long long)max_len;
  *step = 1;

  /* Check for range indicator (colon) before parsing */
  int is_range = (strchr(s, ':') != NULL);

  /* Special case: full dimension. Keep 1-based (start=1, stop=max_len). */
  if (strcmp(s, ":") == 0) {
    return SLICE_RANGE;
  }

  /* sscanf_s returns the number of successfully filled variables */
  int count = sscanf_s(s, "%lld:%lld:%lld", start, stop, step);
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
    *start = (long long)max_len + *start + 1;
  }
    
  /* Handle negative indices for 'stop' */
  if (count >= 2 && *stop < 0) {
    *stop = (long long)max_len + *stop + 1;
  }

  /* Bounds checking to prevent memory access errors */
  if (*start < 1) *start = 1;
  if (*start > (long long)max_len) *start = (long long)max_len;
  if (*start < 1 || *start > (long long)max_len || *stop > (long long)max_len) {
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
  const numlu_dtype_info* dtype = numlu_dtype_check(L, 2);
  
  numlu_ndarray* arr = (numlu_ndarray*)lua_newuserdatauv(L, sizeof(numlu_ndarray), 3);
  memset(arr, 0, sizeof(numlu_ndarray));
  luaL_setmetatable(L, "numlu.ndarray");

  /* Slot 1: Anchor DType singleton */
  numlu_push_dtype(L, dtype);
  lua_setiuservalue(L, -2, 1);

  size_t total_size = 1;

  /* 1. Determine ndims (ndims == 0 is now allowed for scalars) */
  if (lua_isnumber(L, 1)) {
    arr->ndims = 1;
  } 
  else if (lua_istable(L, 1)) {
    arr->ndims = (int)lua_rawlen(L, 1);
    /* No error for empty table; ndims will be 0 */
  } 
  else {
    return luaL_typeerror(L, 1, "number or table");
  }

  /* 2. Allocate metadata buffers IMMEDIATELY.
     alloc_metadata_buffer returns NULL for ndims == 0 */
  arr->shape = (size_t*)alloc_metadata_buffer(L, arr->ndims, 2);   /* Slot 2 */
  arr->strides = (size_t*)alloc_metadata_buffer(L, arr->ndims, 3); /* Slot 3 */

  /* 3. Populate buffers and calculate total size */
  if (lua_isnumber(L, 1)) {
    lua_Integer s = luaL_checkinteger(L, 1);
    if (s <= 0) return luaL_error(L, "numlu: size must be positive");
    
    arr->shape[0] = (size_t)s;
    arr->strides[0] = 1;
    total_size = arr->shape[0];
  } 
  else if (arr->ndims > 0) {
    /* Standard N-D initialization */
    for (int i = 1; i <= arr->ndims; i++) {
      lua_rawgeti(L, 1, i);
      arr->shape[i-1] = (size_t)luaL_checkinteger(L, -1);
      total_size *= arr->shape[i-1];
      lua_pop(L, 1);
    }
    calc_strides_row_major(arr->ndims, arr->shape, arr->strides);
  } 
  else {
    /* 0-D Scalar Case: total_size remains 1, shape/strides stay NULL */
    total_size = 1;
  }

  /* 4. Allocate and zero-initialize MKL memory */
  arr->size = total_size;
  arr->dtype = dtype;
  arr->data = mkl_malloc(total_size * dtype->itemsize, 64);
  
  if (!arr->data) {
    return luaL_error(L, "numlu: mkl_malloc failed for data");
  }
  
  memset(arr->data, 0, total_size * dtype->itemsize);
  arr->is_view = 0;
  arr->offset = 0;
  
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

  /* NumPy-conformity: 0-D arrays cannot be indexed/called with indices */
  if (arr->ndims == 0) {
    return luaL_error(L, "IndexError: too many indices for array: array is 0nd-dimensional");
  }

  /* Ensure stack space for N-dimensional metadata + overhead */
  luaL_checkstack(L, total_args + 10, "numlu: too many dimensions for stack");

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
          long long st, sp, sk;
          if (parse_slice_string(lua_tostring(L, i + 2), arr->shape[i], &st, &sp, &sk) == SLICE_RANGE) {
            view_ndims++;
          }
        }
        /* Scalar integers collapse the dimension, so view_ndims does not increase */
      }
      else {
        /* Missing indices in partial indexing are treated as ":" (Full Range) */
        view_ndims++;
      }
    }

    /* Step 2: Initialize the new View object with 3 User Value slots */
    numlu_ndarray *view = (numlu_ndarray *)lua_newuserdatauv(L, sizeof(numlu_ndarray), 3);
    memset(view, 0, sizeof(numlu_ndarray)); /* Safety: ensure all pointers are NULL */
    luaL_getmetatable(L, "numlu.ndarray");
    lua_setmetatable(L, -2);

    /* Slot 1: Anchor original array/owner to prevent GC while view exists */
    if (arr->is_view) {
      lua_getiuservalue(L, 1, 1); /* Get the root owner from current view */
    }
    else {
      lua_pushvalue(L, 1); /* Current is the owner */
    }
    lua_setiuservalue(L, -2, 1);

    /* Slot 2 & 3: Allocate GC-managed metadata for the view */
    /* Note: We handle the 0-dim case (scalar view) by allocating at least 1 element */
    /* or handling it specifically */
    int alloc_dims = (view_ndims > 0) ? view_ndims : 1;
    view->shape = (size_t*)alloc_metadata_buffer(L, alloc_dims, 2);
    view->strides = (size_t*)alloc_metadata_buffer(L, alloc_dims, 3);

    view->ndims = view_ndims;
    view->data = arr->data; 
    view->dtype = arr->dtype;
    view->is_view = 1;
    view->offset = arr->offset;
    view->size = 1;
    
    /* Step 3: Populate view metadata and calculate offset */
    int target_dim = 0;
    for (int i = 0; i < (int)arr->ndims; i++) {
      long long start, stop, step;
      slice_type stype = SLICE_INVALID;

      if (i < total_args) {
        if (lua_type(L, i + 2) == LUA_TSTRING) {
          stype = parse_slice_string(lua_tostring(L, i + 2), arr->shape[i], &start, &stop, &step);
          if (stype == SLICE_INVALID) {
            /* No manual free needed: Lua GC handles the already allocated view and its buffers */
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

/* Method: arr:squeeze([axis]) - Removes dimensions of size 1 */
static int l_ndarray_at_squeeze(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  int axis = (int)luaL_optinteger(L, 2, 0); /* 0 means: squeeze all */

  /* 1. Validate axis if provided (1-based for Lua) */
  if (axis != 0 && (axis < 1 || axis > arr->ndims)) {
    return luaL_error(L, "numlu: axis %d out of bounds (ndims: %d)", axis, arr->ndims);
  }

  /* 2. Count resulting dimensions */
  int new_ndims = 0;
  if (axis == 0) {
    for (int i = 0; i < arr->ndims; i++) {
      if (arr->shape[i] > 1) new_ndims++;
    }
  }
  else {
    if (arr->shape[axis - 1] != 1) {
      return luaL_error(L, "numlu: cannot squeeze axis %d, size is %d (expected 1)", 
                        axis, (int)arr->shape[axis - 1]);
    }
    new_ndims = arr->ndims - 1;
  }

  if (new_ndims == 0) new_ndims = 1;

  /* 3. Create the new view object with 3 User Value slots */
  numlu_ndarray *view = (numlu_ndarray *)lua_newuserdatauv(L, sizeof(numlu_ndarray), 3);
  memset(view, 0, sizeof(numlu_ndarray));
  luaL_getmetatable(L, "numlu.ndarray");
  lua_setmetatable(L, -2);
    
  /* Slot 1: Anchor original owner */
  if (arr->is_view) lua_getiuservalue(L, 1, 1);
  else lua_pushvalue(L, 1);
  lua_setiuservalue(L, -2, 1);

  /* Slot 2 & 3: Allocate GC-managed metadata buffers */
  view->ndims = new_ndims;
  view->shape = (size_t*)alloc_metadata_buffer(L, view->ndims, 2);
  view->strides = (size_t*)alloc_metadata_buffer(L, view->ndims, 3);

  view->data = arr->data;
  view->dtype = arr->dtype;
  view->is_view = 1;
  view->offset = arr->offset;
  view->size = 1; 
  
  /* 4. Populate new shape and strides */
  int target = 0;
  for (int i = 0; i < arr->ndims; i++) {
    bool skip = false;
    if (axis == 0) {
      if (arr->shape[i] == 1 && target < new_ndims) skip = true;
    }
    else {
      if (i == axis - 1) skip = true;
    }
    if (!skip && target < new_ndims) {
      view->shape[target] = arr->shape[i];
      view->strides[target] = arr->strides[i];
      view->size *= view->shape[target];
      target++;
    }
  }
    
  if (target == 0) {
    view->shape[0] = 1;
    view->strides[0] = 1;
  }
  return 1;
}

/* Method: __tostring - Provides a readable description of the ndarray */
static int l_ndarray_tostring(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  
  luaL_Buffer b;
  luaL_buffinit(L, &b);
  
  /* Start building the string: numlu.ndarray<type>({shape}) */
  luaL_addstring(&b, "numlu.ndarray<");
  luaL_addstring(&b, arr->dtype->name);
  luaL_addstring(&b, ">({");
  
  /* 
   * Only iterate if we have dimensions (ndims > 0) and a valid shape pointer.
   * For 0-D arrays (scalars), this part is skipped, resulting in ({})
   */
  if (arr->ndims > 0 && arr->shape != NULL) {
    for (int i = 0; i < arr->ndims; i++) {
      char dim[32];
      sprintf_s(dim, sizeof(dim), "%zu", arr->shape[i]);
      luaL_addstring(&b, dim);
      if (i < arr->ndims - 1) {
        luaL_addstring(&b, ", ");
      }
    }
  }
  
  luaL_addstring(&b, "})");
  luaL_pushresult(&b);
  return 1;
}

static int l_ndarray_reshape(lua_State* L) {
  numlu_ndarray* arr = luaL_checkudata(L, 1, "numlu.ndarray");
  int new_ndims = 0;
  
  /* 1. Determine new dimensionality */
  if (lua_isnumber(L, 2)) new_ndims = 1;
  else if (lua_istable(L, 2)) new_ndims = (int)lua_rawlen(L, 2);
  else return luaL_typeerror(L, 2, "number or table");

  if (new_ndims == 0) return luaL_error(L, "numlu: shape cannot be empty");

  /* 2. Check if we can stay as a View or must Copy */
  bool needs_copy = !numlu_is_contiguous(arr);
  
  /* Create the result object (3 slots: Owner/DType, Shape, Strides) */
  numlu_ndarray *res = (numlu_ndarray *)lua_newuserdatauv(L, sizeof(numlu_ndarray), 3);
  memset(res, 0, sizeof(numlu_ndarray));
  luaL_setmetatable(L, "numlu.ndarray");

  /* Setup Metadata Buffers */
  res->ndims = new_ndims;
  res->shape = (size_t*)alloc_metadata_buffer(L, res->ndims, 2);
  res->strides = (size_t*)alloc_metadata_buffer(L, res->ndims, 3);
  res->dtype = arr->dtype;

  /* 3. Parse Shape (Directly into res->shape) */
  int auto_dim_idx = -1;
  size_t product_other_dims = 1;
  if (lua_isnumber(L, 2)) {
    long dim = (long)luaL_checkinteger(L, 2);
    res->shape[0] = (dim == -1) ? arr->size : (size_t)dim;
    product_other_dims = res->shape[0];
  }
  else {
    for (int i = 1; i <= new_ndims; i++) {
      lua_rawgeti(L, 2, i);
      long dim = (long)luaL_checkinteger(L, -1);
      if (dim == -1) {
        if (auto_dim_idx != -1) return luaL_error(L, "numlu: only one -1 allowed");
        auto_dim_idx = i - 1;
      }
      else {
        if (dim <= 0) return luaL_error(L, "numlu: dimensions must be positive");
        res->shape[i-1] = (size_t)dim;
        product_other_dims *= res->shape[i-1];
      }
      lua_pop(L, 1);
    }
    if (auto_dim_idx != -1) {
      if (arr->size % product_other_dims != 0)
        return luaL_error(L, "numlu: size %zu not divisible by product %zu",
			  arr->size, product_other_dims);
      res->shape[auto_dim_idx] = arr->size / product_other_dims;
      product_other_dims *= res->shape[auto_dim_idx];
    }
  }

  if (product_other_dims != arr->size) 
    return luaL_error(L, "numlu: reshape size mismatch");

  calc_strides_row_major(res->ndims, res->shape, res->strides);
  res->size = arr->size;

  /* 4. Handle Data: VIEW vs COPY */
  if (!needs_copy) {
    /* CASE A: Contiguous -> Create View */
    res->is_view = 1;
    res->data = arr->data;
    res->offset = arr->offset;
    /* Anchor original owner in Slot 1 */
    if (arr->is_view) lua_getiuservalue(L, 1, 1);
    else lua_pushvalue(L, 1);
    lua_setiuservalue(L, -2, 1);
  } 
  else {
    /* CASE B: Non-Contiguous -> Perform Deep Copy to new MKL block */
    res->is_view = 0;
    res->offset = 0;
    res->data = mkl_malloc(res->size * res->dtype->itemsize, 64);
    if (!res->data) return luaL_error(L, "numlu: mkl_malloc failed for reshape copy");

    /* Anchor DType in Slot 1 (since res is now a fresh owner) */
    numlu_push_dtype(L, res->dtype);
    lua_setiuservalue(L, -2, 1);

    /* Performance Note: For now, we do a simple element-by-element copy. 
       In the future, we could optimize this for specific stride patterns. */
    for (size_t i = 0; i < res->size; i++) {
      size_t src_offset = get_flat_offset(arr, i);
      memcpy((char*)res->data + (i * res->dtype->itemsize),
             (char*)arr->data + (src_offset * res->dtype->itemsize),
             res->dtype->itemsize);
    }
  }

  return 1;
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

  /* Register squeeze method */
  lua_pushcfunction(L, l_ndarray_at_squeeze);
  lua_setfield(L, -2, "squeeze");

  /* Register the tostring operator */
  lua_pushcfunction(L, l_ndarray_tostring);
  lua_setfield(L, -2, "__tostring");

  /* Register reshape method */
  lua_pushcfunction(L, l_ndarray_reshape);
  lua_setfield(L, -2, "reshape");
  
  lua_pop(L, 1);
}
