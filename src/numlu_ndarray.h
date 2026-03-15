#ifndef NUMLU_NDARRAY_H
#define NUMLU_NDARRAY_H

#include "numlu_dtype.h"

typedef struct {
  void* data;              /* Pointer to MKL-aligned memory block */
  const numlu_dtype_info* dtype;
    
  size_t size;             /* Total number of elements (flat count) */
  int ndims;               /* Number of dimensions (e.g., 2 for matrix) */
    
  /* Metadata for multi-dimensional access */
  size_t* shape;           /* Array of sizes per dimension: {rows, cols, ...} */
  size_t* strides;         /* Step size in memory per dimension (in elements) */
    
  size_t offset;           /* Offset for views/slicing (base pointer + offset) */
  int is_view;             /* Boolean: 1 if this is a slice/view, 0 if owner */
} numlu_ndarray;

void numlu_ndarray_register(lua_State* L);
int l_ndarray_new(lua_State* L);

typedef enum {
  SLICE_INVALID = 0,
  SLICE_SCALAR  = 1, /* Single index: dimension collapses (e.g., "5" or "-1") */
  SLICE_RANGE   = 2  /* Range: dimension persists (e.g., "1:5" or ":") */
} slice_type;

#endif
