#ifndef NUMLU_NDARRAY_H
#define NUMLU_NDARRAY_H

#include "numlu_dtype.h"

typedef struct {
  void* data;
  const numlu_dtype_info* dtype;
  size_t size;
} numlu_ndarray;

void numlu_ndarray_register(lua_State* L);
int l_ndarray_new(lua_State* L);

#endif
