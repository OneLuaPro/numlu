#ifndef NUMLU_DTYPE_H
#define NUMLU_DTYPE_H

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include <stddef.h>

typedef enum {
  NUMLU_TYPE_F32,
  NUMLU_TYPE_F64,
  NUMLU_TYPE_C64,
  NUMLU_TYPE_C128
} numlu_type_id;

typedef struct {
  numlu_type_id id;
  size_t itemsize;
  const char* name;
} numlu_dtype_info;

extern const numlu_dtype_info NUMLU_DTYPE_F32;
extern const numlu_dtype_info NUMLU_DTYPE_F64;
extern const numlu_dtype_info NUMLU_DTYPE_C64;
extern const numlu_dtype_info NUMLU_DTYPE_C128;

const numlu_dtype_info* numlu_dtype_from_string(const char* name);
const numlu_dtype_info* numlu_dtype_check(lua_State* L, int arg);
void numlu_push_dtype(lua_State* L, const numlu_dtype_info* info);
void numlu_dtype_register(lua_State* L);

#endif
