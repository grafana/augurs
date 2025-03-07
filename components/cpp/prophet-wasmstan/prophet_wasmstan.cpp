// Generated by `wit-bindgen` 0.36.0. DO NOT EDIT!
#include "prophet_wasmstan.h"
#include <stdlib.h>
#include <string.h>

// Exported Functions from `augurs:prophet-wasmstan/optimizer`

__attribute__((__weak__, __export_name__("cabi_post_augurs:prophet-wasmstan/optimizer#optimize")))
void __wasm_export_exports_augurs_prophet_wasmstan_optimizer_optimize_post_return(uint8_t * arg0) {
  switch ((int32_t) (int32_t) *((uint8_t*) (arg0 + 0))) {
    case 0: {
      if ((*((size_t*) (arg0 + 12))) > 0) {
        free(*((uint8_t **) (arg0 + 8)));
      }
      if ((*((size_t*) (arg0 + 20))) > 0) {
        free(*((uint8_t **) (arg0 + 16)));
      }
      if ((*((size_t*) (arg0 + 28))) > 0) {
        free(*((uint8_t **) (arg0 + 24)));
      }
      if ((*((size_t*) (arg0 + 36))) > 0) {
        free(*((uint8_t **) (arg0 + 32)));
      }
      if ((*((size_t*) (arg0 + 44))) > 0) {
        free(*((uint8_t **) (arg0 + 40)));
      }
      size_t len = *((size_t*) (arg0 + 68));
      if (len > 0) {
        uint8_t *ptr = *((uint8_t **) (arg0 + 64));
        for (size_t i = 0; i < len; i++) {
          uint8_t *base = ptr + i * 8;
          (void) base;
        }
        free(ptr);
      }
      size_t len0 = *((size_t*) (arg0 + 76));
      if (len0 > 0) {
        uint8_t *ptr1 = *((uint8_t **) (arg0 + 72));
        for (size_t i2 = 0; i2 < len0; i2++) {
          uint8_t *base = ptr1 + i2 * 8;
          (void) base;
        }
        free(ptr1);
      }
      size_t len3 = *((size_t*) (arg0 + 92));
      if (len3 > 0) {
        uint8_t *ptr4 = *((uint8_t **) (arg0 + 88));
        for (size_t i5 = 0; i5 < len3; i5++) {
          uint8_t *base = ptr4 + i5 * 8;
          (void) base;
        }
        free(ptr4);
      }
      break;
    }
    case 1: {
      if ((*((size_t*) (arg0 + 12))) > 0) {
        free(*((uint8_t **) (arg0 + 8)));
      }
      break;
    }
  }
}

// Canonical ABI intrinsics

__attribute__((__weak__, __export_name__("cabi_realloc")))
void *cabi_realloc(void *ptr, size_t old_size, size_t align, size_t new_size) {
  (void) old_size;
  if (new_size == 0) return (void*) align;
  void *ret = realloc(ptr, new_size);
  if (!ret) abort();
  return ret;
}

// Helper Functions

void prophet_wasmstan_list_f64_free(prophet_wasmstan_list_f64_t *ptr) {
  size_t list_len = ptr->len;
  if (list_len > 0) {
    double *list_ptr = ptr->ptr;
    for (size_t i = 0; i < list_len; i++) {
    }
    free(list_ptr);
  }
}

void augurs_prophet_wasmstan_types_inits_free(augurs_prophet_wasmstan_types_inits_t *ptr) {
  prophet_wasmstan_list_f64_free(&ptr->delta);
  prophet_wasmstan_list_f64_free(&ptr->beta);
}

void prophet_wasmstan_list_s32_free(prophet_wasmstan_list_s32_t *ptr) {
  size_t list_len = ptr->len;
  if (list_len > 0) {
    int32_t *list_ptr = ptr->ptr;
    for (size_t i = 0; i < list_len; i++) {
    }
    free(list_ptr);
  }
}

void augurs_prophet_wasmstan_types_data_free(augurs_prophet_wasmstan_types_data_t *ptr) {
  prophet_wasmstan_list_f64_free(&ptr->y);
  prophet_wasmstan_list_f64_free(&ptr->t);
  prophet_wasmstan_list_f64_free(&ptr->cap);
  prophet_wasmstan_list_f64_free(&ptr->t_change);
  prophet_wasmstan_list_s32_free(&ptr->s_a);
  prophet_wasmstan_list_s32_free(&ptr->s_m);
  prophet_wasmstan_list_f64_free(&ptr->x);
  prophet_wasmstan_list_f64_free(&ptr->sigmas);
}

void augurs_prophet_wasmstan_types_data_json_free(augurs_prophet_wasmstan_types_data_json_t *ptr) {
  prophet_wasmstan_string_free(ptr);
}

void augurs_prophet_wasmstan_types_option_algorithm_free(augurs_prophet_wasmstan_types_option_algorithm_t *ptr) {
  if (ptr->is_some) {
  }
}

void prophet_wasmstan_option_u32_free(prophet_wasmstan_option_u32_t *ptr) {
  if (ptr->is_some) {
  }
}

void prophet_wasmstan_option_f64_free(prophet_wasmstan_option_f64_t *ptr) {
  if (ptr->is_some) {
  }
}

void prophet_wasmstan_option_bool_free(prophet_wasmstan_option_bool_t *ptr) {
  if (ptr->is_some) {
  }
}

void augurs_prophet_wasmstan_types_optimize_opts_free(augurs_prophet_wasmstan_types_optimize_opts_t *ptr) {
  augurs_prophet_wasmstan_types_option_algorithm_free(&ptr->algorithm);
  prophet_wasmstan_option_u32_free(&ptr->seed);
  prophet_wasmstan_option_u32_free(&ptr->chain);
  prophet_wasmstan_option_f64_free(&ptr->init_alpha);
  prophet_wasmstan_option_f64_free(&ptr->tol_obj);
  prophet_wasmstan_option_f64_free(&ptr->tol_rel_obj);
  prophet_wasmstan_option_f64_free(&ptr->tol_grad);
  prophet_wasmstan_option_f64_free(&ptr->tol_rel_grad);
  prophet_wasmstan_option_f64_free(&ptr->tol_param);
  prophet_wasmstan_option_u32_free(&ptr->history_size);
  prophet_wasmstan_option_u32_free(&ptr->iter);
  prophet_wasmstan_option_bool_free(&ptr->jacobian);
  prophet_wasmstan_option_u32_free(&ptr->refresh);
}

void augurs_prophet_wasmstan_types_logs_free(augurs_prophet_wasmstan_types_logs_t *ptr) {
  prophet_wasmstan_string_free(&ptr->debug);
  prophet_wasmstan_string_free(&ptr->info);
  prophet_wasmstan_string_free(&ptr->warn);
  prophet_wasmstan_string_free(&ptr->error);
  prophet_wasmstan_string_free(&ptr->fatal);
}

void augurs_prophet_wasmstan_types_optimized_params_free(augurs_prophet_wasmstan_types_optimized_params_t *ptr) {
  prophet_wasmstan_list_f64_free(&ptr->delta);
  prophet_wasmstan_list_f64_free(&ptr->beta);
  prophet_wasmstan_list_f64_free(&ptr->trend);
}

void augurs_prophet_wasmstan_types_optimize_output_free(augurs_prophet_wasmstan_types_optimize_output_t *ptr) {
  augurs_prophet_wasmstan_types_logs_free(&ptr->logs);
  augurs_prophet_wasmstan_types_optimized_params_free(&ptr->params);
}

void exports_augurs_prophet_wasmstan_optimizer_inits_free(exports_augurs_prophet_wasmstan_optimizer_inits_t *ptr) {
  augurs_prophet_wasmstan_types_inits_free(ptr);
}

void exports_augurs_prophet_wasmstan_optimizer_data_json_free(exports_augurs_prophet_wasmstan_optimizer_data_json_t *ptr) {
  augurs_prophet_wasmstan_types_data_json_free(ptr);
}

void exports_augurs_prophet_wasmstan_optimizer_optimize_opts_free(exports_augurs_prophet_wasmstan_optimizer_optimize_opts_t *ptr) {
  augurs_prophet_wasmstan_types_optimize_opts_free(ptr);
}

void exports_augurs_prophet_wasmstan_optimizer_optimize_output_free(exports_augurs_prophet_wasmstan_optimizer_optimize_output_t *ptr) {
  augurs_prophet_wasmstan_types_optimize_output_free(ptr);
}

void exports_augurs_prophet_wasmstan_optimizer_result_optimize_output_string_free(exports_augurs_prophet_wasmstan_optimizer_result_optimize_output_string_t *ptr) {
  if (!ptr->is_err) {
    exports_augurs_prophet_wasmstan_optimizer_optimize_output_free(&ptr->val.ok);
  } else {
    prophet_wasmstan_string_free(&ptr->val.err);
  }
}

void prophet_wasmstan_string_set(prophet_wasmstan_string_t *ret, const char*s) {
  ret->ptr = (uint8_t*) s;
  ret->len = strlen(s);
}

void prophet_wasmstan_string_dup(prophet_wasmstan_string_t *ret, const char*s) {
  ret->len = strlen(s);
  ret->ptr = (uint8_t*) cabi_realloc(NULL, 0, 1, ret->len * 1);
  memcpy(ret->ptr, s, ret->len * 1);
}

void prophet_wasmstan_string_free(prophet_wasmstan_string_t *ret) {
  if (ret->len > 0) {
    free(ret->ptr);
  }
  ret->ptr = NULL;
  ret->len = 0;
}

// Component Adapters

__attribute__((__aligned__(8)))
static uint8_t RET_AREA[96];

__attribute__((__export_name__("augurs:prophet-wasmstan/optimizer#optimize")))
uint8_t * __wasm_export_exports_augurs_prophet_wasmstan_optimizer_optimize(uint8_t * arg) {
  augurs_prophet_wasmstan_types_option_algorithm_t option;
  switch ((int32_t) *((uint8_t*) (arg + 48))) {
    case 0: {
      option.is_some = false;
      break;
    }
    case 1: {
      option.is_some = true;
      option.val = (int32_t) *((uint8_t*) (arg + 49));
      break;
    }
  }
  prophet_wasmstan_option_u32_t option0;
  switch ((int32_t) *((uint8_t*) (arg + 52))) {
    case 0: {
      option0.is_some = false;
      break;
    }
    case 1: {
      option0.is_some = true;
      option0.val = (uint32_t) (*((int32_t*) (arg + 56)));
      break;
    }
  }
  prophet_wasmstan_option_u32_t option1;
  switch ((int32_t) *((uint8_t*) (arg + 60))) {
    case 0: {
      option1.is_some = false;
      break;
    }
    case 1: {
      option1.is_some = true;
      option1.val = (uint32_t) (*((int32_t*) (arg + 64)));
      break;
    }
  }
  prophet_wasmstan_option_f64_t option2;
  switch ((int32_t) *((uint8_t*) (arg + 72))) {
    case 0: {
      option2.is_some = false;
      break;
    }
    case 1: {
      option2.is_some = true;
      option2.val = *((double*) (arg + 80));
      break;
    }
  }
  prophet_wasmstan_option_f64_t option3;
  switch ((int32_t) *((uint8_t*) (arg + 88))) {
    case 0: {
      option3.is_some = false;
      break;
    }
    case 1: {
      option3.is_some = true;
      option3.val = *((double*) (arg + 96));
      break;
    }
  }
  prophet_wasmstan_option_f64_t option4;
  switch ((int32_t) *((uint8_t*) (arg + 104))) {
    case 0: {
      option4.is_some = false;
      break;
    }
    case 1: {
      option4.is_some = true;
      option4.val = *((double*) (arg + 112));
      break;
    }
  }
  prophet_wasmstan_option_f64_t option5;
  switch ((int32_t) *((uint8_t*) (arg + 120))) {
    case 0: {
      option5.is_some = false;
      break;
    }
    case 1: {
      option5.is_some = true;
      option5.val = *((double*) (arg + 128));
      break;
    }
  }
  prophet_wasmstan_option_f64_t option6;
  switch ((int32_t) *((uint8_t*) (arg + 136))) {
    case 0: {
      option6.is_some = false;
      break;
    }
    case 1: {
      option6.is_some = true;
      option6.val = *((double*) (arg + 144));
      break;
    }
  }
  prophet_wasmstan_option_f64_t option7;
  switch ((int32_t) *((uint8_t*) (arg + 152))) {
    case 0: {
      option7.is_some = false;
      break;
    }
    case 1: {
      option7.is_some = true;
      option7.val = *((double*) (arg + 160));
      break;
    }
  }
  prophet_wasmstan_option_u32_t option8;
  switch ((int32_t) *((uint8_t*) (arg + 168))) {
    case 0: {
      option8.is_some = false;
      break;
    }
    case 1: {
      option8.is_some = true;
      option8.val = (uint32_t) (*((int32_t*) (arg + 172)));
      break;
    }
  }
  prophet_wasmstan_option_u32_t option9;
  switch ((int32_t) *((uint8_t*) (arg + 176))) {
    case 0: {
      option9.is_some = false;
      break;
    }
    case 1: {
      option9.is_some = true;
      option9.val = (uint32_t) (*((int32_t*) (arg + 180)));
      break;
    }
  }
  prophet_wasmstan_option_bool_t option10;
  switch ((int32_t) *((uint8_t*) (arg + 184))) {
    case 0: {
      option10.is_some = false;
      break;
    }
    case 1: {
      option10.is_some = true;
      option10.val = (int32_t) *((uint8_t*) (arg + 185));
      break;
    }
  }
  prophet_wasmstan_option_u32_t option11;
  switch ((int32_t) *((uint8_t*) (arg + 188))) {
    case 0: {
      option11.is_some = false;
      break;
    }
    case 1: {
      option11.is_some = true;
      option11.val = (uint32_t) (*((int32_t*) (arg + 192)));
      break;
    }
  }
  exports_augurs_prophet_wasmstan_optimizer_inits_t arg12 = (augurs_prophet_wasmstan_types_inits_t) {
    (double) *((double*) (arg + 0)),
    (double) *((double*) (arg + 8)),
    (prophet_wasmstan_list_f64_t) (prophet_wasmstan_list_f64_t) { (double*)(*((uint8_t **) (arg + 16))), (*((size_t*) (arg + 20))) },
    (prophet_wasmstan_list_f64_t) (prophet_wasmstan_list_f64_t) { (double*)(*((uint8_t **) (arg + 24))), (*((size_t*) (arg + 28))) },
    (double) *((double*) (arg + 32)),
  };
  exports_augurs_prophet_wasmstan_optimizer_data_json_t arg13 = (prophet_wasmstan_string_t) { (uint8_t*)(*((uint8_t **) (arg + 40))), (*((size_t*) (arg + 44))) };
  exports_augurs_prophet_wasmstan_optimizer_optimize_opts_t arg14 = (augurs_prophet_wasmstan_types_optimize_opts_t) {
    (augurs_prophet_wasmstan_types_option_algorithm_t) option,
    (prophet_wasmstan_option_u32_t) option0,
    (prophet_wasmstan_option_u32_t) option1,
    (prophet_wasmstan_option_f64_t) option2,
    (prophet_wasmstan_option_f64_t) option3,
    (prophet_wasmstan_option_f64_t) option4,
    (prophet_wasmstan_option_f64_t) option5,
    (prophet_wasmstan_option_f64_t) option6,
    (prophet_wasmstan_option_f64_t) option7,
    (prophet_wasmstan_option_u32_t) option8,
    (prophet_wasmstan_option_u32_t) option9,
    (prophet_wasmstan_option_bool_t) option10,
    (prophet_wasmstan_option_u32_t) option11,
  };
  exports_augurs_prophet_wasmstan_optimizer_result_optimize_output_string_t ret;
  exports_augurs_prophet_wasmstan_optimizer_optimize_output_t ok;
  prophet_wasmstan_string_t err;
  ret.is_err = !exports_augurs_prophet_wasmstan_optimizer_optimize(&arg12, &arg13, &arg14, &ok, &err);
  if (ret.is_err) {
    ret.val.err = err;
  }
  if (!ret.is_err) {
    ret.val.ok = ok;
  }
  free(arg);
  uint8_t *ptr = (uint8_t *) &RET_AREA;
  if ((ret).is_err) {
    const prophet_wasmstan_string_t *payload15 = &(ret).val.err;*((int8_t*)(ptr + 0)) = 1;
    *((size_t*)(ptr + 12)) = (*payload15).len;
    *((uint8_t **)(ptr + 8)) = (uint8_t *) (*payload15).ptr;
  } else {
    const exports_augurs_prophet_wasmstan_optimizer_optimize_output_t *payload = &(ret).val.ok;*((int8_t*)(ptr + 0)) = 0;
    *((size_t*)(ptr + 12)) = (((*payload).logs).debug).len;
    *((uint8_t **)(ptr + 8)) = (uint8_t *) (((*payload).logs).debug).ptr;
    *((size_t*)(ptr + 20)) = (((*payload).logs).info).len;
    *((uint8_t **)(ptr + 16)) = (uint8_t *) (((*payload).logs).info).ptr;
    *((size_t*)(ptr + 28)) = (((*payload).logs).warn).len;
    *((uint8_t **)(ptr + 24)) = (uint8_t *) (((*payload).logs).warn).ptr;
    *((size_t*)(ptr + 36)) = (((*payload).logs).error).len;
    *((uint8_t **)(ptr + 32)) = (uint8_t *) (((*payload).logs).error).ptr;
    *((size_t*)(ptr + 44)) = (((*payload).logs).fatal).len;
    *((uint8_t **)(ptr + 40)) = (uint8_t *) (((*payload).logs).fatal).ptr;
    *((double*)(ptr + 48)) = ((*payload).params).k;
    *((double*)(ptr + 56)) = ((*payload).params).m;
    *((size_t*)(ptr + 68)) = (((*payload).params).delta).len;
    *((uint8_t **)(ptr + 64)) = (uint8_t *) (((*payload).params).delta).ptr;
    *((size_t*)(ptr + 76)) = (((*payload).params).beta).len;
    *((uint8_t **)(ptr + 72)) = (uint8_t *) (((*payload).params).beta).ptr;
    *((double*)(ptr + 80)) = ((*payload).params).sigma_obs;
    *((size_t*)(ptr + 92)) = (((*payload).params).trend).len;
    *((uint8_t **)(ptr + 88)) = (uint8_t *) (((*payload).params).trend).ptr;
  }
  return ptr;
}

// Ensure that the *_component_type.o object is linked in

extern void __component_type_object_force_link_prophet_wasmstan(void);
void __component_type_object_force_link_prophet_wasmstan_public_use_in_this_compilation_unit(void) {
  __component_type_object_force_link_prophet_wasmstan();
}
