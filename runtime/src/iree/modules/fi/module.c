// Minimal FI native module that exposes fi_plugin_f32.
// It marshals VM buffer arguments into host pointers and calls the runtime
// function fi_plugin_f32 (provided in runtime/fi_plugin.cpp).

#include "iree/modules/fi/module.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/buffer_view.h"
#include "iree/modules/hal/module.h"

extern void fi_plugin_f32(float* C, int64_t M, int64_t N, int64_t enable_flag);

static float fi_parse_env_float(const char* name, float default_value) {
  const char* env = getenv(name);
  if (!env || env[0] == '\0') return default_value;
  char* endptr = NULL;
  float value = strtof(env, &endptr);
  if (endptr == env || (endptr && *endptr != '\0')) return default_value;
  return value;
}

static bool fi_is_enabled(int64_t enable_flag) {
  return enable_flag != 0;
}

#define IREE_FI_MODULE_VERSION_0_0 0x00000000u
#define IREE_FI_MODULE_VERSION_LATEST IREE_FI_MODULE_VERSION_0_0

IREE_VM_ABI_FIXED_STRUCT(rIII, {
  iree_vm_ref_t r0;
  int64_t i1;
  int64_t i2;
  int64_t i3;
});

IREE_VM_ABI_DEFINE_SHIM(rIII, r);

typedef struct iree_fi_module_t {
  iree_allocator_t host_allocator;
} iree_fi_module_t;

#define IREE_FI_MODULE_CAST(module) \
  (iree_fi_module_t*)((uint8_t*)(module) + iree_vm_native_module_size())

typedef struct iree_fi_module_state_t {
  iree_allocator_t host_allocator;
} iree_fi_module_state_t;

static void IREE_API_PTR iree_fi_module_destroy(void* base_module) {
  iree_fi_module_t* module = IREE_FI_MODULE_CAST(base_module);
  (void)module;
}

static iree_status_t IREE_API_PTR iree_fi_module_alloc_state(
    void* self, iree_allocator_t host_allocator,
    iree_vm_module_state_t** out_module_state) {
  iree_fi_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*state),
                                            (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}

static void IREE_API_PTR iree_fi_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  iree_fi_module_state_t* state = (iree_fi_module_state_t*)module_state;
  iree_allocator_free(state->host_allocator, state);
}

static iree_status_t IREE_API_PTR iree_fi_module_fork_state(
    void* self, iree_vm_module_state_t* parent_state,
    iree_allocator_t host_allocator, iree_vm_module_state_t** out_child_state) {
  return iree_fi_module_alloc_state(self, host_allocator, out_child_state);
}

static iree_status_t IREE_API_PTR iree_fi_module_notify(
    void* self, iree_vm_module_state_t* module_state, iree_vm_signal_t signal) {
  (void)self;
  (void)module_state;
  (void)signal;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_fi_module_fi_plugin_f32, iree_fi_module_state_t,
                   rIII, r) {
  iree_hal_buffer_view_t* c_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r0, &c_view));

  int64_t M = args->i1;
  int64_t N = args->i2;

  // Skip mapping if FI is disabled.
  if (!fi_is_enabled(args->i3)) {
    iree_hal_buffer_view_retain(c_view);
    rets->r0 = iree_hal_buffer_view_move_ref(c_view);
    return iree_ok_status();
  }

  iree_hal_buffer_t* c_buf = iree_hal_buffer_view_buffer(c_view);
  iree_hal_buffer_mapping_t c_mapping = {{0}};
  iree_device_size_t c_byte_offset = 0;
  iree_device_size_t c_byte_length = 0;

  iree_host_size_t c_rank = iree_hal_buffer_view_shape_rank(c_view);
  iree_hal_dim_t c_start[8] = {0};
  iree_hal_dim_t c_lengths[8] = {0};
  if (c_rank > IREE_ARRAYSIZE(c_start)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_view rank too large for fi");
  }
  for (iree_host_size_t i = 0; i < c_rank; ++i) {
    c_lengths[i] = iree_hal_buffer_view_shape_dim(c_view, i);
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_compute_range(
      c_view, c_rank, c_start, c_rank, c_lengths, &c_byte_offset,
      &c_byte_length));

    // NOTE: compute_range() already provides the correct mapping range for the
    // buffer view. Adding buffer_byte_offset(c_buf) again can double-apply an
    // offset for subview-backed buffers and cause out-of-range mapping.

  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      c_buf, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      c_byte_offset, c_byte_length, &c_mapping));

  float* C = (float*)c_mapping.contents.data;
  fi_plugin_f32(C, M, N, args->i3);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&c_mapping));

  iree_hal_buffer_view_retain(c_view);
  rets->r0 = iree_hal_buffer_view_move_ref(c_view);

  return iree_ok_status();
}

static const iree_vm_native_function_ptr_t iree_fi_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)               \
  {                                                                     \
      .shim = (iree_vm_native_function_shim_t)                          \
          iree_vm_shim_##arg_types##_##ret_types,                       \
      .target = (iree_vm_native_function_target_t)(target_fn),          \
  },
#include "iree/modules/fi/exports.inl"
#undef EXPORT_FN
};

static const iree_vm_native_import_descriptor_t iree_fi_module_imports_[1];

static const iree_vm_native_export_descriptor_t iree_fi_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)               \
  {                                                                     \
      .local_name = iree_string_view_literal(name),                     \
      .calling_convention =                                            \
          iree_string_view_literal("0" #arg_types "_" #ret_types),      \
      .attr_count = 0,                                                  \
      .attrs = NULL,                                                    \
  },
#include "iree/modules/fi/exports.inl"
#undef EXPORT_FN
};

static const iree_vm_native_module_descriptor_t iree_fi_module_descriptor_ = {
    .name = iree_string_view_literal("fi"),
    .version = IREE_FI_MODULE_VERSION_LATEST,
    .attr_count = 0,
    .attrs = NULL,
    .dependency_count = 0,
    .dependencies = NULL,
    .import_count = 0,
    .imports = iree_fi_module_imports_,
    .export_count = IREE_ARRAYSIZE(iree_fi_module_exports_),
    .exports = iree_fi_module_exports_,
    .function_count = IREE_ARRAYSIZE(iree_fi_module_funcs_),
    .functions = iree_fi_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_fi_module_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** IREE_RESTRICT out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  static const iree_vm_module_t interface = {
      .destroy = iree_fi_module_destroy,
      .alloc_state = iree_fi_module_alloc_state,
      .free_state = iree_fi_module_free_state,
      .fork_state = iree_fi_module_fork_state,
      .notify = iree_fi_module_notify,
  };

  iree_vm_module_t* base_module = NULL;
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_fi_module_t);
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);

  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_fi_module_descriptor_, instance, host_allocator,
      base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_fi_module_t* module = IREE_FI_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;

  *out_module = base_module;
  return iree_ok_status();
}
