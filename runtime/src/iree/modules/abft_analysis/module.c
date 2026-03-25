// Minimal ABFT analysis module for reporting and querying detected failures.

#include "iree/modules/abft_analysis/module.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/native_module.h"
#include "iree/hal/buffer.h"
#include "iree/hal/buffer_view.h"
#include "iree/modules/hal/module.h"

#define IREE_ABFT_ANALYSIS_MODULE_VERSION_0_0 0x00000000u
#define IREE_ABFT_ANALYSIS_MODULE_VERSION_LATEST IREE_ABFT_ANALYSIS_MODULE_VERSION_0_0
#define IREE_ABFT_ANALYSIS_DELTA_ASSERT_THRESHOLD 1.0e3f

// Define shims for (f) -> (), (ff) -> (), (fff) -> (), and () -> (I) signatures.
IREE_VM_ABI_FIXED_STRUCT(ff, {
  float f0;
  float f1;
});
IREE_VM_ABI_FIXED_STRUCT(fff, {
  float f0;
  float f1;
  float f2;
});

IREE_VM_ABI_FIXED_STRUCT(ffff, {
  float f0;
  float f1;
  float f2;
  float f3;
});
IREE_VM_ABI_FIXED_STRUCT(fffff, {
  float f0;
  float f1;
  float f2;
  float f3;
  float f4;
});
IREE_VM_ABI_FIXED_STRUCT(fffffffff, {
  float f0;
  float f1;
  float f2;
  float f3;
  float f4;
  float f5;
  float f6;
  float f7;
  float f8;
});
IREE_VM_ABI_DEFINE_SHIM(f, v);
IREE_VM_ABI_DEFINE_SHIM(ff, v);
IREE_VM_ABI_DEFINE_SHIM(fff, v);
IREE_VM_ABI_DEFINE_SHIM(ffff, v);
IREE_VM_ABI_DEFINE_SHIM(fffff, v);
IREE_VM_ABI_DEFINE_SHIM(fffffffff, v);
IREE_VM_ABI_DEFINE_SHIM(v, I);

typedef struct iree_abft_analysis_module_t {
  iree_allocator_t host_allocator;
} iree_abft_analysis_module_t;

#define IREE_ABFT_ANALYSIS_MODULE_CAST(module) \
  (iree_abft_analysis_module_t*)((uint8_t*)(module) + iree_vm_native_module_size())

typedef struct iree_abft_analysis_module_state_t {
  iree_allocator_t host_allocator;
  int64_t failure_count;
  int report_enabled;
  int delta_assert_enabled;
  float delta_assert_threshold;
  const char* log_path;
} iree_abft_analysis_module_state_t;

static void IREE_API_PTR iree_abft_analysis_module_destroy(void* base_module) {
  iree_abft_analysis_module_t* module = IREE_ABFT_ANALYSIS_MODULE_CAST(base_module);
  (void)module;
}

static iree_status_t IREE_API_PTR iree_abft_analysis_module_alloc_state(
    void* self, iree_allocator_t host_allocator,
    iree_vm_module_state_t** out_module_state) {
  iree_abft_analysis_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*state),
                                            (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  state->failure_count = 0;
  // Make failure reporting opt-in so instrumentation for profiling/runtime
  // comparisons does not force MLONMCU EXIT: 1 in clean runs.
  // Set IREE_ABFT_REPORT=1 to enable strict failure counting/reporting.
  const char* report_env = getenv("IREE_ABFT_REPORT");
  if (report_env && report_env[0] == '1' && report_env[1] == '\0') {
    state->report_enabled = 1;
  } else {
    state->report_enabled = 0;
  }
  // Optional hard assertion on logged row/col deltas.
  // Disabled by default to avoid aborting normal runs.
  state->delta_assert_enabled = 0;
  state->delta_assert_threshold = IREE_ABFT_ANALYSIS_DELTA_ASSERT_THRESHOLD;
  const char* assert_env = getenv("IREE_ABFT_DELTA_ASSERT");
  if (assert_env && assert_env[0] != '\0' && assert_env[0] != '0') {
    state->delta_assert_enabled = 1;
  }
  const char* assert_thr_env = getenv("IREE_ABFT_DELTA_ASSERT_THRESHOLD");
  if (assert_thr_env && assert_thr_env[0] != '\0') {
    float parsed = strtof(assert_thr_env, NULL);
    if (parsed > 0.0f) state->delta_assert_threshold = parsed;
  }
  const char* log_env = getenv("IREE_ABFT_LOG_PATH");
  if (log_env && log_env[0] != '\0') {
    state->log_path = log_env;
  } else {
    // Default logfile when no explicit path is configured.
    state->log_path = "abft_analysis.log";
  }
  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}

static void IREE_API_PTR iree_abft_analysis_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  iree_abft_analysis_module_state_t* state =
      (iree_abft_analysis_module_state_t*)module_state;
  iree_allocator_free(state->host_allocator, state);
}

static iree_status_t IREE_API_PTR iree_abft_analysis_module_fork_state(
    void* self, iree_vm_module_state_t* parent_state,
    iree_allocator_t host_allocator, iree_vm_module_state_t** out_child_state) {
  return iree_abft_analysis_module_alloc_state(self, host_allocator,
                                               out_child_state);
}

static iree_status_t IREE_API_PTR iree_abft_analysis_module_notify(
    void* self, iree_vm_module_state_t* module_state, iree_vm_signal_t signal) {
  (void)self;
  (void)module_state;
  (void)signal;
  return iree_ok_status();
}

static void abft_log_scalar(iree_abft_analysis_module_state_t* state,
                            const char* label, float value) {
  if (!state->log_path) return;
  FILE* f = fopen(state->log_path, "a");
  if (!f) return;
  fprintf(f, "%s=%g\n", label, value);
  fclose(f);
}


static float abft_read_buffer_view_scalar_f32(iree_hal_buffer_view_t* view) {
  if (!view) return 0.0f;
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(view);
  iree_hal_buffer_mapping_t mapping = {{0}};
  iree_device_size_t byte_offset = 0;
  iree_device_size_t byte_length = 0;
  iree_host_size_t rank = iree_hal_buffer_view_shape_rank(view);
  iree_hal_dim_t start[8] = {0};
  iree_hal_dim_t lengths[8] = {0};
  if (rank > IREE_ARRAYSIZE(start)) return 0.0f;
  for (iree_host_size_t i = 0; i < rank; ++i) {
    lengths[i] = iree_hal_buffer_view_shape_dim(view, i);
  }
  if (!iree_status_is_ok(iree_hal_buffer_view_compute_range(
          view, rank, start, rank, lengths, &byte_offset, &byte_length))) {
    return 0.0f;
  }
  byte_offset += iree_hal_buffer_byte_offset(buffer);
  if (!iree_status_is_ok(iree_hal_buffer_map_range(
          buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
          byte_offset, byte_length, &mapping))) {
    return 0.0f;
  }
  float value = 0.0f;
  if (byte_length >= sizeof(float)) {
    value = ((const float*)mapping.contents.data)[0];
  }
  iree_hal_buffer_unmap_range(&mapping);
  return value;
}

// Report a failure if the predicate is false (0). The caller passes an f32
// flag where 1.0 means ok and 0.0 means failure.
IREE_VM_ABI_EXPORT(iree_abft_analysis_module_abft_report_failure,
                   iree_abft_analysis_module_state_t, f, v) {
  if (args->f0 < 0.5f) {
    if (state->report_enabled && state->failure_count == 0) {
      fprintf(stderr, "FIC: values differ more than epsilon\n");
    }
    if (state->report_enabled) {
      state->failure_count++;
    }
  }
  return iree_ok_status();
}

// Logs per-layer full-checksum max absolute deltas.
IREE_VM_ABI_EXPORT(iree_abft_analysis_module_abft_log_rowcol_delta,
                   iree_abft_analysis_module_state_t, fff, v) {
  const float row_delta = args->f1;
  const float col_delta = args->f2;
  if (state->delta_assert_enabled &&
      (fabsf(row_delta) > state->delta_assert_threshold ||
       fabsf(col_delta) > state->delta_assert_threshold)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "ABFT/Freivalds delta assertion failed: layer=%g rowMaxDelta=%g "
        "colMaxDelta=%g (limit=%g)",
        args->f0, row_delta, col_delta,
        state->delta_assert_threshold);
  }
  if (!state->log_path) return iree_ok_status();
  FILE* f = fopen(state->log_path, "a");
  if (!f) return iree_ok_status();
    fprintf(f, "layer=%.15g rowMaxDelta=%.15g colMaxDelta=%.15g\n", args->f0,
            row_delta, col_delta);
  fclose(f);
  return iree_ok_status();
}

// Logs per-layer full-checksum absolute and relative metrics.
IREE_VM_ABI_EXPORT(iree_abft_analysis_module_abft_log_rowcol_metrics,
                   iree_abft_analysis_module_state_t, fffff, v) {
  const float row_delta = args->f1;
  const float col_delta = args->f2;
  if (state->delta_assert_enabled &&
      (fabsf(row_delta) > state->delta_assert_threshold ||
       fabsf(col_delta) > state->delta_assert_threshold)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "ABFT/Freivalds delta assertion failed: layer=%g rowMaxDelta=%g "
        "colMaxDelta=%g (limit=%g)",
        args->f0, row_delta, col_delta,
        state->delta_assert_threshold);
  }
  if (!state->log_path) return iree_ok_status();
  FILE* f = fopen(state->log_path, "a");
  if (!f) return iree_ok_status();
    fprintf(
      f,
      "layer=%.15g rowMaxDelta=%.15g colMaxDelta=%.15g rowMaxRel=%.15g colMaxRel=%.15g\n",
      args->f0, row_delta, col_delta, args->f3, args->f4);
  fclose(f);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_abft_analysis_module_abft_log_rowcol_debug,
                   iree_abft_analysis_module_state_t, fffff, v) {
  if (!state->log_path) return iree_ok_status();
  FILE* f = fopen(state->log_path, "a");
  if (!f) return iree_ok_status();
    fprintf(f,
      "layer=%.15g rowExpMax=%.15g rowCalcMax=%.15g colExpMax=%.15g colCalcMax=%.15g\n",
      args->f0, args->f1, args->f2, args->f3, args->f4);
  fclose(f);
  return iree_ok_status();
}


// Returns the number of detected failures recorded by this module state.
IREE_VM_ABI_EXPORT(iree_abft_analysis_module_abft_get_failure_count,
                   iree_abft_analysis_module_state_t, v, I) {
  rets->i0 = state->failure_count;
  return iree_ok_status();
}

static const iree_vm_native_function_ptr_t iree_abft_analysis_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)               \
  {                                                                     \
      .shim = (iree_vm_native_function_shim_t)                          \
          iree_vm_shim_##arg_types##_##ret_types,                       \
      .target = (iree_vm_native_function_target_t)(target_fn),          \
  },
#include "iree/modules/abft_analysis/exports.inl"
#undef EXPORT_FN
};

static const iree_vm_native_import_descriptor_t iree_abft_analysis_module_imports_[1];

static const iree_vm_native_export_descriptor_t iree_abft_analysis_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)               \
  {                                                                     \
      .local_name = iree_string_view_literal(name),                     \
      .calling_convention =                                            \
          iree_string_view_literal("0" #arg_types "_" #ret_types),      \
      .attr_count = 0,                                                  \
      .attrs = NULL,                                                    \
  },
#include "iree/modules/abft_analysis/exports.inl"
#undef EXPORT_FN
};

static const iree_vm_native_module_descriptor_t iree_abft_analysis_module_descriptor_ = {
    .name = iree_string_view_literal("abft_analysis"),
    .version = IREE_ABFT_ANALYSIS_MODULE_VERSION_LATEST,
    .attr_count = 0,
    .attrs = NULL,
    .dependency_count = 0,
    .dependencies = NULL,
    .import_count = 0,
    .imports = iree_abft_analysis_module_imports_,
    .export_count = IREE_ARRAYSIZE(iree_abft_analysis_module_exports_),
    .exports = iree_abft_analysis_module_exports_,
    .function_count = IREE_ARRAYSIZE(iree_abft_analysis_module_funcs_),
    .functions = iree_abft_analysis_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_abft_analysis_module_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** IREE_RESTRICT out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  static const iree_vm_module_t interface = {
      .destroy = iree_abft_analysis_module_destroy,
      .alloc_state = iree_abft_analysis_module_alloc_state,
      .free_state = iree_abft_analysis_module_free_state,
      .fork_state = iree_abft_analysis_module_fork_state,
      .notify = iree_abft_analysis_module_notify,
  };

  iree_vm_module_t* base_module = NULL;
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_abft_analysis_module_t);
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);

  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_abft_analysis_module_descriptor_, instance,
      host_allocator, base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_abft_analysis_module_t* module = IREE_ABFT_ANALYSIS_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;

  *out_module = base_module;
  return iree_ok_status();
}
