#ifndef IREE_MODULES_ABFT_ANALYSIS_MODULE_H_
#define IREE_MODULES_ABFT_ANALYSIS_MODULE_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a minimal abft_analysis native module that exposes reporting helpers.
IREE_API_EXPORT iree_status_t iree_abft_analysis_module_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** IREE_RESTRICT out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_ABFT_ANALYSIS_MODULE_H_
