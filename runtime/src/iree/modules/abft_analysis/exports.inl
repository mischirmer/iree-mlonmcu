// Minimal export list for abft_analysis module.
// Keep the functions sorted by name.

// clang-format off

EXPORT_FN("abft_get_failure_count", iree_abft_analysis_module_abft_get_failure_count, v, I)
EXPORT_FN("abft_log_rowcol_debug", iree_abft_analysis_module_abft_log_rowcol_debug, fffff, v)
EXPORT_FN("abft_log_rowcol_delta", iree_abft_analysis_module_abft_log_rowcol_delta, fff, v)
EXPORT_FN("abft_log_rowcol_metrics", iree_abft_analysis_module_abft_log_rowcol_metrics, fffff, v)
EXPORT_FN("abft_report_failure", iree_abft_analysis_module_abft_report_failure, f, v)

// clang-format on
