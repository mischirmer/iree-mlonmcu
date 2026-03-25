#!/usr/bin/env bash
set -euo pipefail

# Override with: TMP_DIR=/path/to/tmp ./vgg_threshold.sh
TMP_DIR="${TMP_DIR:-/tmp/iree_vgg_abyzft}"
RESULTS_DIR="${TMP_DIR}/partitioned"
mkdir -p "${TMP_DIR}" "${RESULTS_DIR}"

VGG_COMPILER_MODE="${VGG_COMPILER_MODE:-abyzft}"
VGG_FREIVALDS_SCALING_MODE="${VGG_FREIVALDS_SCALING_MODE:-standard}"

GTSRB_DIR=/nas/ei/share/TUEIEDAprojects/SystemDesign/work/nas_tinyml/Simulator/simulator/Simulator/datasets/GTSRB/gtsrb/
LINALG_MLIR="${TMP_DIR}/vgg16_linalg_img2col_instrumentable.mlir"
case "${VGG_COMPILER_MODE}" in
  freivald)
    CLEAN_VMFB="${TMP_DIR}/vgg16_freivalds_only_nofi.vmfb"
    FI_VMFB=""
    ;;
  abft)
    CLEAN_VMFB="${TMP_DIR}/vgg16_abft_only_nofi.vmfb"
    FI_VMFB="${TMP_DIR}/vgg16_abft_fuc.vmfb"
    ;;
  abyzft)
    CLEAN_VMFB="${TMP_DIR}/vgg16_abft_scale_only_nofi.vmfb"
    FI_VMFB="${TMP_DIR}/vgg16_abft_fuc_scale.vmfb"
    ;;
  *)
    echo "Unsupported VGG_COMPILER_MODE=${VGG_COMPILER_MODE}" >&2
    exit 2
    ;;
esac
FORMULA_THRESHOLDS_JSON="${RESULTS_DIR}/vgg_thresholds_formula.json"

COMMON_ARGS=(
	--tmp-dir "${TMP_DIR}"
	--gtsrb-dir "${GTSRB_DIR}"
	--input-normalization imagenet_torch
	--vmfb-path "${CLEAN_VMFB}"
	--linalg-mlir-path "${LINALG_MLIR}"
	--normalization-mode equation
	--row-normalization-expr "sqrt(m*k)"
	--col-normalization-expr "sqrt(n*k)"
	--compiler-mode "${VGG_COMPILER_MODE}"
	--subset-count 10
)

if [ "${VGG_COMPILER_MODE}" = "abyzft" ]; then
	COMMON_ARGS+=(--enable-scaling --abyzft-scale-sampling-mode=2)
fi

if [ "${VGG_COMPILER_MODE}" = "freivald" ]; then
	COMMON_ARGS+=(--freivalds-scaling-mode "${VGG_FREIVALDS_SCALING_MODE}")
fi

python3 run_vgg_abft_normalized_summary.py \
	"${COMMON_ARGS[@]}" \
	--output-json "${RESULTS_DIR}/vgg_thresholds_distribution.json" \
	--threshold-json "${FORMULA_THRESHOLDS_JSON}" \
	--overwrite-thresholds \
	--plot-path "${RESULTS_DIR}/vgg_threshold_boxplot.png"

# python3 evolve_threshold_equation.py --input-json "${RESULTS_DIR}/vgg_thresholds_distribution.json" --fit-both-delta-thresholds --generations 400 --population 800 --restarts 2 --seed 1 --delta-train-aggregation max --normalized-threshold-stat max --fitness-p95-weight 1.0 --fitness-max-weight 1.2 --max-normalized-cap 2.0 --cap-penalty-weight 400 --threshold-overhead-factor 1.2 --output-json "${RESULTS_DIR}/vgg_evolved_equations_strict_metrics.json" --export-equations-json "${RESULTS_DIR}/vgg_evolved_equations_strict.json" --export-thresholds-json "${RESULTS_DIR}/vgg_thresholds_generated_strict.json"

python3 run_vgg_abft_normalized_summary.py \
	"${COMMON_ARGS[@]}" \
	--redline-threshold-json "${FORMULA_THRESHOLDS_JSON}" \
	--plot-path "${RESULTS_DIR}/vgg_equation_redline_check_strict.png" \
	--output-json "${RESULTS_DIR}/vgg_thresholds_distribution_eq_strict.json"

if [ "${VGG_COMPILER_MODE}" = "freivald" ]; then
	exit 0
fi

FI_EXTRA_ARGS=()
if [ "${VGG_COMPILER_MODE}" = "abyzft" ]; then
	FI_EXTRA_ARGS+=(--enable-scaling --abyzft-scale-sampling-mode=2)
fi

python3 run_vgg_abft_normalized_summary.py \
	--tmp-dir "${TMP_DIR}" \
	--gtsrb-dir "${GTSRB_DIR}" \
	--input-normalization imagenet_torch \
	--vmfb-path "${FI_VMFB}" \
	--linalg-mlir-path "${LINALG_MLIR}" \
	--normalization-mode equation \
	--row-normalization-expr "sqrt(m*k)" \
	--col-normalization-expr "sqrt(n*k)" \
	--redline-threshold-json "${FORMULA_THRESHOLDS_JSON}" \
	--plot-path "${RESULTS_DIR}/vgg_equation_redline_check_layer0_fault10.png" \
	--output-json "${RESULTS_DIR}/vgg_thresholds_distribution_eq_layer0_fault10.json" \
	--subset-count 100 \
	--enable-fault-injection \
	--fault-value 10 \
	--fault-pattern trivial \
	--layer-index 0 \
	"${FI_EXTRA_ARGS[@]}"

python3 run_vgg_abft_normalized_summary.py \
	--tmp-dir "${TMP_DIR}" \
	--gtsrb-dir "${GTSRB_DIR}" \
	--input-normalization imagenet_torch \
	--vmfb-path "${FI_VMFB}" \
	--linalg-mlir-path "${LINALG_MLIR}" \
	--normalization-mode equation \
	--row-normalization-expr "sqrt(m*k)" \
	--col-normalization-expr "sqrt(n*k)" \
	--redline-threshold-json "${FORMULA_THRESHOLDS_JSON}" \
	--plot-path "${RESULTS_DIR}/vgg_equation_redline_check_layer8_fault2.png" \
	--output-json "${RESULTS_DIR}/vgg_thresholds_distribution_eq_layer8_fault2.json" \
	--subset-count 100 \
	--enable-fault-injection \
	--fault-value 2 \
	--fault-pattern trivial \
	--layer-index 8 \
	"${FI_EXTRA_ARGS[@]}"

python3 run_vgg_abft_normalized_summary.py \
	--tmp-dir "${TMP_DIR}" \
	--gtsrb-dir "${GTSRB_DIR}" \
	--input-normalization imagenet_torch \
	--vmfb-path "${FI_VMFB}" \
	--linalg-mlir-path "${LINALG_MLIR}" \
	--normalization-mode equation \
	--row-normalization-expr "sqrt(m*k)" \
	--col-normalization-expr "sqrt(n*k)" \
	--redline-threshold-json "${FORMULA_THRESHOLDS_JSON}" \
	--plot-path "${RESULTS_DIR}/vgg_equation_redline_check_layer15_fault10.png" \
	--output-json "${RESULTS_DIR}/vgg_thresholds_distribution_eq_layer15_fault10.json" \
	--subset-count 100 \
	--enable-fault-injection \
	--fault-value 10 \
	--fault-pattern trivial \
	--layer-index 15 \
	"${FI_EXTRA_ARGS[@]}"
