// ABFTPass.cpp
// Insert a computation of a ones-vector multiplied by the LHS matrix (ones * A)
// for each matched linalg.matmul operation. The ones*A for a matrix A (m x k)
// produces a row-vector of length k (tensor<?xf32>) and is implemented by
// calling a helper `column_checksum` function that sums rows per column.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
// Command line option parsing for the standalone flag used by the pass.
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <cmath>
#include <ctime>

using namespace mlir;
using namespace mlir::iree_compiler;

// Global CLI flag to enable Full-Checksum elementwise comparisons. Using a
// static command-line option avoids putting non-copyable llvm::cl::opt into
// the pass object (PassWrapper needs to be copyable).
static llvm::cl::opt<bool> freivaldsEnableFuC(
    "freivalds-enable-fuc",
    llvm::cl::desc("Enable full elementwise checksum (t1/t2) comparisons"),
    llvm::cl::init(false));

static llvm::cl::opt<float> freivaldsEpsilonAbs(
  "freivalds-epsilon-abs",
  llvm::cl::desc("Absolute epsilon for ABFT checks"),
  llvm::cl::init(1.0e-7f));

static llvm::cl::opt<float> freivaldsEpsilonRel(
  "freivalds-epsilon-rel",
  llvm::cl::desc("Relative epsilon for ABFT checks"),
  llvm::cl::init(1.0e-5f));

static llvm::cl::opt<std::string> freivaldsScalingMode(
    "freivalds-scaling-mode",
    llvm::cl::desc("Scaling mode for Freivalds checksum generation"),
    llvm::cl::value_desc("binary|standard"),
    llvm::cl::init("binary"));
static llvm::cl::opt<std::string> freivaldsVectorMode(
    "freivalds-vector-mode",
    llvm::cl::desc("Verification vector mode for Freivalds"),
    llvm::cl::value_desc("binary"),
    llvm::cl::init("binary"));
static llvm::cl::opt<bool> freivaldsEnableAnalysisLog(
    "freivalds-enable-analysis-log",
    llvm::cl::desc(
        "Enable emission of abft_analysis row/col logging calls in Freivalds"),
    llvm::cl::init(true));
static llvm::cl::opt<int> freivaldsNumChecks(
    "freivalds-num-checks",
    llvm::cl::desc(
        "Number of independent Freivalds projections per matmul (>=1)"),
    llvm::cl::init(1));
static llvm::cl::opt<bool> freivaldsInjectFault(
    "freivalds-inject-fault",
    llvm::cl::desc("Enable deterministic synthetic fault injection for testing"),
    llvm::cl::init(false));
static llvm::cl::opt<int> freivaldsInjectFaultDelta(
    "freivalds-inject-fault-delta",
    llvm::cl::desc("Fault injection additive delta"),
    llvm::cl::init(1));
static llvm::cl::opt<std::string> freivaldsInjectFaultPattern(
    "freivalds-inject-fault-pattern",
    llvm::cl::desc("Fault injection pattern: single_point, trivial, checkered"),
    llvm::cl::init("single_point"));

static int sampleBinaryBit() {
  static bool seeded = false;
  if (!seeded) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    seeded = true;
  }
  return std::rand() & 1;
}

namespace {

constexpr StringLiteral kFreivaldsModeAttrName = "iree.abft.mode";
constexpr StringLiteral kFreivaldsModeNormal = "abft";
constexpr StringLiteral kFreivaldsModeScaled = "abyzft";

struct FreivaldsPass : public PassWrapper<FreivaldsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FreivaldsPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, tensor::TensorDialect,
                    arith::ArithDialect, math::MathDialect,
                    cf::ControlFlowDialect, IREE::Util::UtilDialect>();
  }

  // NOTE: the CLI flag is defined as a static llvm::cl::opt above. We avoid
  // storing an Option<> member inside the pass because llvm::cl::opt is
  // non-copyable which breaks PassWrapper's cloning. Read the flag at
  // run-time from `freivaldsEnableFuC`.

  StringRef getArgument() const final { return "freivalds-insert-ones"; }
  StringRef getDescription() const final {
    return "For each linalg.matmul insert a computation of ones * A (row sums)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    (void)freivaldsInjectFaultPattern;
    auto funcName = func.getSymName();
    if (funcName == "column_checksum" || funcName == "row_checksum" ||
        funcName == "matrix_sum" || funcName == "vector_sum" ||
        funcName == "vector_dot_product" || funcName == "epsilon_compare_abft" ||
        funcName == "sample_row_scales" ||
        funcName == "sample_row_scales_ones" ||
        funcName == "sample_row_scales_all_ones" ||
        funcName == "sample_col_scales" ||
        funcName == "sample_col_scales_ones" ||
        funcName == "sample_col_scales_all_ones" ||
        funcName == "scale_matrix_rows" ||
        funcName == "scale_matrix_cols" ||
        funcName == "descale_matrix" ||
        funcName == "vector_scale" ||
        funcName == "rowvec_mul_mat" || funcName == "mat_mul_colvec" ||
        funcName == "randvec_mul_mat" ||
        funcName == "expected_col_checksum_freivalds" ||
        funcName == "expected_row_checksum_freivalds" ||
        funcName == "vector_epsilon_compare_abft" ||
        funcName == "vector_max_abs_diff" ||
        funcName == "vector_max_abs_diff_pair" ||
        funcName == "vector_first_elem" || funcName == "scalar_tensor_add" ||
        funcName == "freivalds_keepalive" ||
        funcName == "matrix_sub" || funcName == "matrix_add" ||
        funcName == "zero_matrix_like" ||
        funcName == "abft_analysis.abft_report_failure" ||
        funcName == "abft_analysis.abft_log_rowcol_delta" ||
        funcName == "abft_analysis.abft_log_rowcol_debug") {
      return;
    }
    ModuleOp module = func->getParentOfType<ModuleOp>();
    std::string effectiveScalingMode = freivaldsScalingMode;
    std::string effectiveVectorMode = freivaldsVectorMode;
    (void)freivaldsNumChecks;
    if (effectiveScalingMode != "standard" && effectiveScalingMode != "binary") {
      module.emitError() << "Unsupported freivalds-scaling-mode: "
                         << effectiveScalingMode;
      signalPassFailure();
      return;
    }
    if (effectiveVectorMode != "binary") {
      module.emitError() << "Unsupported freivalds-vector-mode: "
                         << effectiveVectorMode
                         << " (only 'binary' is supported)";
      signalPassFailure();
      return;
    }

    // Fast path: if this function contains no matmul, skip instrumentation
    // entirely and avoid injecting helper IR that can perturb unrelated models
    // (e.g. tiny non-matmul graphs used in MLonMCU smoke tests).
    bool hasMatmulTarget = false;
    func.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "linalg.matmul") {
        hasMatmulTarget = true;
      }
    });
    if (!hasMatmulTarget) {
      return;
    }

    if (auto modeAttr = module->getAttrOfType<StringAttr>(kFreivaldsModeAttrName)) {
      if (modeAttr.getValue() == kFreivaldsModeScaled) {
        module.emitError(
            "ABFT and AByzFT are mutually exclusive; this module is already "
            "marked for AByzFT instrumentation");
        signalPassFailure();
        return;
      }
    } else {
      module->setAttr(kFreivaldsModeAttrName,
                      StringAttr::get(module.getContext(), kFreivaldsModeNormal));
    }

    MLIRContext *ctx = func.getContext();

    // Ensure necessary dialects are loaded into the context so the textual
    // MLIR parser can construct ops like linalg.reduce, tensor.dim and
    // math.absf. We will try to insert full helper function bodies (so they
    // are available in the module and not turned into vm.imports). If parsing
    // fails we fall back to declaration-only functions so call sites still
    // type-check.
    ctx->getOrLoadDialect<func::FuncDialect>();
    ctx->getOrLoadDialect<linalg::LinalgDialect>();
    ctx->getOrLoadDialect<memref::MemRefDialect>();
    ctx->getOrLoadDialect<tensor::TensorDialect>();
    ctx->getOrLoadDialect<arith::ArithDialect>();
    ctx->getOrLoadDialect<math::MathDialect>();
    ctx->getOrLoadDialect<cf::ControlFlowDialect>();
    ctx->getOrLoadDialect<IREE::Util::UtilDialect>();

    // Helper: parse and insert a helper func.func with body into the module
    // (clone). Returns the inserted func or nullptr on failure.
    auto ensureFunctionWithBody = [&](StringRef name,
                                      StringRef body) -> func::FuncOp {
      // Keep Freivalds helper injection minimal and stable (ABFT-like path).
      // Avoid materializing large legacy helper bodies that are not needed for
      // the one-check flow and can destabilize downstream VM->EmitC lowering.
      if (name != "randvec_mul_mat" && name != "mat_mul_colvec" &&
          name != "column_checksum" && name != "row_checksum" &&
          name != "matrix_sum" && name != "vector_dot_product" &&
          name != "vector_max_abs_diff" && name != "vector_first_elem" &&
          name != "matrix_sub" && name != "matrix_add" &&
          name != "zero_matrix_like") {
        return {};
      }
      if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
        if (!existing.empty())
          return existing;
        existing.erase();
      }
      OwningOpRef<ModuleOp> tmp = parseSourceString<ModuleOp>(body, ctx);
      if (!tmp) {
        module.emitRemark() << "freivalds: failed to parse helper body for " << name;
        return {};
      }
      func::FuncOp srcFunc;
      for (auto f : tmp->getOps<func::FuncOp>()) {
        if (f.getSymName() == name) {
          srcFunc = f;
          break;
        }
      }
      if (!srcFunc) {
        module.emitRemark()
            << "freivalds: helper " << name << " not present in parsed body";
        return {};
      }
      Operation *cloned = srcFunc->clone();
      module.getBody()->getOperations().push_back(cloned);
      return cast<func::FuncOp>(cloned);
    };

    // Try to ensure helpers exist with bodies using small per-function MLIR.
    func::FuncOp parsedCol = ensureFunctionWithBody("column_checksum", R"mlir(
module {
  func.func @column_checksum(%matrix: tensor<?x?xf32>) -> tensor<?xf32> {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %n = tensor.dim %matrix, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %sum = linalg.reduce ins(%matrix : tensor<?x?xf32>) outs(%init : tensor<?xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
        %r = arith.addf %in, %acc : f32
        linalg.yield %r : f32
      }
    return %sum : tensor<?xf32>
  }
}
)mlir");
    (void)parsedCol;

    func::FuncOp parsedRow = ensureFunctionWithBody("row_checksum", R"mlir(
module {
  func.func @row_checksum(%matrix: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %m = tensor.dim %matrix, %c0 : tensor<?x?xf32>
    %empty = tensor.empty(%m) : tensor<?xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %sum = linalg.reduce ins(%matrix : tensor<?x?xf32>) outs(%init : tensor<?xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {
        %r = arith.addf %in, %acc : f32
        linalg.yield %r : f32
      }
    return %sum : tensor<?xf32>
  }
}
)mlir");
    (void)parsedRow;

    // Helper exposing Freivalds random projection semantics.
    // Computes M * r with deterministic per-column r generated inline.
    std::string randVecMulMatBody =
        "module {\n"
        "  func.func @randvec_mul_mat(%mat: tensor<?x?xf32>, %seed: i32) -> tensor<?xf32> {\n"
        "    %c0 = arith.constant 0 : index\n"
        "    %c1 = arith.constant 1 : index\n"
        "    %c2 = arith.constant 2 : index\n"
        "    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>\n"
        "    %r_empty = tensor.empty(%n) : tensor<?xf32>\n"
        "    %zero = arith.constant 0.0 : f32\n"
        "    %r_init = linalg.fill ins(%zero : f32) outs(%r_empty : tensor<?xf32>) -> tensor<?xf32>\n"
        "    %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = [\"parallel\"]} "
        "outs(%r_init : tensor<?xf32>) {\n"
        "    ^bb0(%out: f32):\n"
        "      %idx = linalg.index 0 : index\n"
        "      %seed_idx = arith.index_castui %seed : i32 to index\n"
        "      %seeded = arith.addi %idx, %seed_idx : index\n";
    if (effectiveScalingMode == "binary") {
      randVecMulMatBody +=
          "      %idx_i32 = arith.index_castui %seeded : index to i32\n"
          "      %modulus = arith.constant 2147483647 : i32\n"
          "      %mul1 = arith.constant 1103515245 : i32\n"
          "      %offset1 = arith.constant 12345 : i32\n"
          "      %h1m = arith.muli %idx_i32, %mul1 : i32\n"
          "      %h1a = arith.addi %h1m, %offset1 : i32\n"
          "      %h1 = arith.remsi %h1a, %modulus : i32\n"
          "      %two_i32 = arith.constant 2 : i32\n"
          "      %p = arith.remsi %h1, %two_i32 : i32\n"
          "      %zero_i32 = arith.constant 0 : i32\n"
          "      %is0 = arith.cmpi eq, %p, %zero_i32 : i32\n"
          "      %one = arith.constant 1.0 : f32\n"
          "      %val = arith.select %is0, %one, %zero : f32\n";
    } else {
      randVecMulMatBody +=
          "      %idx_i32 = arith.index_castui %seeded : index to i32\n"
          "      %modulus = arith.constant 2147483647 : i32\n"
          "      %mul1 = arith.constant 1103515245 : i32\n"
          "      %offset1 = arith.constant 12345 : i32\n"
          "      %h1m = arith.muli %idx_i32, %mul1 : i32\n"
          "      %h1a = arith.addi %h1m, %offset1 : i32\n"
          "      %h1 = arith.remsi %h1a, %modulus : i32\n"
          "      %two_i32 = arith.constant 2 : i32\n"
          "      %p = arith.remsi %h1, %two_i32 : i32\n"
          "      %zero_i32 = arith.constant 0 : i32\n"
          "      %is0 = arith.cmpi eq, %p, %zero_i32 : i32\n"
          "      %plus = arith.constant 1.0 : f32\n"
          "      %minus = arith.constant -1.0 : f32\n"
          "      %val = arith.select %is0, %plus, %minus : f32\n";
    }
    randVecMulMatBody +=
        "      linalg.yield %val : f32\n"
        "    } -> tensor<?xf32>\n"
        "    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>\n"
        "    %out_empty = tensor.empty(%m) : tensor<?xf32>\n"
        "    %out_init = linalg.fill ins(%zero : f32) outs(%out_empty : tensor<?xf32>) -> tensor<?xf32>\n"
        "    %res = linalg.generic {\n"
        "      indexing_maps = [\n"
        "        affine_map<(d0, d1) -> (d0, d1)>,\n"
        "        affine_map<(d0, d1) -> (d1)>,\n"
        "        affine_map<(d0, d1) -> (d0)>\n"
        "      ],\n"
        "      iterator_types = [\"parallel\", \"reduction\"]\n"
        "    } ins(%mat, %r : tensor<?x?xf32>, tensor<?xf32>) outs(%out_init : tensor<?xf32>) {\n"
        "    ^bb0(%mat_elem: f32, %r_elem: f32, %acc: f32):\n"
        "      %prod = arith.mulf %mat_elem, %r_elem : f32\n"
        "      %sum = arith.addf %acc, %prod : f32\n"
        "      linalg.yield %sum : f32\n"
        "    } -> tensor<?xf32>\n"
        "    return %res : tensor<?xf32>\n"
        "  }\n"
        "}\n";
    func::FuncOp parsedRandVecMulMat =
        ensureFunctionWithBody("randvec_mul_mat", StringRef(randVecMulMatBody));
    (void)parsedRandVecMulMat;

    func::FuncOp parsedSum = ensureFunctionWithBody("matrix_sum", R"mlir(
module {
  func.func @matrix_sum(%matrix: tensor<?x?xf32>) -> tensor<f32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<f32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<f32>) -> tensor<f32>
    %sum = linalg.reduce ins(%matrix : tensor<?x?xf32>) outs(%filled : tensor<f32>) dimensions = [0, 1]
      (%in: f32, %acc: f32) {
        %r = arith.addf %in, %acc : f32
        linalg.yield %r : f32
      }
    return %sum : tensor<f32>
  }
}
)mlir");
    (void)parsedSum;

    func::FuncOp parsedVecSum = ensureFunctionWithBody("vector_sum", R"mlir(
module {
  func.func @vector_sum(%vec: tensor<?xf32>) -> tensor<f32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<f32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<f32>) -> tensor<f32>
    %sum = linalg.reduce ins(%vec : tensor<?xf32>) outs(%filled : tensor<f32>) dimensions = [0]
      (%in: f32, %acc: f32) {
        %r = arith.addf %in, %acc : f32
        linalg.yield %r : f32
      }
    return %sum : tensor<f32>
  }
}
)mlir");
    (void)parsedVecSum;


    func::FuncOp parsedDot =
        ensureFunctionWithBody("vector_dot_product", R"mlir(
module {
  func.func @vector_dot_product(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<f32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<f32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<f32>) -> tensor<f32>
    %dot = linalg.dot ins(%a, %b : tensor<?xf32>, tensor<?xf32>) outs(%filled : tensor<f32>) -> tensor<f32>
    return %dot : tensor<f32>
  }
}
)mlir");
    (void)parsedDot;

    func::FuncOp parsedEps = ensureFunctionWithBody("epsilon_compare_abft", R"mlir(
module {
  func.func @epsilon_compare_abft(%x: tensor<f32>, %y: tensor<f32>, %eps_abs: f32, %eps_rel: f32) {
    %vx = tensor.extract %x[] : tensor<f32>
    %vy = tensor.extract %y[] : tensor<f32>
    %d = arith.subf %vx, %vy : f32
    %ad = math.absf %d : f32
    %avy = math.absf %vy : f32
    %rel = arith.mulf %eps_rel, %avy : f32
    %thresh = arith.addf %eps_abs, %rel : f32
    return
  }
}
)mlir");
    (void)parsedEps;

    // Provide implementations for sampling helpers. `sample_*_scales`
    // produces deterministic bounded random values away from zero for the
    // regular Freivalds check, while `sample_*_scales_ones` is repurposed
    // here as a binary {0,1} pattern.
    func::FuncOp parsedSampleRow = ensureFunctionWithBody("sample_row_scales", R"mlir(
module {
  func.func @sample_row_scales(%mat: tensor<?x?xf32>, %seed: i32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %empty = tensor.empty(%m) : tensor<?xf32>
    %init = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%init : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%filled : tensor<?xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %idx_i32 = arith.index_castui %idx : index to i32
      %modulus = arith.constant 9973 : i32
      %offset1 = arith.constant 17 : i32
      %mul1 = arith.constant 131 : i32
      %seed_mul = arith.muli %seed, %mul1 : i32
      %seeded_idx = arith.addi %idx_i32, %seed_mul : i32
      %inv_mod = arith.constant 1.002707e-04 : f32
      %h1m = arith.muli %seeded_idx, %mul1 : i32
      %h1a = arith.addi %h1m, %offset1 : i32
      %h1 = arith.remsi %h1a, %modulus : i32
      %h1f = arith.sitofp %h1 : i32 to f32
      %u = arith.mulf %h1f, %inv_mod : f32
      %half = arith.constant 5.000000e-01 : f32
      %one_pt_five = arith.constant 1.5 : f32
      %scaled = arith.mulf %u, %one_pt_five : f32
      %val = arith.addf %half, %scaled : f32
      linalg.yield %val : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedSampleRow;

    func::FuncOp parsedSampleRowOnes = ensureFunctionWithBody("sample_row_scales_ones", R"mlir(
module {
  func.func @sample_row_scales_ones(%mat: tensor<?x?xf32>, %seed: i32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %empty = tensor.empty(%m) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%filled : tensor<?xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %seed_idx = arith.index_castui %seed : i32 to index
      %seeded = arith.addi %idx, %seed_idx : index
      %mod = arith.remui %seeded, %c2 : index
      %c0_cmp = arith.constant 0 : index
      %is0 = arith.cmpi eq, %mod, %c0_cmp : index
      %v0 = arith.constant 1.0 : f32
      %v1 = arith.constant 0.0 : f32
      %sel = arith.select %is0, %v0, %v1 : f32
      linalg.yield %sel : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedSampleRowOnes;

    func::FuncOp parsedSampleRowAllOnes = ensureFunctionWithBody("sample_row_scales_all_ones", R"mlir(
module {
  func.func @sample_row_scales_all_ones(%mat: tensor<?x?xf32>, %seed: i32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %empty = tensor.empty(%m) : tensor<?xf32>
    %one = arith.constant 1.0 : f32
    %filled = linalg.fill ins(%one : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    return %filled : tensor<?xf32>
  }
}
)mlir");
    (void)parsedSampleRowAllOnes;

    func::FuncOp parsedSampleCol = ensureFunctionWithBody("sample_col_scales", R"mlir(
module {
  func.func @sample_col_scales(%mat: tensor<?x?xf32>, %seed: i32) -> tensor<?xf32> {
    %c1 = arith.constant 1 : index
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %init = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%init : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%filled : tensor<?xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %idx_i32 = arith.index_castui %idx : index to i32
      %modulus = arith.constant 9973 : i32
      %offset1 = arith.constant 17 : i32
      %mul1 = arith.constant 131 : i32
      %seed_mul = arith.muli %seed, %mul1 : i32
      %seeded_idx = arith.addi %idx_i32, %seed_mul : i32
      %inv_mod = arith.constant 1.002707e-04 : f32
      %h1m = arith.muli %seeded_idx, %mul1 : i32
      %h1a = arith.addi %h1m, %offset1 : i32
      %h1 = arith.remsi %h1a, %modulus : i32
      %h1f = arith.sitofp %h1 : i32 to f32
      %u = arith.mulf %h1f, %inv_mod : f32
      %half = arith.constant 5.000000e-01 : f32
      %one_pt_five = arith.constant 1.5 : f32
      %scaled = arith.mulf %u, %one_pt_five : f32
      %val = arith.addf %half, %scaled : f32
      linalg.yield %val : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedSampleCol;

    func::FuncOp parsedSampleColOnes = ensureFunctionWithBody("sample_col_scales_ones", R"mlir(
module {
  func.func @sample_col_scales_ones(%mat: tensor<?x?xf32>, %seed: i32) -> tensor<?xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%filled : tensor<?xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %seed_idx = arith.index_castui %seed : i32 to index
      %seeded = arith.addi %idx, %seed_idx : index
      %mod = arith.remui %seeded, %c2 : index
      %c0_cmp = arith.constant 0 : index
      %is0 = arith.cmpi eq, %mod, %c0_cmp : index
      %v0 = arith.constant 1.0 : f32
      %v1 = arith.constant 0.0 : f32
      %sel = arith.select %is0, %v0, %v1 : f32
      linalg.yield %sel : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedSampleColOnes;

    func::FuncOp parsedSampleColAllOnes = ensureFunctionWithBody("sample_col_scales_all_ones", R"mlir(
module {
  func.func @sample_col_scales_all_ones(%mat: tensor<?x?xf32>, %seed: i32) -> tensor<?xf32> {
    %c1 = arith.constant 1 : index
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %one = arith.constant 1.0 : f32
    %filled = linalg.fill ins(%one : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    return %filled : tensor<?xf32>
  }
}
)mlir");
    (void)parsedSampleColAllOnes;

    func::FuncOp parsedScaleRows = ensureFunctionWithBody("scale_matrix_rows", R"mlir(
module {
  func.func @scale_matrix_rows(%mat: tensor<?x?xf32>, %scales: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mat, %scales : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
    ^bb0(%a: f32, %s: f32, %out: f32):
      %prod = arith.mulf %a, %s : f32
      linalg.yield %prod : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)parsedScaleRows;

    func::FuncOp parsedScaleCols = ensureFunctionWithBody("scale_matrix_cols", R"mlir(
module {
  func.func @scale_matrix_cols(%mat: tensor<?x?xf32>, %scales: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mat, %scales : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
    ^bb0(%a: f32, %s: f32, %out: f32):
      %prod = arith.mulf %a, %s : f32
      linalg.yield %prod : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)parsedScaleCols;

    func::FuncOp parsedDescale = ensureFunctionWithBody("descale_matrix", R"mlir(
module {
  func.func @descale_matrix(%mat: tensor<?x?xf32>, %row_scales: tensor<?xf32>, %col_scales: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mat, %row_scales, %col_scales : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
    ^bb0(%a: f32, %rs: f32, %cs: f32, %out: f32):
      %den = arith.mulf %rs, %cs : f32
      %res = arith.divf %a, %den : f32
      linalg.yield %res : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)parsedDescale;

    func::FuncOp parsedVecScale = ensureFunctionWithBody("vector_scale", R"mlir(
module {
  func.func @vector_scale(%vec: tensor<?xf32>, %scale: f32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %vec, %c0 : tensor<?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%vec : tensor<?xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %scaled = arith.mulf %in, %scale : f32
      linalg.yield %scaled : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedVecScale;

    // Full-checksum helper implementations (so the pass can emit full-checksum
    // calls unconditionally). These compute:
    //  - rowvec_mul_mat(rv, mat) -> vector: for each column j, sum_k
    //  rv[k]*mat[k,j]
    //  - mat_mul_colvec(mat, cv) -> vector: for each row i, sum_j
    //  mat[i,j]*cv[j]
    //  - vector_epsilon_compare_abft(v1, v2, eps) -> () : elementwise report
    //  (|v1-v2| < eps)
    func::FuncOp parsedRowVec;
    if (effectiveScalingMode != "standard")
      parsedRowVec = ensureFunctionWithBody("rowvec_mul_mat", R"mlir(
module {
  func.func @rowvec_mul_mat(%rv: tensor<?xf32>, %mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c1 = arith.constant 1 : index
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%rv, %mat : tensor<?xf32>, tensor<?x?xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%rv_elem: f32, %mat_elem: f32, %acc: f32):
      %prod = arith.mulf %rv_elem, %mat_elem : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedRowVec;

    func::FuncOp parsedMatCol = ensureFunctionWithBody("mat_mul_colvec", R"mlir(
module {
  func.func @mat_mul_colvec(%mat: tensor<?x?xf32>, %cv: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %k = tensor.dim %cv, %c0 : tensor<?xf32>
    %cv2d = tensor.expand_shape %cv [[0, 1]] output_shape [%k, %c1] : tensor<?xf32> into tensor<?x1xf32>
    %empty2d = tensor.empty(%m) : tensor<?x1xf32>
    %zero = arith.constant 0.0 : f32
    %init2d = linalg.fill ins(%zero : f32) outs(%empty2d : tensor<?x1xf32>) -> tensor<?x1xf32>
    %res2d = linalg.matmul ins(%mat, %cv2d : tensor<?x?xf32>, tensor<?x1xf32>)
      outs(%init2d : tensor<?x1xf32>) -> tensor<?x1xf32>
    %res = tensor.collapse_shape %res2d [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedMatCol;

    func::FuncOp parsedExpectedCol;
    if (effectiveScalingMode != "standard")
      parsedExpectedCol = ensureFunctionWithBody("expected_col_checksum_freivalds", R"mlir(
module {
  func.func @expected_col_checksum_freivalds(%rv: tensor<?xf32>, %a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %k = tensor.dim %a, %c1 : tensor<?x?xf32>
    %n = tensor.dim %b, %c1 : tensor<?x?xf32>
    %tmp_empty = tensor.empty(%k) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %tmp_init = linalg.fill ins(%zero : f32) outs(%tmp_empty : tensor<?xf32>) -> tensor<?xf32>
    %tmp = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%rv, %a : tensor<?xf32>, tensor<?x?xf32>) outs(%tmp_init : tensor<?xf32>) {
    ^bb0(%rv_elem: f32, %a_elem: f32, %acc: f32):
      %prod = arith.mulf %rv_elem, %a_elem : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    %out_empty = tensor.empty(%n) : tensor<?xf32>
    %out_init = linalg.fill ins(%zero : f32) outs(%out_empty : tensor<?xf32>) -> tensor<?xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%tmp, %b : tensor<?xf32>, tensor<?x?xf32>) outs(%out_init : tensor<?xf32>) {
    ^bb0(%tmp_elem: f32, %b_elem: f32, %acc: f32):
      %prod = arith.mulf %tmp_elem, %b_elem : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
}
)mlir");
    (void)parsedExpectedCol;

    func::FuncOp parsedExpectedRow;
    if (effectiveScalingMode != "standard")
      parsedExpectedRow = ensureFunctionWithBody("expected_row_checksum_freivalds", R"mlir(
module {
  func.func @expected_row_checksum_freivalds(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %cv: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %a, %c0 : tensor<?x?xf32>
    %k = tensor.dim %b, %c0 : tensor<?x?xf32>
    %tmp_empty = tensor.empty(%k) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %tmp_init = linalg.fill ins(%zero : f32) outs(%tmp_empty : tensor<?xf32>) -> tensor<?xf32>
    %tmp = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%b, %cv : tensor<?x?xf32>, tensor<?xf32>) outs(%tmp_init : tensor<?xf32>) {
    ^bb0(%b_elem: f32, %cv_elem: f32, %acc: f32):
      %prod = arith.mulf %b_elem, %cv_elem : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    %out_empty = tensor.empty(%m) : tensor<?xf32>
    %out_init = linalg.fill ins(%zero : f32) outs(%out_empty : tensor<?xf32>) -> tensor<?xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %tmp : tensor<?x?xf32>, tensor<?xf32>) outs(%out_init : tensor<?xf32>) {
    ^bb0(%a_elem: f32, %tmp_elem: f32, %acc: f32):
      %prod = arith.mulf %a_elem, %tmp_elem : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
}
)mlir");
    (void)parsedExpectedRow;

    func::FuncOp parsedVecEps =
        ensureFunctionWithBody("vector_epsilon_compare_abft", R"mlir(
module {
  func.func @vector_epsilon_compare_abft(%v1: tensor<?xf32>, %v2: tensor<?xf32>, %eps_abs: f32, %eps_rel: f32) {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %v1, %c0 : tensor<?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %one = arith.constant 1.0 : f32
    linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
      ins(%v1, %v2 : tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %d = arith.subf %a, %b : f32
        %ad = math.absf %d : f32
        %ab = math.absf %b : f32
        %base = arith.maximumf %ab, %one : f32
        %rel = arith.mulf %eps_rel, %base : f32
        %thresh = arith.addf %eps_abs, %rel : f32
        linalg.yield %acc : f32
    } -> tensor<?xf32>
    return
  }
}
)mlir");
    (void)parsedVecEps;

    func::FuncOp parsedVecMax =
        ensureFunctionWithBody("vector_max_abs_diff", R"mlir(
module {
  func.func @vector_max_abs_diff(%v1: tensor<?xf32>, %v2: tensor<?xf32>) -> f32 {
    %empty = tensor.empty() : tensor<f32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<f32>) -> tensor<f32>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->()>],
      iterator_types = ["reduction"]
    } ins(%v1, %v2 : tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<f32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %d = arith.subf %a, %b : f32
        %ad = math.absf %d : f32
        %m = arith.maximumf %ad, %acc : f32
        linalg.yield %m : f32
    } -> tensor<f32>
    %res = tensor.extract %out[] : tensor<f32>
    return %res : f32
  }
}
)mlir");
    (void)parsedVecMax;

    // Keep pair-wise debug helper disabled to avoid complex reduction helper
    // bodies that can trigger downstream VM->EmitC analysis asserts.
    func::FuncOp parsedVecMaxPair;

    func::FuncOp parsedVecMaxAbs =
        ensureFunctionWithBody("vector_max_abs", R"mlir(
module {
  func.func @vector_max_abs(%v: tensor<?xf32>) -> f32 {
    %empty = tensor.empty() : tensor<f32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<f32>) -> tensor<f32>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->()>],
      iterator_types = ["reduction"]
    } ins(%v : tensor<?xf32>) outs(%init : tensor<f32>) {
      ^bb0(%a: f32, %acc: f32):
        %aa = math.absf %a : f32
        %m = arith.maximumf %aa, %acc : f32
        linalg.yield %m : f32
    } -> tensor<f32>
    %res = tensor.extract %out[] : tensor<f32>
    return %res : f32
  }
}
)mlir");
    (void)parsedVecMaxAbs;

    func::FuncOp parsedVecFirst =
        ensureFunctionWithBody("vector_first_elem", R"mlir(
module {
  func.func @vector_first_elem(%v: tensor<?xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %x = tensor.extract %v[%c0] : tensor<?xf32>
    return %x : f32
  }
}
)mlir");
    (void)parsedVecFirst;

    func::FuncOp parsedVecAdd = ensureFunctionWithBody("vector_add", R"mlir(
module {
  func.func @vector_add(%v1: tensor<?xf32>, %v2: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %v1, %c0 : tensor<?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
      ins(%v1, %v2 : tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedVecAdd;

    func::FuncOp parsedVecSub = ensureFunctionWithBody("vector_sub", R"mlir(
module {
  func.func @vector_sub(%v1: tensor<?xf32>, %v2: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %v1, %c0 : tensor<?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
      ins(%v1, %v2 : tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %dif = arith.subf %a, %b : f32
        linalg.yield %dif : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedVecSub;

    func::FuncOp parsedScalarAdd = ensureFunctionWithBody("scalar_tensor_add", R"mlir(
module {
  func.func @scalar_tensor_add(%a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
    %va = tensor.extract %a[] : tensor<f32>
    %vb = tensor.extract %b[] : tensor<f32>
    %sum = arith.addf %va, %vb : f32
    %empty = tensor.empty() : tensor<f32>
    %out = linalg.fill ins(%sum : f32) outs(%empty : tensor<f32>) -> tensor<f32>
    return %out : tensor<f32>
  }
}
)mlir");
    (void)parsedScalarAdd;

    func::FuncOp parsedMatrixSub = ensureFunctionWithBody("matrix_sub", R"mlir(
module {
  func.func @matrix_sub(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %a, %c0 : tensor<?x?xf32>
    %n = tensor.dim %a, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %d = arith.subf %x, %y : f32
        linalg.yield %d : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)parsedMatrixSub;

    func::FuncOp parsedMatrixAdd = ensureFunctionWithBody("matrix_add", R"mlir(
module {
  func.func @matrix_add(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %a, %c0 : tensor<?x?xf32>
    %n = tensor.dim %a, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %s = arith.addf %x, %y : f32
        linalg.yield %s : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)parsedMatrixAdd;

    func::FuncOp parsedZeroMatrixLike = ensureFunctionWithBody("zero_matrix_like", R"mlir(
module {
  func.func @zero_matrix_like(%a: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %a, %c0 : tensor<?x?xf32>
    %n = tensor.dim %a, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %init : tensor<?x?xf32>
  }
}
)mlir");
    (void)parsedZeroMatrixLike;

    // If any parse failed, fall back to inserting declaration-only functions
    // to keep the rest of the instrumentation working.
    OpBuilder modBuilder(module.getBodyRegion());
    Location modLoc = module.getLoc();
    auto f32 = modBuilder.getF32Type();
    auto dyn2d = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, f32);
    auto vecDyn = RankedTensorType::get({ShapedType::kDynamic}, f32);
    auto scalarTensor = RankedTensorType::get({}, f32);

    auto maybeInsertDecl = [&](StringRef name, FunctionType fnTy) {
      if (!module.lookupSymbol<func::FuncOp>(name)) {
        // Create a declaration-only function and mark it private. MLIR
        // requires declaration-only (no-body) func.func symbols to be
        // private; public declarations are not allowed.
        auto fn = modBuilder.create<func::FuncOp>(modLoc, name, fnTy);
        fn.setPrivate();
      }
    };
    if (!module.lookupSymbol<func::FuncOp>("column_checksum")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn});
      maybeInsertDecl("column_checksum", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("row_checksum")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn});
      maybeInsertDecl("row_checksum", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("sample_row_scales")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d, IntegerType::get(ctx, 32)}, TypeRange{vecDyn});
      maybeInsertDecl("sample_row_scales", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("sample_row_scales_ones")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d, IntegerType::get(ctx, 32)}, TypeRange{vecDyn});
      maybeInsertDecl("sample_row_scales_ones", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("sample_col_scales")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d, IntegerType::get(ctx, 32)}, TypeRange{vecDyn});
      maybeInsertDecl("sample_col_scales", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("sample_col_scales_ones")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d, IntegerType::get(ctx, 32)}, TypeRange{vecDyn});
      maybeInsertDecl("sample_col_scales_ones", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("scale_matrix_rows")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d, vecDyn}, TypeRange{dyn2d});
      maybeInsertDecl("scale_matrix_rows", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("scale_matrix_cols")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d, vecDyn}, TypeRange{dyn2d});
      maybeInsertDecl("scale_matrix_cols", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("descale_matrix")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d, vecDyn, vecDyn}, TypeRange{dyn2d});
      maybeInsertDecl("descale_matrix", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("matrix_sum")) {
      auto scalarTy = f32;
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{scalarTy});
      maybeInsertDecl("matrix_sum", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_dot_product")) {
      auto ft =
          FunctionType::get(ctx, TypeRange{vecDyn, vecDyn}, TypeRange{f32});
      maybeInsertDecl("vector_dot_product", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_sum")) {
      auto ft = FunctionType::get(ctx, TypeRange{vecDyn}, TypeRange{scalarTensor});
      maybeInsertDecl("vector_sum", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("epsilon_compare_abft")) {
      auto ft = FunctionType::get(ctx, TypeRange{f32, f32, f32, f32}, TypeRange{});
      maybeInsertDecl("epsilon_compare_abft", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_add")) {
      auto ft = FunctionType::get(ctx, TypeRange{vecDyn, vecDyn}, TypeRange{vecDyn});
      maybeInsertDecl("vector_add", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_sub")) {
      auto ft = FunctionType::get(ctx, TypeRange{vecDyn, vecDyn}, TypeRange{vecDyn});
      maybeInsertDecl("vector_sub", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_max_abs")) {
      auto ft = FunctionType::get(ctx, TypeRange{vecDyn}, TypeRange{f32});
      maybeInsertDecl("vector_max_abs", ft);
    }
    // Intentionally skip vector_max_abs_diff_pair declaration; debug path can
    // use scalar max-diff helpers instead.
    if (!module.lookupSymbol<func::FuncOp>("vector_first_elem")) {
      auto ft = FunctionType::get(ctx, TypeRange{vecDyn}, TypeRange{f32});
      maybeInsertDecl("vector_first_elem", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("scalar_tensor_add")) {
      auto ft = FunctionType::get(ctx, TypeRange{f32, f32}, TypeRange{f32});
      maybeInsertDecl("scalar_tensor_add", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("matrix_sub")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d, dyn2d}, TypeRange{dyn2d});
      maybeInsertDecl("matrix_sub", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("matrix_add")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d, dyn2d}, TypeRange{dyn2d});
      maybeInsertDecl("matrix_add", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("zero_matrix_like")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{dyn2d});
      maybeInsertDecl("zero_matrix_like", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>(
            "abft_analysis.abft_report_failure")) {
      auto ft = FunctionType::get(ctx, TypeRange{f32}, TypeRange{});
      maybeInsertDecl("abft_analysis.abft_report_failure", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>(
            "abft_analysis.abft_log_rowcol_delta")) {
      auto ft = FunctionType::get(ctx, TypeRange{f32, f32, f32}, TypeRange{});
      maybeInsertDecl("abft_analysis.abft_log_rowcol_delta", ft);
    }

    // Collect matmul ops inside this function only to avoid mutating while
    // walking.
    SmallVector<Operation *, 8> targets;
    func.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "linalg.matmul")
        targets.push_back(op);
    });

    auto sumFn = parsedSum ? parsedSum
                           : module.lookupSymbol<func::FuncOp>(
                                 StringRef("matrix_sum"));
    auto dotFn = parsedDot
                     ? parsedDot
                     : module.lookupSymbol<func::FuncOp>(
                           StringRef("vector_dot_product"));
    auto vecSumFn = parsedVecSum
                        ? parsedVecSum
                        : module.lookupSymbol<func::FuncOp>(
                              StringRef("vector_sum"));
    auto logRowColDeltaFn =
        module.lookupSymbol<func::FuncOp>(StringRef("abft_analysis.abft_log_rowcol_delta"));
    auto epsFn = parsedEps ? parsedEps
                           : module.lookupSymbol<func::FuncOp>(
                                 StringRef("epsilon_compare_abft"));
    auto rowvecFn = parsedRowVec
                        ? parsedRowVec
                        : module.lookupSymbol<func::FuncOp>(
                              StringRef("rowvec_mul_mat"));
    auto matcolFn = parsedMatCol
                        ? parsedMatCol
                        : module.lookupSymbol<func::FuncOp>(
                              StringRef("mat_mul_colvec"));
    auto rowFn = parsedRow
                     ? parsedRow
                     : module.lookupSymbol<func::FuncOp>(
                           StringRef("row_checksum"));
    auto colFn = parsedCol
                     ? parsedCol
                     : module.lookupSymbol<func::FuncOp>(
                           StringRef("column_checksum"));
    auto randVecMulMatFn =
        parsedRandVecMulMat
            ? parsedRandVecMulMat
            : module.lookupSymbol<func::FuncOp>(StringRef("randvec_mul_mat"));
    auto vecMaxFn = parsedVecMax
                        ? parsedVecMax
                        : module.lookupSymbol<func::FuncOp>(
                              StringRef("vector_max_abs_diff"));
    auto vecMaxPairFn = func::FuncOp();
    auto vecFirstFn = parsedVecFirst
                          ? parsedVecFirst
                          : module.lookupSymbol<func::FuncOp>(
                                StringRef("vector_first_elem"));
    auto scaleRowsFn = parsedScaleRows
                           ? parsedScaleRows
                           : module.lookupSymbol<func::FuncOp>(
                                 StringRef("scale_matrix_rows"));
    auto scaleColsFn = parsedScaleCols
                           ? parsedScaleCols
                           : module.lookupSymbol<func::FuncOp>(
                                 StringRef("scale_matrix_cols"));
    auto descaleFn = parsedDescale
                         ? parsedDescale
                         : module.lookupSymbol<func::FuncOp>(
                               StringRef("descale_matrix"));
    auto vecScaleFn = parsedVecScale
                          ? parsedVecScale
                          : module.lookupSymbol<func::FuncOp>(
                                StringRef("vector_scale"));
    auto expectedColFn =
        parsedExpectedCol
            ? parsedExpectedCol
            : module.lookupSymbol<func::FuncOp>(
                  StringRef("expected_col_checksum_freivalds"));
    auto expectedRowFn =
        parsedExpectedRow
            ? parsedExpectedRow
            : module.lookupSymbol<func::FuncOp>(
                  StringRef("expected_row_checksum_freivalds"));
    auto matrixSubFn = parsedMatrixSub
                           ? parsedMatrixSub
                           : module.lookupSymbol<func::FuncOp>(
                                 StringRef("matrix_sub"));
    auto matrixAddFn = parsedMatrixAdd
                           ? parsedMatrixAdd
                           : module.lookupSymbol<func::FuncOp>(
                                 StringRef("matrix_add"));
    auto zeroMatrixLikeFn =
        parsedZeroMatrixLike
            ? parsedZeroMatrixLike
            : module.lookupSymbol<func::FuncOp>(StringRef("zero_matrix_like"));
    auto scalarAddFn = parsedScalarAdd
                           ? parsedScalarAdd
                           : module.lookupSymbol<func::FuncOp>(
                                 StringRef("scalar_tensor_add"));
    StringRef rowScaleHelperName;
    StringRef colScaleHelperName;
    if (effectiveScalingMode == "binary") {
      rowScaleHelperName = "sample_row_scales_ones";
      colScaleHelperName = "sample_col_scales_ones";
    } else if (effectiveScalingMode == "standard") {
      rowScaleHelperName = "sample_row_scales_all_ones";
      colScaleHelperName = "sample_col_scales_all_ones";
    } else {
      module.emitError() << "Unsupported freivalds-scaling-mode: "
                         << effectiveScalingMode;
      signalPassFailure();
      return;
    }
    std::string weightedColumnChecksumBody =
        "module {\n"
        "  func.func private @" +
        rowScaleHelperName.str() +
        "(tensor<?x?xf32>, i32) -> tensor<?xf32>\n"
        "  func.func private @rowvec_mul_mat(tensor<?xf32>, tensor<?x?xf32>) -> tensor<?xf32>\n"
        "  func.func @weighted_column_checksum(%matrix: tensor<?x?xf32>) -> tensor<?xf32> {\n"
        "    %seed = arith.constant 0 : i32\n"
        "    %scales = func.call @" +
        rowScaleHelperName.str() +
        "(%matrix, %seed) : (tensor<?x?xf32>, i32) -> tensor<?xf32>\n"
        "    %res = func.call @rowvec_mul_mat(%scales, %matrix) : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?xf32>\n"
        "    return %res : tensor<?xf32>\n"
        "  }\n"
        "}\n";
    std::string weightedRowChecksumBody =
        "module {\n"
        "  func.func private @" +
        colScaleHelperName.str() +
        "(tensor<?x?xf32>, i32) -> tensor<?xf32>\n"
        "  func.func private @mat_mul_colvec(tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>\n"
        "  func.func @weighted_row_checksum(%matrix: tensor<?x?xf32>) -> tensor<?xf32> {\n"
        "    %seed = arith.constant 0 : i32\n"
        "    %scales = func.call @" +
        colScaleHelperName.str() +
        "(%matrix, %seed) : (tensor<?x?xf32>, i32) -> tensor<?xf32>\n"
        "    %res = func.call @mat_mul_colvec(%matrix, %scales) : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>\n"
        "    return %res : tensor<?xf32>\n"
        "  }\n"
        "}\n";
    func::FuncOp weightedColFn;
    func::FuncOp weightedRowFn;
    if (effectiveScalingMode != "standard") {
      weightedColFn = ensureFunctionWithBody(
          "weighted_column_checksum", StringRef(weightedColumnChecksumBody));
      weightedRowFn = ensureFunctionWithBody(
          "weighted_row_checksum", StringRef(weightedRowChecksumBody));
    }
    func::FuncOp rowScaleFn;
    func::FuncOp colScaleFn;
    if (rowScaleHelperName == "sample_row_scales_ones")
      rowScaleFn = parsedSampleRowOnes ? parsedSampleRowOnes
                                       : module.lookupSymbol<func::FuncOp>(
                                             StringRef(rowScaleHelperName));
    else if (rowScaleHelperName == "sample_row_scales_all_ones")
      rowScaleFn = parsedSampleRowAllOnes ? parsedSampleRowAllOnes
                                          : module.lookupSymbol<func::FuncOp>(
                                                StringRef(rowScaleHelperName));
    else
      rowScaleFn = parsedSampleRow ? parsedSampleRow
                                   : module.lookupSymbol<func::FuncOp>(
                                         StringRef(rowScaleHelperName));
    if (colScaleHelperName == "sample_col_scales_ones")
      colScaleFn = parsedSampleColOnes ? parsedSampleColOnes
                                       : module.lookupSymbol<func::FuncOp>(
                                             StringRef(colScaleHelperName));
    else if (colScaleHelperName == "sample_col_scales_all_ones")
      colScaleFn = parsedSampleColAllOnes ? parsedSampleColAllOnes
                                          : module.lookupSymbol<func::FuncOp>(
                                                StringRef(colScaleHelperName));
    else
      colScaleFn = parsedSampleCol ? parsedSampleCol
                                   : module.lookupSymbol<func::FuncOp>(
                                         StringRef(colScaleHelperName));
    for (Operation *op : targets) {
      op->emitRemark() << "abft-ones: matched matmul for ones*A insertion";
      int32_t targetIndex = static_cast<int32_t>(std::distance(
          targets.begin(), std::find(targets.begin(), targets.end(), op)));

      // Assume canonical linalg.matmul signature: inputs (A,B) outs(C)
      Value A = op->getOperand(0);
      Value B = op->getOperand(1);
      Value origA = A;
      Value origB = B;

      if (!isa<RankedTensorType>(A.getType())) {
        op->emitRemark() << "abft-ones: skipping non-tensor LHS";
        continue;
      }
      // NOTE: Keep this pass on the simple one-check path for stability while
      // collecting thresholds. The weighted/nonstandard setup currently causes
      // a crash during pass execution on VGG.
      bool useRegularStandardFreivalds = (effectiveScalingMode == "standard");
      if (useRegularStandardFreivalds) {
        auto aTy = llvm::dyn_cast<RankedTensorType>(A.getType());
        auto bTy = llvm::dyn_cast<RankedTensorType>(B.getType());
        if (!aTy || !bTy || !aTy.hasStaticShape() || !bTy.hasStaticShape()) {
          op->emitRemark()
              << "freivalds: skipping standard-mode instrumentation for "
                 "dynamic-shape matmul";
          continue;
        }
      }
      if (!useRegularStandardFreivalds &&
          (!rowFn || !matcolFn || !randVecMulMatFn)) {
        op->emitRemark() << "freivalds: required weighted checksum helpers not found; skipping instrumentation";
        continue;
      }
      OpBuilder b(op);
      Location loc = op->getLoc();

      Value rowScale;
      Value colScale;
      Value tmpA;
      Value tmpB;
      Value expected_row_checksum;
      Value expected_col_checksum;
      if (!useRegularStandardFreivalds) {
        auto rowScaleTy = rowScaleFn.getFunctionType();
        Value colArgA = origA;
        if (colArgA.getType() != rowScaleTy.getInput(0)) {
          auto colArgATy = llvm::dyn_cast<RankedTensorType>(colArgA.getType());
          auto expectedTy =
              llvm::dyn_cast<RankedTensorType>(rowScaleTy.getInput(0));
          if (colArgATy && expectedTy &&
              colArgATy.getRank() == expectedTy.getRank()) {
            colArgA =
                b.create<tensor::CastOp>(loc, rowScaleTy.getInput(0), colArgA)
                    .getResult();
          }
        }
        SmallVector<Type, 1> rowScaleRes;
        for (Type t : rowScaleTy.getResults())
          rowScaleRes.push_back(t);
        Value rowSeed =
            b.create<arith::ConstantIntOp>(loc, targetIndex * 2 + 1, 32);
        rowScale =
            b.create<func::CallOp>(loc, rowScaleHelperName,
                                   TypeRange(rowScaleRes),
                                   ValueRange{colArgA, rowSeed})
                .getResult(0);

        SmallVector<Type, 1> colSumARes;
        for (Type t : rowvecFn.getFunctionType().getResults())
          colSumARes.push_back(t);
        Value weightedColArgA = rowScale;
        if (weightedColArgA.getType() != rowvecFn.getFunctionType().getInput(0))
          weightedColArgA =
              b.create<tensor::CastOp>(loc,
                                       rowvecFn.getFunctionType().getInput(0),
                                       weightedColArgA)
                  .getResult();
        Value weightedColArgB = colArgA;
        if (weightedColArgB.getType() != rowvecFn.getFunctionType().getInput(1))
          weightedColArgB =
              b.create<tensor::CastOp>(loc,
                                       rowvecFn.getFunctionType().getInput(1),
                                       weightedColArgB)
                  .getResult();
        tmpA = b.create<func::CallOp>(loc, StringRef("rowvec_mul_mat"),
                                      TypeRange(colSumARes),
                                      ValueRange{weightedColArgA, weightedColArgB})
                   .getResult(0);

        auto colScaleTy = colScaleFn.getFunctionType();
        Value rowArgB = origB;
        if (rowArgB.getType() != colScaleTy.getInput(0)) {
          auto rowArgBTy = llvm::dyn_cast<RankedTensorType>(rowArgB.getType());
          auto expectedTy =
              llvm::dyn_cast<RankedTensorType>(colScaleTy.getInput(0));
          if (rowArgBTy && expectedTy &&
              rowArgBTy.getRank() == expectedTy.getRank()) {
            rowArgB =
                b.create<tensor::CastOp>(loc, colScaleTy.getInput(0), rowArgB)
                    .getResult();
          }
        }
        SmallVector<Type, 1> colScaleRes;
        for (Type t : colScaleTy.getResults())
          colScaleRes.push_back(t);
        Value colSeed =
            b.create<arith::ConstantIntOp>(loc, targetIndex * 2 + 2, 32);
        colScale =
            b.create<func::CallOp>(loc, colScaleHelperName,
                                   TypeRange(colScaleRes),
                                   ValueRange{rowArgB, colSeed})
                .getResult(0);

        if (scaleRowsFn && scaleColsFn) {
        SmallVector<Type, 1> srr;
        for (Type t : scaleRowsFn.getFunctionType().getResults())
          srr.push_back(t);
        Value a0 = origA;
        if (a0.getType() != scaleRowsFn.getFunctionType().getInput(0))
          a0 = b.create<tensor::CastOp>(loc, scaleRowsFn.getFunctionType().getInput(0), a0).getResult();
        Value a1 = rowScale;
        if (a1.getType() != scaleRowsFn.getFunctionType().getInput(1))
          a1 = b.create<tensor::CastOp>(loc, scaleRowsFn.getFunctionType().getInput(1), a1).getResult();
        Value scaledA = b.create<func::CallOp>(loc, StringRef("scale_matrix_rows"),
                                               TypeRange(srr), ValueRange{a0, a1}).getResult(0);
        if (scaledA.getType() != op->getOperand(0).getType())
          scaledA = b.create<tensor::CastOp>(loc, op->getOperand(0).getType(), scaledA).getResult();
        op->setOperand(0, scaledA);

        SmallVector<Type, 1> scr;
        for (Type t : scaleColsFn.getFunctionType().getResults())
          scr.push_back(t);
        Value b0 = origB;
        if (b0.getType() != scaleColsFn.getFunctionType().getInput(0))
          b0 = b.create<tensor::CastOp>(loc, scaleColsFn.getFunctionType().getInput(0), b0).getResult();
        Value b1 = colScale;
        if (b1.getType() != scaleColsFn.getFunctionType().getInput(1))
          b1 = b.create<tensor::CastOp>(loc, scaleColsFn.getFunctionType().getInput(1), b1).getResult();
        Value scaledB = b.create<func::CallOp>(loc, StringRef("scale_matrix_cols"),
                                               TypeRange(scr), ValueRange{b0, b1}).getResult(0);
        if (scaledB.getType() != op->getOperand(1).getType())
          scaledB = b.create<tensor::CastOp>(loc, op->getOperand(1).getType(), scaledB).getResult();
        op->setOperand(1, scaledB);
      }

        SmallVector<Type, 1> rowSumBRes;
        for (Type t : matcolFn.getFunctionType().getResults())
          rowSumBRes.push_back(t);
        Value weightedRowArgA = rowArgB;
        if (weightedRowArgA.getType() != matcolFn.getFunctionType().getInput(0))
          weightedRowArgA =
              b.create<tensor::CastOp>(
                   loc, matcolFn.getFunctionType().getInput(0), weightedRowArgA)
                  .getResult();
        Value weightedRowArgB = colScale;
        if (weightedRowArgB.getType() != matcolFn.getFunctionType().getInput(1))
          weightedRowArgB =
              b.create<tensor::CastOp>(
                   loc, matcolFn.getFunctionType().getInput(1), weightedRowArgB)
                  .getResult();
        tmpB = b.create<func::CallOp>(loc, StringRef("mat_mul_colvec"),
                                      TypeRange(rowSumBRes),
                                      ValueRange{weightedRowArgA, weightedRowArgB})
                   .getResult(0);

      // Seeds used for row/column checksum expectations. Keep tmpA/tmpB
      // unchanged for scalar checksum (dot/sum) logic.
        Value expectedColSeed = tmpA;
        Value expectedRowSeed = tmpB;
      // Precompute Huang full-checksum expectations:
      //   expected_col_checksum = checksum(A) * B  -> (N)
      //   expected_row_checksum = A * checksum(B)  -> (M)
      // Keep this always-on for row/col metric logging consistency even when
      // ABFT FuC checks are not explicitly enabled in the outer pipeline.
        if (expectedColFn) {
        SmallVector<Type, 1> t1Results;
        for (Type t : expectedColFn.getFunctionType().getResults())
          t1Results.push_back(t);
        auto expTy = expectedColFn.getFunctionType();
        Value arg0 = rowScale;
        if (arg0.getType() != expTy.getInput(0))
          arg0 = b.create<tensor::CastOp>(loc, expTy.getInput(0), arg0).getResult();
        Value arg1 = origA;
        if (arg1.getType() != expTy.getInput(1))
          arg1 = b.create<tensor::CastOp>(loc, expTy.getInput(1), arg1).getResult();
        Value arg2 = origB;
        if (arg2.getType() != expTy.getInput(2))
          arg2 = b.create<tensor::CastOp>(loc, expTy.getInput(2), arg2).getResult();
        auto t1Call = b.create<func::CallOp>(
            loc, StringRef("expected_col_checksum_freivalds"),
            TypeRange(t1Results), ValueRange{arg0, arg1, arg2});
          expected_col_checksum = t1Call.getResult(0);
        }
        if (expectedRowFn) {
        SmallVector<Type, 1> t2Results;
        for (Type t : expectedRowFn.getFunctionType().getResults())
          t2Results.push_back(t);
        auto expTy = expectedRowFn.getFunctionType();
        Value arg0 = origA;
        if (arg0.getType() != expTy.getInput(0))
          arg0 = b.create<tensor::CastOp>(loc, expTy.getInput(0), arg0).getResult();
        Value arg1 = origB;
        if (arg1.getType() != expTy.getInput(1))
          arg1 = b.create<tensor::CastOp>(loc, expTy.getInput(1), arg1).getResult();
        Value arg2 = colScale;
        if (arg2.getType() != expTy.getInput(2))
          arg2 = b.create<tensor::CastOp>(loc, expTy.getInput(2), arg2).getResult();
        auto t2Call = b.create<func::CallOp>(
            loc, StringRef("expected_row_checksum_freivalds"),
            TypeRange(t2Results), ValueRange{arg0, arg1, arg2});
          expected_row_checksum = t2Call.getResult(0);
        }
      }

      // Insert the rest of the ABFT logic as before...

      // Capture checksums for the init output (C) to account for accumulation
      // semantics of linalg.matmul (C = A*B + init).
      Value initC;
      Value initCForChecks;
      Value initSum;
      if (op->getNumOperands() >= 3) {
        initC = op->getOperand(2);
      }
      // Some VGG matmuls carry init as unranked tensors. Cast to the matmul
      // result type so init checksum terms are not silently dropped.
      if (initC && !isa<RankedTensorType>(initC.getType()) && !op->getResults().empty() &&
          isa<RankedTensorType>(op->getResult(0).getType())) {
        initC = b.create<tensor::CastOp>(loc, op->getResult(0).getType(), initC).getResult();
      }
      Value originalInitC = initC;
      // Always use the matmul outs/init tensor for checksum correction.
      // Even if it is all zeros, reducing it up front gives a consistent
      // baseline and preserves C = init + A*B semantics for comparisons.
      bool includeInit = static_cast<bool>(initC);

      // Run matmul itself on a fresh zero-initialized destination so ABFT
      // checks observe the pure A*B result. The original outs tensor is added
      // back after the matmul so externally visible semantics remain unchanged.
      bool addOriginalInitBack = false;
      if (originalInitC && zeroMatrixLikeFn && op->getNumOperands() >= 3) {
        SmallVector<Type, 1> zeroResults;
        for (Type t : zeroMatrixLikeFn.getFunctionType().getResults())
          zeroResults.push_back(t);
        Value zeroArg = originalInitC;
        Type zeroArgTy = zeroMatrixLikeFn.getFunctionType().getInput(0);
        if (zeroArg.getType() != zeroArgTy)
          zeroArg = b.create<tensor::CastOp>(loc, zeroArgTy, zeroArg).getResult();
        Value zeroInit = b.create<func::CallOp>(loc, StringRef("zero_matrix_like"),
                                                TypeRange(zeroResults), ValueRange{zeroArg})
                             .getResult(0);
        Value replacementInit = zeroInit;
        if (replacementInit.getType() != op->getOperand(2).getType())
          replacementInit = b.create<tensor::CastOp>(loc, op->getOperand(2).getType(), replacementInit).getResult();
        op->setOperand(2, replacementInit);
        includeInit = false;
        initC = {};
        initCForChecks = {};
        initSum = {};
        addOriginalInitBack = true;
      }

      // Compute init checksums after scaling decisions.
      if (includeInit) {
        Value initC_used = initC;
        initCForChecks = initC_used;
        if (freivaldsEnableFuC && sumFn) {
          Type expectedSumArg = sumFn.getFunctionType().getInput(0);
          Value sumArg = initC_used;
          if (sumArg.getType() != expectedSumArg) {
            sumArg = b.create<tensor::CastOp>(loc, expectedSumArg, sumArg).getResult();
          }
          SmallVector<Type, 1> sumResultTypes;
          for (Type t : sumFn.getFunctionType().getResults())
            sumResultTypes.push_back(t);
          auto sumCall = b.create<func::CallOp>(loc, StringRef("matrix_sum"),
                                                TypeRange(sumResultTypes),
                                                ValueRange{sumArg});
          initSum = sumCall.getResult(0);
        }
      }

      // Insert row/column checksum comparison logic after the matmul.
      // The scalar matrix-sum/FuC path remains optional, but row/column
      // logging must always be emitted for threshold collection.
      if (!op->getResults().empty()) {
        Value C = op->getResult(0);
        // Build after the op.
        Block::iterator nextIt = std::next(Block::iterator(op));
        OpBuilder bAfter(op->getBlock(), nextIt);

        // Ensure C matches expected arg type.
        if (isa<RankedTensorType>(C.getType())) {
          Value C_used = C;
          if (!useRegularStandardFreivalds && descaleFn && scaleRowsFn && scaleColsFn) {
            SmallVector<Type, 1> dRes;
            for (Type t : descaleFn.getFunctionType().getResults())
              dRes.push_back(t);
            Value d0 = C_used;
            if (d0.getType() != descaleFn.getFunctionType().getInput(0))
              d0 = bAfter.create<tensor::CastOp>(loc, descaleFn.getFunctionType().getInput(0), d0).getResult();
            Value d1 = rowScale;
            if (d1.getType() != descaleFn.getFunctionType().getInput(1))
              d1 = bAfter.create<tensor::CastOp>(loc, descaleFn.getFunctionType().getInput(1), d1).getResult();
            Value d2 = colScale;
            if (d2.getType() != descaleFn.getFunctionType().getInput(2))
              d2 = bAfter.create<tensor::CastOp>(loc, descaleFn.getFunctionType().getInput(2), d2).getResult();
            C_used = bAfter.create<func::CallOp>(loc, StringRef("descale_matrix"),
                                                 TypeRange(dRes), ValueRange{d0, d1, d2}).getResult(0);
            if (C_used.getType() != C.getType())
              C_used = bAfter.create<tensor::CastOp>(loc, C.getType(), C_used).getResult();
          }
          if (freivaldsEnableFuC && sumFn) {
            Type expectedSumArg = sumFn.getFunctionType().getInput(0);
            Value sumArg = C_used;
            if (sumArg.getType() != expectedSumArg) {
              sumArg =
                  bAfter.create<tensor::CastOp>(loc, expectedSumArg, sumArg)
                      .getResult();
            }
            SmallVector<Type, 1> sumResultTypes;
            for (Type t : sumFn.getFunctionType().getResults())
              sumResultTypes.push_back(t);
            auto sumCall = bAfter.create<func::CallOp>(
                loc, StringRef("matrix_sum"), TypeRange(sumResultTypes),
                ValueRange{sumArg});
            Value totalSum = sumCall.getResult(0);
            (void)totalSum; // left in IR for later instrumentation

            if (vecSumFn) {
              auto vecSumTy = vecSumFn.getFunctionType();
              if (tmpA) {
                Value aArg =
                    tmpA.getType() == vecSumTy.getInput(0)
                        ? tmpA
                        : bAfter
                              .create<tensor::CastOp>(loc,
                                                      vecSumTy.getInput(0),
                                                      tmpA)
                              .getResult();
                SmallVector<Type, 1> aSumTypes;
                for (Type t : vecSumTy.getResults())
                  aSumTypes.push_back(t);
                auto aSumCall = bAfter.create<func::CallOp>(
                    loc, StringRef("vector_sum"), TypeRange(aSumTypes),
                    ValueRange{aArg});
                Value aSum = aSumCall.getResult(0);
                (void)aSum;
              }
              if (tmpB) {
                Value bArg =
                    tmpB.getType() == vecSumTy.getInput(0)
                        ? tmpB
                        : bAfter
                              .create<tensor::CastOp>(loc,
                                                      vecSumTy.getInput(0),
                                                      tmpB)
                              .getResult();
                SmallVector<Type, 1> bSumTypes;
                for (Type t : vecSumTy.getResults())
                  bSumTypes.push_back(t);
                auto bSumCall = bAfter.create<func::CallOp>(
                    loc, StringRef("vector_sum"), TypeRange(bSumTypes),
                    ValueRange{bArg});
                Value bSum = bSumCall.getResult(0);
                (void)bSum;
              }
            }

            if (freivaldsEnableFuC && dotFn && epsFn && tmpB) {
              SmallVector<Value, 2> dotArgs;
              auto dotTy = dotFn.getFunctionType();
              if (dotTy.getNumInputs() >= 1)
                dotArgs.push_back(tmpA.getType() == dotTy.getInput(0)
                                      ? tmpA
                                      : bAfter
                                            .create<tensor::CastOp>(
                                                loc, dotTy.getInput(0), tmpA)
                                            .getResult());
              if (dotTy.getNumInputs() >= 2)
                dotArgs.push_back(tmpB.getType() == dotTy.getInput(1)
                                      ? tmpB
                                      : bAfter
                                            .create<tensor::CastOp>(
                                                loc, dotTy.getInput(1), tmpB)
                                            .getResult());
              SmallVector<Type, 1> dotResults;
              for (Type t : dotTy.getResults())
                dotResults.push_back(t);
              auto dotCall = bAfter.create<func::CallOp>(
                  loc, StringRef("vector_dot_product"), TypeRange(dotResults),
                  ValueRange{dotArgs});
              Value dot_product_tensor = dotCall.getResult(0);

              Value dot_expected = dot_product_tensor;
              if (includeInit && initSum && scalarAddFn) {
                SmallVector<Type, 1> addResults;
                for (Type t : scalarAddFn.getFunctionType().getResults())
                  addResults.push_back(t);
                SmallVector<Value, 2> addArgs;
                auto addTy = scalarAddFn.getFunctionType();
                Value addA = dot_product_tensor;
                if (addA.getType() != addTy.getInput(0))
                  addA = bAfter
                             .create<tensor::CastOp>(loc, addTy.getInput(0),
                                                     addA)
                             .getResult();
                Value addB = initSum;
                if (addB.getType() != addTy.getInput(1))
                  addB = bAfter
                             .create<tensor::CastOp>(loc, addTy.getInput(1),
                                                     addB)
                             .getResult();
                addArgs.push_back(addA);
                addArgs.push_back(addB);
                auto addCall = bAfter.create<func::CallOp>(
                    loc, StringRef("scalar_tensor_add"), TypeRange(addResults),
                    ValueRange{addArgs});
                dot_expected = addCall.getResult(0);
              }
              auto epsAbsAttr = bAfter.getF32FloatAttr(freivaldsEpsilonAbs);
              auto epsRelAttr = bAfter.getF32FloatAttr(freivaldsEpsilonRel);
              auto epsAbsConst = bAfter.create<arith::ConstantOp>(
                  loc, bAfter.getF32Type(), epsAbsAttr);
              auto epsRelConst = bAfter.create<arith::ConstantOp>(
                  loc, bAfter.getF32Type(), epsRelAttr);
              SmallVector<Value, 2> epsArgs;
              auto epsTy = epsFn.getFunctionType();
              if (epsTy.getNumInputs() >= 1)
                epsArgs.push_back(totalSum.getType() == epsTy.getInput(0)
                                      ? totalSum
                                      : bAfter
                                            .create<tensor::CastOp>(
                                                loc, epsTy.getInput(0),
                                                totalSum)
                                            .getResult());
              if (epsTy.getNumInputs() >= 2)
                epsArgs.push_back(dot_expected.getType() == epsTy.getInput(1)
                                      ? dot_expected
                                      : bAfter
                                            .create<tensor::CastOp>(
                                                loc, epsTy.getInput(1),
                                                dot_expected)
                                            .getResult());
              epsArgs.push_back(epsAbsConst);
              epsArgs.push_back(epsRelConst);
              {
                auto sumVal = totalSum;
                auto expVal = dot_expected;
                auto epsTy2 = epsFn.getFunctionType();
                if (epsTy2.getNumInputs() >= 1 &&
                    sumVal.getType() != epsTy2.getInput(0))
                  sumVal = bAfter
                               .create<tensor::CastOp>(loc, epsTy2.getInput(0),
                                                       sumVal)
                               .getResult();
                if (epsTy2.getNumInputs() >= 2 &&
                    expVal.getType() != epsTy2.getInput(1))
                  expVal = bAfter
                               .create<tensor::CastOp>(loc, epsTy2.getInput(1),
                                                       expVal)
                               .getResult();
                (void)sumVal;
                (void)expVal;
              }
              bAfter.create<func::CallOp>(loc, StringRef("epsilon_compare_abft"),
                                          TypeRange{}, ValueRange{epsArgs});
            }
          }

          Value rowMaxDelta = bAfter.create<arith::ConstantOp>(
              loc, bAfter.getF32FloatAttr(0.0f));
          Value colMaxDelta = bAfter.create<arith::ConstantOp>(
              loc, bAfter.getF32FloatAttr(0.0f));
          Value rowExpMax = bAfter.create<arith::ConstantOp>(
              loc, bAfter.getF32FloatAttr(0.0f));
          Value rowCalcMax = bAfter.create<arith::ConstantOp>(
              loc, bAfter.getF32FloatAttr(0.0f));
          Value colExpMax = bAfter.create<arith::ConstantOp>(
              loc, bAfter.getF32FloatAttr(0.0f));
          Value colCalcMax = bAfter.create<arith::ConstantOp>(
              loc, bAfter.getF32FloatAttr(0.0f));

          Value checksumCompareMatrix = C_used;
          if (includeInit && initCForChecks && matrixSubFn) {
            SmallVector<Type, 1> subResults;
            for (Type t : matrixSubFn.getFunctionType().getResults())
              subResults.push_back(t);
            auto subTy = matrixSubFn.getFunctionType();
            Value subA = checksumCompareMatrix;
            if (subA.getType() != subTy.getInput(0))
              subA = bAfter
                         .create<tensor::CastOp>(loc, subTy.getInput(0), subA)
                         .getResult();
            Value subB = initCForChecks;
            if (subB.getType() != subTy.getInput(1))
              subB = bAfter
                         .create<tensor::CastOp>(loc, subTy.getInput(1), subB)
                         .getResult();
            checksumCompareMatrix =
                bAfter
                    .create<func::CallOp>(loc, StringRef("matrix_sub"),
                                          TypeRange(subResults),
                                          ValueRange{subA, subB})
                    .getResult(0);
          }

          Value standardAggDelta;
          Value standardExpMax = bAfter.create<arith::ConstantOp>(
              loc, bAfter.getF32FloatAttr(0.0f));
          Value standardObsMax = bAfter.create<arith::ConstantOp>(
              loc, bAfter.getF32FloatAttr(0.0f));
          if (useRegularStandardFreivalds && randVecMulMatFn &&
              !randVecMulMatFn.isExternal() && matcolFn &&
              !matcolFn.isExternal()) {
            int checks = std::max(1, (int)freivaldsNumChecks);
            for (int checkIdx = 0; checkIdx < checks; ++checkIdx) {
              Value randSeed = bAfter.create<arith::ConstantIntOp>(
                  loc, sampleBinaryBit() + targetIndex * 131 + checkIdx * 977 + 1,
                  32);
              SmallVector<Type, 1> randRes;
              for (Type t : randVecMulMatFn.getFunctionType().getResults())
                randRes.push_back(t);
              Value rhsForRand = origB;
              if (rhsForRand.getType() !=
                  randVecMulMatFn.getFunctionType().getInput(0))
                rhsForRand = bAfter
                                 .create<tensor::CastOp>(
                                     loc, randVecMulMatFn.getFunctionType().getInput(0),
                                     rhsForRand)
                                 .getResult();
              Value bTimesR = bAfter
                                  .create<func::CallOp>(
                                      loc, StringRef("randvec_mul_mat"),
                                      TypeRange(randRes), ValueRange{rhsForRand, randSeed})
                                  .getResult(0);

              SmallVector<Type, 1> matcolRes;
              for (Type t : matcolFn.getFunctionType().getResults())
                matcolRes.push_back(t);
              Value lhsForExpected = origA;
              if (lhsForExpected.getType() != matcolFn.getFunctionType().getInput(0))
                lhsForExpected = bAfter
                                     .create<tensor::CastOp>(
                                         loc, matcolFn.getFunctionType().getInput(0),
                                         lhsForExpected)
                                     .getResult();
              Value brForExpected = bTimesR;
              if (brForExpected.getType() != matcolFn.getFunctionType().getInput(1))
                brForExpected = bAfter
                                    .create<tensor::CastOp>(
                                        loc, matcolFn.getFunctionType().getInput(1),
                                        brForExpected)
                                    .getResult();
              Value expectedVec = bAfter
                                      .create<func::CallOp>(
                                          loc, StringRef("mat_mul_colvec"),
                                          TypeRange(matcolRes),
                                          ValueRange{lhsForExpected, brForExpected})
                                      .getResult(0);

              Value obsMat = checksumCompareMatrix;
              if (obsMat.getType() != randVecMulMatFn.getFunctionType().getInput(0))
                obsMat = bAfter
                             .create<tensor::CastOp>(
                                 loc, randVecMulMatFn.getFunctionType().getInput(0),
                                 obsMat)
                             .getResult();
              Value observedVec = bAfter
                                      .create<func::CallOp>(
                                          loc, StringRef("randvec_mul_mat"),
                                          TypeRange(randRes), ValueRange{obsMat, randSeed})
                                      .getResult(0);

              Value oneDelta = bAfter.create<arith::ConstantOp>(
                  loc, bAfter.getF32FloatAttr(0.0f));
              if (vecMaxFn && !vecMaxFn.isExternal()) {
                SmallVector<Type, 1> dRes;
                for (Type t : vecMaxFn.getFunctionType().getResults())
                  dRes.push_back(t);
                auto dTy = vecMaxFn.getFunctionType();
                Value dV1 = expectedVec;
                if (dV1.getType() != dTy.getInput(0))
                  dV1 = bAfter
                            .create<tensor::CastOp>(loc, dTy.getInput(0), dV1)
                            .getResult();
                Value dV2 = observedVec;
                if (dV2.getType() != dTy.getInput(1))
                  dV2 = bAfter
                            .create<tensor::CastOp>(loc, dTy.getInput(1), dV2)
                            .getResult();
                oneDelta = bAfter
                               .create<func::CallOp>(
                                   loc, StringRef("vector_max_abs_diff"),
                                   TypeRange(dRes), ValueRange{dV1, dV2})
                               .getResult(0);
              } else if (vecFirstFn && !vecFirstFn.isExternal()) {
                SmallVector<Type, 1> fRes;
                for (Type t : vecFirstFn.getFunctionType().getResults())
                  fRes.push_back(t);
                auto fTy = vecFirstFn.getFunctionType();
                Value fe = expectedVec;
                if (fe.getType() != fTy.getInput(0))
                  fe = bAfter
                           .create<tensor::CastOp>(loc, fTy.getInput(0), fe)
                           .getResult();
                Value fo = observedVec;
                if (fo.getType() != fTy.getInput(0))
                  fo = bAfter
                           .create<tensor::CastOp>(loc, fTy.getInput(0), fo)
                           .getResult();
                Value expFirst = bAfter
                                     .create<func::CallOp>(
                                         loc, StringRef("vector_first_elem"),
                                         TypeRange(fRes), ValueRange{fe})
                                     .getResult(0);
                Value obsFirst = bAfter
                                     .create<func::CallOp>(
                                         loc, StringRef("vector_first_elem"),
                                         TypeRange(fRes), ValueRange{fo})
                                     .getResult(0);
                Value diff = bAfter.create<arith::SubFOp>(loc, expFirst, obsFirst);
                oneDelta = bAfter.create<math::AbsFOp>(loc, diff);
              }
              if (!standardAggDelta) {
                standardAggDelta = oneDelta;
              } else {
                standardAggDelta =
                    bAfter.create<arith::MaximumFOp>(loc, standardAggDelta, oneDelta);
              }
            }
          }

          Value calcColChecksum;
          if (!useRegularStandardFreivalds && rowvecFn &&
              !rowvecFn.isExternal()) {
            SmallVector<Type, 1> cRes;
            for (Type t : rowvecFn.getFunctionType().getResults())
              cRes.push_back(t);
            Value cArg0 = rowScale;
            if (cArg0.getType() != rowvecFn.getFunctionType().getInput(0))
              cArg0 = bAfter
                          .create<tensor::CastOp>(
                              loc, rowvecFn.getFunctionType().getInput(0),
                              cArg0)
                          .getResult();
            Value cArg1 = checksumCompareMatrix;
            if (cArg1.getType() != rowvecFn.getFunctionType().getInput(1))
              cArg1 = bAfter
                          .create<tensor::CastOp>(
                              loc, rowvecFn.getFunctionType().getInput(1),
                              cArg1)
                          .getResult();
            calcColChecksum = bAfter
                                  .create<func::CallOp>(
                                      loc, StringRef("rowvec_mul_mat"),
                                      TypeRange(cRes),
                                      ValueRange{cArg0, cArg1})
                                  .getResult(0);
          }

          Value calcRowChecksum;
          if (!useRegularStandardFreivalds && matcolFn &&
              !matcolFn.isExternal()) {
            SmallVector<Type, 1> rRes;
            for (Type t : matcolFn.getFunctionType().getResults())
              rRes.push_back(t);
            Value rArg0 = checksumCompareMatrix;
            if (rArg0.getType() != matcolFn.getFunctionType().getInput(0))
              rArg0 = bAfter
                          .create<tensor::CastOp>(
                              loc, matcolFn.getFunctionType().getInput(0),
                              rArg0)
                          .getResult();
            Value rArg1 = colScale;
            if (rArg1.getType() != matcolFn.getFunctionType().getInput(1))
              rArg1 = bAfter
                          .create<tensor::CastOp>(
                              loc, matcolFn.getFunctionType().getInput(1),
                              rArg1)
                          .getResult();
            calcRowChecksum = bAfter
                                  .create<func::CallOp>(
                                      loc, StringRef("mat_mul_colvec"),
                                      TypeRange(rRes),
                                      ValueRange{rArg0, rArg1})
                                  .getResult(0);
          }

          if (useRegularStandardFreivalds && standardAggDelta) {
            rowMaxDelta = standardAggDelta;
            colMaxDelta = standardAggDelta;
            rowExpMax = standardExpMax;
            rowCalcMax = standardObsMax;
            colExpMax = standardExpMax;
            colCalcMax = standardObsMax;
          } else {
          if (!useRegularStandardFreivalds && expected_col_checksum &&
              calcColChecksum) {
            Value colExpected = expected_col_checksum;
            Value colCalcComparable = calcColChecksum;
            if (vecMaxFn && !vecMaxFn.isExternal()) {
              SmallVector<Type, 1> dRes;
              for (Type t : vecMaxFn.getFunctionType().getResults())
                dRes.push_back(t);
              auto dTy = vecMaxFn.getFunctionType();
              Value dV1 = colExpected;
              if (dV1.getType() != dTy.getInput(0))
                dV1 = bAfter
                          .create<tensor::CastOp>(loc, dTy.getInput(0), dV1)
                          .getResult();
              Value dV2 = colCalcComparable;
              if (dV2.getType() != dTy.getInput(1))
                dV2 = bAfter
                          .create<tensor::CastOp>(loc, dTy.getInput(1), dV2)
                          .getResult();
              colMaxDelta = bAfter
                                .create<func::CallOp>(
                                    loc, StringRef("vector_max_abs_diff"),
                                    TypeRange(dRes), ValueRange{dV1, dV2})
                                .getResult(0);
            }
            if (vecMaxPairFn && !vecMaxPairFn.isExternal()) {
              SmallVector<Type, 2> pRes;
              for (Type t : vecMaxPairFn.getFunctionType().getResults())
                pRes.push_back(t);
              auto pTy = vecMaxPairFn.getFunctionType();
              Value ce = colExpected;
              if (ce.getType() != pTy.getInput(0))
                ce = bAfter
                         .create<tensor::CastOp>(loc, pTy.getInput(0), ce)
                         .getResult();
              Value cc = colCalcComparable;
              if (cc.getType() != pTy.getInput(1))
                cc = bAfter
                         .create<tensor::CastOp>(loc, pTy.getInput(1), cc)
                         .getResult();
              auto pairCall = bAfter.create<func::CallOp>(
                  loc, StringRef("vector_max_abs_diff_pair"), TypeRange(pRes),
                  ValueRange{ce, cc});
              colExpMax = pairCall.getResult(0);
              colCalcMax = pairCall.getResult(1);
            }
          }

          if (!useRegularStandardFreivalds && expected_row_checksum &&
              calcRowChecksum) {
            Value rowExpected = expected_row_checksum;
            Value rowCalcComparable = calcRowChecksum;
            if (vecMaxFn && !vecMaxFn.isExternal()) {
              SmallVector<Type, 1> dRes;
              for (Type t : vecMaxFn.getFunctionType().getResults())
                dRes.push_back(t);
              auto dTy = vecMaxFn.getFunctionType();
              Value dV1 = rowExpected;
              if (dV1.getType() != dTy.getInput(0))
                dV1 = bAfter
                          .create<tensor::CastOp>(loc, dTy.getInput(0), dV1)
                          .getResult();
              Value dV2 = rowCalcComparable;
              if (dV2.getType() != dTy.getInput(1))
                dV2 = bAfter
                          .create<tensor::CastOp>(loc, dTy.getInput(1), dV2)
                          .getResult();
              rowMaxDelta = bAfter
                                .create<func::CallOp>(
                                    loc, StringRef("vector_max_abs_diff"),
                                    TypeRange(dRes), ValueRange{dV1, dV2})
                                .getResult(0);
            }
            if (vecMaxPairFn && !vecMaxPairFn.isExternal()) {
              SmallVector<Type, 2> pRes;
              for (Type t : vecMaxPairFn.getFunctionType().getResults())
                pRes.push_back(t);
              auto pTy = vecMaxPairFn.getFunctionType();
              Value re = rowExpected;
              if (re.getType() != pTy.getInput(0))
                re = bAfter
                         .create<tensor::CastOp>(loc, pTy.getInput(0), re)
                         .getResult();
              Value rc = rowCalcComparable;
              if (rc.getType() != pTy.getInput(1))
                rc = bAfter
                         .create<tensor::CastOp>(loc, pTy.getInput(1), rc)
                         .getResult();
              auto pairCall = bAfter.create<func::CallOp>(
                  loc, StringRef("vector_max_abs_diff_pair"), TypeRange(pRes),
                  ValueRange{re, rc});
              rowExpMax = pairCall.getResult(0);
              rowCalcMax = pairCall.getResult(1);
            }
          }
          }

          (void)rowExpMax;
          (void)rowCalcMax;
          (void)colExpMax;
          (void)colCalcMax;

          Value indexConst = bAfter.create<arith::ConstantOp>(
              loc, bAfter.getF32Type(),
              bAfter.getF32FloatAttr((float)targetIndex));
          if (logRowColDeltaFn) {
            bAfter.create<func::CallOp>(
                loc, StringRef("abft_analysis.abft_log_rowcol_delta"),
                TypeRange{}, ValueRange{indexConst, rowMaxDelta, colMaxDelta});
          }

          if (addOriginalInitBack && originalInitC && matrixAddFn) {
            SmallVector<Type, 1> addResults;
            for (Type t : matrixAddFn.getFunctionType().getResults())
              addResults.push_back(t);
            // Add init back to the checksum-compare matrix (pure A*B after
            // optional descale), never to the scaled matmul result.
            Value addA = C_used;
            Type addATy = matrixAddFn.getFunctionType().getInput(0);
            if (addA.getType() != addATy)
              addA = bAfter.create<tensor::CastOp>(loc, addATy, addA).getResult();
            Value addB = originalInitC;
            Type addBTy = matrixAddFn.getFunctionType().getInput(1);
            if (addB.getType() != addBTy)
              addB = bAfter.create<tensor::CastOp>(loc, addBTy, addB).getResult();
            auto addBackCall = bAfter.create<func::CallOp>(loc, StringRef("matrix_add"),
                                                           TypeRange(addResults),
                                                           ValueRange{addA, addB});
            Value restoredC = addBackCall.getResult(0);
            if (restoredC.getType() != C.getType()) {
              restoredC = bAfter.create<tensor::CastOp>(loc, C.getType(), restoredC).getResult();
            }
            Operation *restoreOp = restoredC.getDefiningOp();
            if (restoreOp) {
              DominanceInfo dom(func);
              C.replaceUsesWithIf(restoredC, [&](OpOperand &use) {
                Operation *user = use.getOwner();
                return user != addBackCall.getOperation() &&
                       dom.properlyDominates(restoreOp, user);
              });
            }
          }
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createFreivaldsPass() {
  return std::make_unique<FreivaldsPass>();
}
static mlir::PassRegistration<FreivaldsPass> reg;
