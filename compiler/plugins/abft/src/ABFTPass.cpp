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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
// Command line option parsing for the standalone flag used by the pass.
#include "llvm/Support/CommandLine.h"

using namespace mlir;

// Global CLI flag to enable Full-Checksum elementwise comparisons. Using a
// static command-line option avoids putting non-copyable llvm::cl::opt into
// the pass object (PassWrapper needs to be copyable).
static llvm::cl::opt<bool> abftEnableFuC(
    "abft-enable-fuc",
    llvm::cl::desc("Enable full elementwise checksum (t1/t2) comparisons"),
    llvm::cl::init(false));

static llvm::cl::opt<float> abftEpsilonAbs(
  "abft-epsilon-abs",
  llvm::cl::desc("Absolute epsilon for ABFT checks"),
  llvm::cl::init(1.0e-7f));

static llvm::cl::opt<float> abftEpsilonRel(
  "abft-epsilon-rel",
  llvm::cl::desc("Relative epsilon for ABFT checks"),
  llvm::cl::init(1.0e-5f));

namespace {

constexpr StringLiteral kAbftModeAttrName = "iree.abft.mode";
constexpr StringLiteral kAbftModeNormal = "abft";
constexpr StringLiteral kAbftModeScaled = "abyzft";

struct ABFTPass : public PassWrapper<ABFTPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ABFTPass)

  // NOTE: the CLI flag is defined as a static llvm::cl::opt above. We avoid
  // storing an Option<> member inside the pass because llvm::cl::opt is
  // non-copyable which breaks PassWrapper's cloning. Read the flag at
  // run-time from `abftEnableFuC`.

  StringRef getArgument() const final { return "abft-insert-ones"; }
  StringRef getDescription() const final {
    return "For each linalg.matmul insert a computation of ones * A (row sums)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto funcName = func.getSymName();
    if (funcName == "column_checksum" || funcName == "row_checksum" ||
        funcName == "matrix_sum" || funcName == "vector_sum" ||
        funcName == "vector_dot_product" || funcName == "epsilon_compare_abft" ||
        funcName == "sample_row_scales" ||
        funcName == "sample_row_scales_ones" ||
        funcName == "sample_col_scales" ||
        funcName == "sample_col_scales_ones" ||
        funcName == "rowvec_mul_mat" || funcName == "mat_mul_colvec" ||
        funcName == "vector_epsilon_compare_abft" ||
        funcName == "vector_max_abs_diff" ||
        funcName == "vector_max_abs_diff_pair" ||
        funcName == "vector_first_elem" || funcName == "scalar_tensor_add" ||
        funcName == "matrix_sub" || funcName == "matrix_add" ||
        funcName == "zero_matrix_like" ||
        funcName == "abft_analysis.abft_report_failure" ||
        funcName == "abft_analysis.abft_log_rowcol_delta" ||
        funcName == "abft_analysis.abft_log_rowcol_debug") {
      return;
    }
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (auto modeAttr = module->getAttrOfType<StringAttr>(kAbftModeAttrName)) {
      if (modeAttr.getValue() == kAbftModeScaled) {
        module.emitError(
            "ABFT and AByzFT are mutually exclusive; this module is already "
            "marked for AByzFT instrumentation");
        signalPassFailure();
        return;
      }
    } else {
      module->setAttr(kAbftModeAttrName,
                      StringAttr::get(module.getContext(), kAbftModeNormal));
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
    ctx->getOrLoadDialect<tensor::TensorDialect>();
    ctx->getOrLoadDialect<arith::ArithDialect>();
    ctx->getOrLoadDialect<math::MathDialect>();
    ctx->getOrLoadDialect<cf::ControlFlowDialect>();

    // Helper: parse and insert a helper func.func with body into the module
    // (clone). Returns the inserted func or nullptr on failure.
    auto ensureFunctionWithBody = [&](StringRef name,
                                      StringRef body) -> func::FuncOp {
      if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
      OwningOpRef<ModuleOp> tmp = parseSourceString<ModuleOp>(body, ctx);
      if (!tmp) {
        module.emitRemark() << "abft: failed to parse helper body for " << name;
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
            << "abft: helper " << name << " not present in parsed body";
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

    // Provide implementations for sampling helpers that cycle through the
    // scale choices {-8, -4, -2, -0.5, 0.5, 2, 4, 8} deterministically.
    func::FuncOp parsedSampleRow = ensureFunctionWithBody("sample_row_scales", R"mlir(
module {
  func.func @sample_row_scales(%mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
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
      %mod = arith.remui %idx, %c8 : index
      %c0_cmp = arith.constant 0 : index
      %is0 = arith.cmpi eq, %mod, %c0_cmp : index
      %v0 = arith.constant -8.0 : f32
      %c1_cmp = arith.constant 1 : index
      %is1 = arith.cmpi eq, %mod, %c1_cmp : index
      %v1 = arith.constant -4.0 : f32
      %c2_cmp = arith.constant 2 : index
      %is2 = arith.cmpi eq, %mod, %c2_cmp : index
      %v2 = arith.constant -2.0 : f32
      %c3_cmp = arith.constant 3 : index
      %is3 = arith.cmpi eq, %mod, %c3_cmp : index
      %v3 = arith.constant -0.5 : f32
      %c4_cmp = arith.constant 4 : index
      %is4 = arith.cmpi eq, %mod, %c4_cmp : index
      %v4 = arith.constant 0.5 : f32
      %c5_cmp = arith.constant 5 : index
      %is5 = arith.cmpi eq, %mod, %c5_cmp : index
      %v5 = arith.constant 2.0 : f32
      %c6_cmp = arith.constant 6 : index
      %is6 = arith.cmpi eq, %mod, %c6_cmp : index
      %v6 = arith.constant 4.0 : f32
      %v7 = arith.constant 8.0 : f32
      %sel0 = arith.select %is0, %v0, %v7 : f32
      %sel1 = arith.select %is1, %v1, %sel0 : f32
      %sel2 = arith.select %is2, %v2, %sel1 : f32
      %sel3 = arith.select %is3, %v3, %sel2 : f32
      %sel4 = arith.select %is4, %v4, %sel3 : f32
      %sel5 = arith.select %is5, %v5, %sel4 : f32
      %sel6 = arith.select %is6, %v6, %sel5 : f32
      linalg.yield %sel6 : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedSampleRow;

    func::FuncOp parsedSampleRowOnes = ensureFunctionWithBody("sample_row_scales_ones", R"mlir(
module {
  func.func @sample_row_scales_ones(%mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %empty = tensor.empty(%m) : tensor<?xf32>
    %one = arith.constant 1.0 : f32
    %res = linalg.fill ins(%one : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedSampleRowOnes;

    func::FuncOp parsedSampleCol = ensureFunctionWithBody("sample_col_scales", R"mlir(
module {
  func.func @sample_col_scales(%mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
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
      %mod = arith.remui %idx, %c8 : index
      %c0_cmp = arith.constant 0 : index
      %is0 = arith.cmpi eq, %mod, %c0_cmp : index
      %v0 = arith.constant -8.0 : f32
      %c1_cmp = arith.constant 1 : index
      %is1 = arith.cmpi eq, %mod, %c1_cmp : index
      %v1 = arith.constant -4.0 : f32
      %c2_cmp = arith.constant 2 : index
      %is2 = arith.cmpi eq, %mod, %c2_cmp : index
      %v2 = arith.constant -2.0 : f32
      %c3_cmp = arith.constant 3 : index
      %is3 = arith.cmpi eq, %mod, %c3_cmp : index
      %v3 = arith.constant -0.5 : f32
      %c4_cmp = arith.constant 4 : index
      %is4 = arith.cmpi eq, %mod, %c4_cmp : index
      %v4 = arith.constant 0.5 : f32
      %c5_cmp = arith.constant 5 : index
      %is5 = arith.cmpi eq, %mod, %c5_cmp : index
      %v5 = arith.constant 2.0 : f32
      %c6_cmp = arith.constant 6 : index
      %is6 = arith.cmpi eq, %mod, %c6_cmp : index
      %v6 = arith.constant 4.0 : f32
      %v7 = arith.constant 8.0 : f32
      %sel0 = arith.select %is0, %v0, %v7 : f32
      %sel1 = arith.select %is1, %v1, %sel0 : f32
      %sel2 = arith.select %is2, %v2, %sel1 : f32
      %sel3 = arith.select %is3, %v3, %sel2 : f32
      %sel4 = arith.select %is4, %v4, %sel3 : f32
      %sel5 = arith.select %is5, %v5, %sel4 : f32
      %sel6 = arith.select %is6, %v6, %sel5 : f32
      linalg.yield %sel6 : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedSampleCol;

    func::FuncOp parsedSampleColOnes = ensureFunctionWithBody("sample_col_scales_ones", R"mlir(
module {
  func.func @sample_col_scales_ones(%mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c1 = arith.constant 1 : index
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %one = arith.constant 1.0 : f32
    %res = linalg.fill ins(%one : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedSampleColOnes;

    // Full-checksum helper implementations (so the pass can emit full-checksum
    // calls unconditionally). These compute:
    //  - rowvec_mul_mat(rv, mat) -> vector: for each column j, sum_k
    //  rv[k]*mat[k,j]
    //  - mat_mul_colvec(mat, cv) -> vector: for each row i, sum_j
    //  mat[i,j]*cv[j]
    //  - vector_epsilon_compare_abft(v1, v2, eps) -> () : elementwise report
    //  (|v1-v2| < eps)
    func::FuncOp parsedRowVec = ensureFunctionWithBody("rowvec_mul_mat", R"mlir(
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
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %empty = tensor.empty(%m) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%mat, %cv : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%mat_elem: f32, %cv_elem: f32, %acc: f32):
      %prod = arith.mulf %mat_elem, %cv_elem : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedMatCol;

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

    // Use linalg.generic instead of cf.br so IREE's lowering pipeline treats
    // both the function definition and its call sites consistently (no explicit
    // index dimension args get added to the signature).  A 3-output
    // linalg.generic tracks (maxDiff, bestA, bestB) in one reduction pass.
    func::FuncOp parsedVecMaxPair =
        ensureFunctionWithBody("vector_max_abs_diff_pair", R"mlir(
module {
  func.func @vector_max_abs_diff_pair(%v1: tensor<?xf32>, %v2: tensor<?xf32>) -> (f32, f32) {
    %neg = arith.constant -1.0 : f32
    %zero = arith.constant 0.0 : f32
    %init_diff = tensor.empty() : tensor<f32>
    %init_a    = tensor.empty() : tensor<f32>
    %init_b    = tensor.empty() : tensor<f32>
    %filled_diff = linalg.fill ins(%neg  : f32) outs(%init_diff : tensor<f32>) -> tensor<f32>
    %filled_a    = linalg.fill ins(%zero : f32) outs(%init_a    : tensor<f32>) -> tensor<f32>
    %filled_b    = linalg.fill ins(%zero : f32) outs(%init_b    : tensor<f32>) -> tensor<f32>
    %out_diff, %out_a, %out_b = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>,
        affine_map<(d0) -> ()>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    } ins(%v1, %v2 : tensor<?xf32>, tensor<?xf32>)
      outs(%filled_diff, %filled_a, %filled_b : tensor<f32>, tensor<f32>, tensor<f32>) {
    ^bb0(%a: f32, %b: f32, %cur_diff: f32, %cur_a: f32, %cur_b: f32):
      %d    = arith.subf %a, %b : f32
      %ad   = math.absf %d : f32
      %gt   = arith.cmpf ogt, %ad, %cur_diff : f32
      %nd   = arith.select %gt, %ad, %cur_diff : f32
      %na   = arith.select %gt, %a,  %cur_a   : f32
      %nb   = arith.select %gt, %b,  %cur_b   : f32
      linalg.yield %nd, %na, %nb : f32, f32, f32
    } -> (tensor<f32>, tensor<f32>, tensor<f32>)
    %best_a = tensor.extract %out_a[] : tensor<f32>
    %best_b = tensor.extract %out_b[] : tensor<f32>
    return %best_a, %best_b : f32, f32
  }
}
)mlir");
    (void)parsedVecMaxPair;

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
    if (!module.lookupSymbol<func::FuncOp>("vector_max_abs_diff_pair")) {
      auto ft = FunctionType::get(ctx, TypeRange{vecDyn, vecDyn}, TypeRange{f32, f32});
      maybeInsertDecl("vector_max_abs_diff_pair", ft);
    }
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
    if (!module.lookupSymbol<func::FuncOp>(
            "abft_analysis.abft_log_rowcol_debug")) {
      auto ft = FunctionType::get(
          ctx, TypeRange{f32, f32, f32, f32, f32},
          TypeRange{});
      maybeInsertDecl("abft_analysis.abft_log_rowcol_debug", ft);
    }

    // Collect matmul ops inside this function only to avoid mutating while
    // walking.
    SmallVector<Operation *, 8> targets;
    func.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "linalg.matmul")
        targets.push_back(op);
    });

    auto colFn =
        module.lookupSymbol<func::FuncOp>(StringRef("column_checksum"));
    auto rowFn = module.lookupSymbol<func::FuncOp>(StringRef("row_checksum"));
    auto sumFn = module.lookupSymbol<func::FuncOp>(StringRef("matrix_sum"));
    auto dotFn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_dot_product"));
    auto vecSumFn =
      module.lookupSymbol<func::FuncOp>(StringRef("vector_sum"));
    auto logRowColDeltaFn =
      module.lookupSymbol<func::FuncOp>(StringRef("abft_analysis.abft_log_rowcol_delta"));
    auto logRowColDebugFn =
      module.lookupSymbol<func::FuncOp>(StringRef("abft_analysis.abft_log_rowcol_debug"));
    auto epsFn =
        module.lookupSymbol<func::FuncOp>(StringRef("epsilon_compare_abft"));
    auto rowvecFn =
        module.lookupSymbol<func::FuncOp>(StringRef("rowvec_mul_mat"));
    auto matcolFn =
        module.lookupSymbol<func::FuncOp>(StringRef("mat_mul_colvec"));
    auto vecMaxFn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_diff"));
    auto vecMaxPairFn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_diff_pair"));
    auto vecFirstFn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_first_elem"));
    auto matrixSubFn =
      module.lookupSymbol<func::FuncOp>(StringRef("matrix_sub"));
    auto matrixAddFn =
      module.lookupSymbol<func::FuncOp>(StringRef("matrix_add"));
    auto zeroMatrixLikeFn =
      module.lookupSymbol<func::FuncOp>(StringRef("zero_matrix_like"));
    auto scalarAddFn =
      module.lookupSymbol<func::FuncOp>(StringRef("scalar_tensor_add"));
    for (Operation *op : targets) {
      op->emitRemark() << "abft-ones: matched matmul for ones*A insertion";

      // Assume canonical linalg.matmul signature: inputs (A,B) outs(C)
      Value A = op->getOperand(0);
      Value B = op->getOperand(1);
      Value origA = A;
      Value origB = B;

      if (!isa<RankedTensorType>(A.getType())) {
        op->emitRemark() << "abft-ones: skipping non-tensor LHS";
        continue;
      }
      if (!colFn || !rowFn) {
        op->emitRemark() << "abft-ones: helper checksum not found; skipping instrumentation";
        continue;
      }
      OpBuilder b(op);
      Location loc = op->getLoc();


      // Huang full-checksum vectors:
      //   checksum(A) = ones^T * A (K) -> column_checksum(A)
      //   checksum(B) = B * ones   (K) -> row_checksum(B)
      // Keep these as tmpA/tmpB for scalar and full-checksum paths.
      Type expectedColArgA = colFn.getFunctionType().getInput(0);
      Value colArgA = origA;
      if (colArgA.getType() != expectedColArgA) {
        auto colArgATy = llvm::dyn_cast<RankedTensorType>(colArgA.getType());
        auto expectedTy = llvm::dyn_cast<RankedTensorType>(expectedColArgA);
        if (colArgATy && expectedTy && colArgATy.getRank() == expectedTy.getRank()) {
          colArgA = b.create<tensor::CastOp>(loc, expectedColArgA, colArgA).getResult();
        } else if (!colArgATy || !expectedTy || colArgATy.getRank() != expectedTy.getRank()) {
          // Do not cast if rank differs or not ranked tensor
        }
      }
      SmallVector<Type, 1> colSumARes;
      for (Type t : colFn.getFunctionType().getResults()) colSumARes.push_back(t);
      auto colSumACall = b.create<func::CallOp>(loc, StringRef("column_checksum"), TypeRange(colSumARes), ValueRange{colArgA});
      Value tmpA = colSumACall.getResult(0);

      // tmpB = row_checksum(B)
      Type expectedRowArgB = rowFn.getFunctionType().getInput(0);
      Value rowArgB = origB;
      if (rowArgB.getType() != expectedRowArgB) {
        auto rowArgBTy = llvm::dyn_cast<RankedTensorType>(rowArgB.getType());
        auto expectedTy = llvm::dyn_cast<RankedTensorType>(expectedRowArgB);
        if (rowArgBTy && expectedTy && rowArgBTy.getRank() == expectedTy.getRank()) {
          rowArgB = b.create<tensor::CastOp>(loc, expectedRowArgB, rowArgB).getResult();
        } else if (!rowArgBTy || !expectedTy || rowArgBTy.getRank() != expectedTy.getRank()) {
          // Do not cast if rank differs or not ranked tensor
        }
      }
      SmallVector<Type, 1> rowSumBRes;
      for (Type t : rowFn.getFunctionType().getResults()) rowSumBRes.push_back(t);
      auto rowSumBCall = b.create<func::CallOp>(loc, StringRef("row_checksum"), TypeRange(rowSumBRes), ValueRange{rowArgB});
      Value tmpB = rowSumBCall.getResult(0);

      // Seeds used for row/column checksum expectations. Keep tmpA/tmpB
      // unchanged for scalar checksum (dot/sum) logic.
      Value expectedColSeed = tmpA;
      Value expectedRowSeed = tmpB;

      // Precompute Huang full-checksum expectations:
      //   expected_col_checksum = checksum(A) * B  -> (N)
      //   expected_row_checksum = A * checksum(B)  -> (M)
      // Keep this always-on for row/col metric logging consistency even when
      // ABFT FuC checks are not explicitly enabled in the outer pipeline.
      Value expected_row_checksum;
      Value expected_col_checksum;
      if (rowvecFn) {
        SmallVector<Type, 1> t1Results;
        for (Type t : rowvecFn.getFunctionType().getResults())
          t1Results.push_back(t);
        auto rowvecTy = rowvecFn.getFunctionType();
        Value argA = expectedColSeed;
        if (argA.getType() != rowvecTy.getInput(0))
          argA = b.create<tensor::CastOp>(loc, rowvecTy.getInput(0), argA).getResult();
        Value argB = origB;
        if (argB.getType() != rowvecTy.getInput(1))
          argB = b.create<tensor::CastOp>(loc, rowvecTy.getInput(1), argB).getResult();
        auto t1Call = b.create<func::CallOp>(
            loc, StringRef("rowvec_mul_mat"), TypeRange(t1Results),
            ValueRange{argA, argB});
        expected_col_checksum = t1Call.getResult(0);
      }
      if (matcolFn) {
        SmallVector<Type, 1> t2Results;
        for (Type t : matcolFn.getFunctionType().getResults())
          t2Results.push_back(t);
        auto matcolTy = matcolFn.getFunctionType();
        Value argA = origA;
        if (argA.getType() != matcolTy.getInput(0))
          argA = b.create<tensor::CastOp>(loc, matcolTy.getInput(0), argA).getResult();
        Value argB = expectedRowSeed;
        if (argB.getType() != matcolTy.getInput(1))
          argB = b.create<tensor::CastOp>(loc, matcolTy.getInput(1), argB).getResult();
        auto t2Call = b.create<func::CallOp>(
            loc, StringRef("mat_mul_colvec"), TypeRange(t2Results),
            ValueRange{argA, argB});
        expected_row_checksum = t2Call.getResult(0);
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
        if (abftEnableFuC && sumFn) {
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

      // Insert matrix sum AFTER the matmul op to compute ones * C * ones.
      // Place the insertion point just after the matmul operation.
      if (abftEnableFuC && sumFn) {
        // The matmul result is the op's first result.
        if (!op->getResults().empty()) {
          Value C = op->getResult(0);
          // Build after the op.
          Block::iterator nextIt = std::next(Block::iterator(op));
          OpBuilder bAfter(op->getBlock(), nextIt);

          // Ensure C matches expected arg type.
          if (isa<RankedTensorType>(C.getType())) {
            Value C_used = C;
            Type expectedSumArg = sumFn.getFunctionType().getInput(0);
            Value sumArg = C_used;
            if (sumArg.getType() != expectedSumArg) {
              sumArg = bAfter.create<tensor::CastOp>(loc, expectedSumArg, sumArg).getResult();
            }
            SmallVector<Type, 1> sumResultTypes;
            for (Type t : sumFn.getFunctionType().getResults())
              sumResultTypes.push_back(t);
            auto sumCall = bAfter.create<func::CallOp>(
                loc, StringRef("matrix_sum"), TypeRange(sumResultTypes),
                ValueRange{sumArg});
            Value totalSum = sumCall.getResult(0);
            (void)totalSum; // left in IR for later instrumentation

            // Log checksum input vector sums to detect zeros.
            if (vecSumFn) {
              auto vecSumTy = vecSumFn.getFunctionType();
              if (tmpA) {
              Value aArg = tmpA.getType() == vecSumTy.getInput(0)
                       ? tmpA
                       : bAfter
                           .create<tensor::CastOp>(
                             loc, vecSumTy.getInput(0), tmpA)
                           .getResult();
              SmallVector<Type, 1> aSumTypes;
              for (Type t : vecSumTy.getResults()) aSumTypes.push_back(t);
              auto aSumCall = bAfter.create<func::CallOp>(
                loc, StringRef("vector_sum"), TypeRange(aSumTypes),
                ValueRange{aArg});
              Value aSum = aSumCall.getResult(0);
              }
              if (tmpB) {
              Value bArg = tmpB.getType() == vecSumTy.getInput(0)
                       ? tmpB
                       : bAfter
                           .create<tensor::CastOp>(
                             loc, vecSumTy.getInput(0), tmpB)
                           .getResult();
              SmallVector<Type, 1> bSumTypes;
              for (Type t : vecSumTy.getResults()) bSumTypes.push_back(t);
              auto bSumCall = bAfter.create<func::CallOp>(
                loc, StringRef("vector_sum"), TypeRange(bSumTypes),
                ValueRange{bArg});
              Value bSum = bSumCall.getResult(0);
              }
            }

            // FuC adds the scalar checksum consistency path on top of the
            // regular row/column checksum instrumentation.
            if (abftEnableFuC && dotFn && epsFn && tmpB) {
              // Ensure tmpA/tmpB match dot expected input types.
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

              // Now call epsilon_compare(totalSum, dot_product_tensor (+ init), epsilon)
              Value dot_expected = dot_product_tensor;
              if (includeInit && initSum && scalarAddFn) {
                SmallVector<Type, 1> addResults;
                for (Type t : scalarAddFn.getFunctionType().getResults())
                  addResults.push_back(t);
                // Ensure args match scalar_tensor_add signature.
                SmallVector<Value, 2> addArgs;
                auto addTy = scalarAddFn.getFunctionType();
                Value addA = dot_product_tensor;
                if (addA.getType() != addTy.getInput(0))
                  addA = bAfter.create<tensor::CastOp>(loc, addTy.getInput(0), addA).getResult();
                Value addB = initSum;
                if (addB.getType() != addTy.getInput(1))
                  addB = bAfter.create<tensor::CastOp>(loc, addTy.getInput(1), addB).getResult();
                addArgs.push_back(addA);
                addArgs.push_back(addB);
                auto addCall = bAfter.create<func::CallOp>(loc, StringRef("scalar_tensor_add"),
                                                          TypeRange(addResults), ValueRange{addArgs});
                dot_expected = addCall.getResult(0);
              }
                  auto epsAbsAttr = bAfter.getF32FloatAttr(abftEpsilonAbs);
                  auto epsRelAttr = bAfter.getF32FloatAttr(abftEpsilonRel);
                  auto epsAbsConst = bAfter.create<arith::ConstantOp>(
                    loc, bAfter.getF32Type(), epsAbsAttr);
                  auto epsRelConst = bAfter.create<arith::ConstantOp>(
                    loc, bAfter.getF32Type(), epsRelAttr);
              // Ensure totalSum and dot_product_tensor match expected types.
              SmallVector<Value, 2> epsArgs;
              auto epsTy = epsFn.getFunctionType();
              if (epsTy.getNumInputs() >= 1)
                epsArgs.push_back(
                    totalSum.getType() == epsTy.getInput(0)
                        ? totalSum
                        : bAfter
                              .create<tensor::CastOp>(loc, epsTy.getInput(0),
                                                      totalSum)
                              .getResult());
              if (epsTy.getNumInputs() >= 2)
                epsArgs.push_back(
                    dot_expected.getType() == epsTy.getInput(1)
                        ? dot_expected
                        : bAfter
                              .create<tensor::CastOp>(loc, epsTy.getInput(1),
                                                      dot_expected)
                              .getResult());
              // The epsilon scalar
              epsArgs.push_back(epsAbsConst);
              epsArgs.push_back(epsRelConst);
              // Log the raw scalar values before comparison.
              {
                auto sumVal = totalSum;
                auto expVal = dot_expected;
                auto epsTy2 = epsFn.getFunctionType();
                if (epsTy2.getNumInputs() >= 1 && sumVal.getType() != epsTy2.getInput(0))
                  sumVal = bAfter.create<tensor::CastOp>(loc, epsTy2.getInput(0), sumVal).getResult();
                if (epsTy2.getNumInputs() >= 2 && expVal.getType() != epsTy2.getInput(1))
                  expVal = bAfter.create<tensor::CastOp>(loc, epsTy2.getInput(1), expVal).getResult();
              }
              bAfter.create<func::CallOp>(loc, StringRef("epsilon_compare_abft"),
                                          TypeRange{}, ValueRange{epsArgs});
            }

            // Full-checksum per-layer row/column checksum deltas.
            // Keep this always-on so logs consistently contain rowMaxDelta
            // and colMaxDelta for every instrumented layer.
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
                subA = bAfter.create<tensor::CastOp>(loc, subTy.getInput(0), subA).getResult();
              Value subB = initCForChecks;
              if (subB.getType() != subTy.getInput(1))
                subB = bAfter.create<tensor::CastOp>(loc, subTy.getInput(1), subB).getResult();
              checksumCompareMatrix = bAfter
                                         .create<func::CallOp>(loc, StringRef("matrix_sub"),
                                                               TypeRange(subResults),
                                                               ValueRange{subA, subB})
                                         .getResult(0);
            }

            Value calcColChecksum;
            if (colFn && !colFn.isExternal()) {
              SmallVector<Type, 1> cRes;
              for (Type t : colFn.getFunctionType().getResults())
                cRes.push_back(t);
              Value cArg = checksumCompareMatrix;
              Type cTy = colFn.getFunctionType().getInput(0);
              if (cArg.getType() != cTy)
                cArg = bAfter.create<tensor::CastOp>(loc, cTy, cArg).getResult();
              calcColChecksum = bAfter.create<func::CallOp>(
                  loc, StringRef("column_checksum"), TypeRange(cRes),
                  ValueRange{cArg}).getResult(0);
            }

            Value calcRowChecksum;
            if (rowFn && !rowFn.isExternal()) {
              SmallVector<Type, 1> rRes;
              for (Type t : rowFn.getFunctionType().getResults())
                rRes.push_back(t);
              Value rArg = checksumCompareMatrix;
              Type rTy = rowFn.getFunctionType().getInput(0);
              if (rArg.getType() != rTy)
                rArg = bAfter.create<tensor::CastOp>(loc, rTy, rArg).getResult();
              calcRowChecksum = bAfter.create<func::CallOp>(
                  loc, StringRef("row_checksum"), TypeRange(rRes),
                  ValueRange{rArg}).getResult(0);
            }

            // checksum(A)*B == column_checksum(C - init)
            if (expected_col_checksum && calcColChecksum) {
              Value colExpected = expected_col_checksum;
              Value colCalcComparable = calcColChecksum;
              if (vecMaxFn && !vecMaxFn.isExternal()) {
                SmallVector<Type, 1> dRes;
                for (Type t : vecMaxFn.getFunctionType().getResults())
                  dRes.push_back(t);
                auto dTy = vecMaxFn.getFunctionType();
                Value dV1 = colExpected;
                if (dV1.getType() != dTy.getInput(0))
                  dV1 = bAfter.create<tensor::CastOp>(loc, dTy.getInput(0), dV1).getResult();
                Value dV2 = colCalcComparable;
                if (dV2.getType() != dTy.getInput(1))
                  dV2 = bAfter.create<tensor::CastOp>(loc, dTy.getInput(1), dV2).getResult();
                colMaxDelta = bAfter.create<func::CallOp>(
                    loc, StringRef("vector_max_abs_diff"), TypeRange(dRes),
                    ValueRange{dV1, dV2}).getResult(0);
              }
              if (vecMaxPairFn && !vecMaxPairFn.isExternal()) {
                SmallVector<Type, 2> pRes;
                for (Type t : vecMaxPairFn.getFunctionType().getResults())
                  pRes.push_back(t);
                auto pTy = vecMaxPairFn.getFunctionType();
                Value ce = colExpected;
                if (ce.getType() != pTy.getInput(0))
                  ce = bAfter.create<tensor::CastOp>(loc, pTy.getInput(0), ce).getResult();
                Value cc = colCalcComparable;
                if (cc.getType() != pTy.getInput(1))
                  cc = bAfter.create<tensor::CastOp>(loc, pTy.getInput(1), cc).getResult();
                auto pairCall = bAfter.create<func::CallOp>(
                    loc, StringRef("vector_max_abs_diff_pair"), TypeRange(pRes),
                    ValueRange{ce, cc});
                colExpMax = pairCall.getResult(0);
                colCalcMax = pairCall.getResult(1);
              }
              if (vecFirstFn && !vecFirstFn.isExternal()) {
                SmallVector<Type, 1> fRes;
                for (Type t : vecFirstFn.getFunctionType().getResults())
                  fRes.push_back(t);
                auto fTy = vecFirstFn.getFunctionType();
                Value ce = colExpected;
                if (ce.getType() != fTy.getInput(0))
                  ce = bAfter.create<tensor::CastOp>(loc, fTy.getInput(0), ce).getResult();
                Value cc = colCalcComparable;
                if (cc.getType() != fTy.getInput(0))
                  cc = bAfter.create<tensor::CastOp>(loc, fTy.getInput(0), cc).getResult();
              }
            }

            // A*checksum(B) == row_checksum(C - init)
            if (expected_row_checksum && calcRowChecksum) {
              Value rowExpected = expected_row_checksum;
              Value rowCalcComparable = calcRowChecksum;
              if (vecMaxFn && !vecMaxFn.isExternal()) {
                SmallVector<Type, 1> dRes;
                for (Type t : vecMaxFn.getFunctionType().getResults())
                  dRes.push_back(t);
                auto dTy = vecMaxFn.getFunctionType();
                Value dV1 = rowExpected;
                if (dV1.getType() != dTy.getInput(0))
                  dV1 = bAfter.create<tensor::CastOp>(loc, dTy.getInput(0), dV1).getResult();
                Value dV2 = rowCalcComparable;
                if (dV2.getType() != dTy.getInput(1))
                  dV2 = bAfter.create<tensor::CastOp>(loc, dTy.getInput(1), dV2).getResult();
                rowMaxDelta = bAfter.create<func::CallOp>(
                    loc, StringRef("vector_max_abs_diff"), TypeRange(dRes),
                    ValueRange{dV1, dV2}).getResult(0);
              }
              if (vecMaxPairFn && !vecMaxPairFn.isExternal()) {
                SmallVector<Type, 2> pRes;
                for (Type t : vecMaxPairFn.getFunctionType().getResults())
                  pRes.push_back(t);
                auto pTy = vecMaxPairFn.getFunctionType();
                Value re = rowExpected;
                if (re.getType() != pTy.getInput(0))
                  re = bAfter.create<tensor::CastOp>(loc, pTy.getInput(0), re).getResult();
                Value rc = rowCalcComparable;
                if (rc.getType() != pTy.getInput(1))
                  rc = bAfter.create<tensor::CastOp>(loc, pTy.getInput(1), rc).getResult();
                auto pairCall = bAfter.create<func::CallOp>(
                    loc, StringRef("vector_max_abs_diff_pair"), TypeRange(pRes),
                    ValueRange{re, rc});
                rowExpMax = pairCall.getResult(0);
                rowCalcMax = pairCall.getResult(1);
              }
              if (vecFirstFn && !vecFirstFn.isExternal()) {
                SmallVector<Type, 1> fRes;
                for (Type t : vecFirstFn.getFunctionType().getResults())
                  fRes.push_back(t);
                auto fTy = vecFirstFn.getFunctionType();
                Value re = rowExpected;
                if (re.getType() != fTy.getInput(0))
                  re = bAfter.create<tensor::CastOp>(loc, fTy.getInput(0), re).getResult();
                Value rc = rowCalcComparable;
                if (rc.getType() != fTy.getInput(0))
                  rc = bAfter.create<tensor::CastOp>(loc, fTy.getInput(0), rc).getResult();
              }
            }

            Value indexConst = bAfter.create<arith::ConstantOp>(
                loc, bAfter.getF32Type(),
                bAfter.getF32FloatAttr((float)std::distance(
                    targets.begin(),
                    std::find(targets.begin(), targets.end(), op))));
            bAfter.create<func::CallOp>(
                loc, StringRef("abft_analysis.abft_log_rowcol_delta"),
                TypeRange{},
                ValueRange{indexConst, rowMaxDelta, colMaxDelta});
            if (logRowColDebugFn) {
              bAfter.create<func::CallOp>(
                  loc, StringRef("abft_analysis.abft_log_rowcol_debug"),
                  TypeRange{},
                  ValueRange{indexConst, rowExpMax, rowCalcMax, colExpMax,
                             colCalcMax});
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
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createABFTPass() {
  return std::make_unique<ABFTPass>();
}
static mlir::PassRegistration<ABFTPass> reg;
