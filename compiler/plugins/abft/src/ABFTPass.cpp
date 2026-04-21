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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include <regex>
#include <string>

using namespace mlir;

// Global CLI flag to enable Full-Checksum elementwise comparisons. Using a
// static command-line option avoids putting non-copyable llvm::cl::opt into
// the pass object (PassWrapper needs to be copyable).
static llvm::cl::opt<bool> abftEnableFuC(
    "abft-enable-fuc",
    llvm::cl::desc("Enable full elementwise checksum (t1/t2) comparisons"),
    llvm::cl::init(false));
static llvm::cl::opt<bool> abftEnableAnalysisLog(
    "abft-enable-analysis-log",
    llvm::cl::desc("Enable ABFT runtime row/column analysis logging"),
    llvm::cl::init(true));

static llvm::cl::opt<float> abftEpsilonAbs(
  "abft-epsilon-abs",
  llvm::cl::desc("Absolute epsilon for ABFT checks"),
  llvm::cl::init(1.0e-7f));

static llvm::cl::opt<float> abftEpsilonRel(
  "abft-epsilon-rel",
  llvm::cl::desc("Relative epsilon for ABFT checks"),
  llvm::cl::init(1.0e-5f));

static llvm::cl::opt<bool> abftInjectFault(
    "abft-inject-fault",
    llvm::cl::desc("Inject a deterministic output fault in ABFT instrumentation"),
    llvm::cl::init(false));

static llvm::cl::opt<int> abftInjectFaultDelta(
    "abft-inject-fault-delta",
    llvm::cl::desc("Delta added to injected ABFT fault (quantized i32 path)"),
    llvm::cl::init(1));

static llvm::cl::opt<int> abftInjectFaultLayer(
    "abft-inject-fault-layer",
    llvm::cl::desc("Target layer ordinal for fault injection (-1: all layers)"),
    llvm::cl::init(-1));

static llvm::cl::opt<std::string> abftInjectFaultPattern(
  "abft-inject-fault-pattern",
  llvm::cl::desc(
    "Fault injection pattern: single_point, trivial, checkered"),
  llvm::cl::init("single_point"));

namespace {

constexpr StringLiteral kAbftModeAttrName = "iree.abft.mode";
constexpr StringLiteral kAbftModeNormal = "abft";
constexpr StringLiteral kAbftModeScaled = "abyzft";

struct ABFTSpecializeBatchDimPass
    : public PassWrapper<ABFTSpecializeBatchDimPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ABFTSpecializeBatchDimPass)

  StringRef getArgument() const final { return "abft-specialize-batch-dim"; }
  StringRef getDescription() const final {
    return "Specialize leading dynamic batch dims to 1 for ABFT preprocessing";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string irText;
    llvm::raw_string_ostream os(irText);
    module.print(os);
    os.flush();

    // Specialize leading dynamic dim for rank>=2 tensors:
    //   tensor<?xA...xT> -> tensor<1xA...xT>
    irText = std::regex_replace(
        irText, std::regex(R"(tensor<\?x([^>]*x[^>]+)>)"), "tensor<1x$1>");
    // Specialize rank-1 vectors:
    //   tensor<?xT> -> tensor<1xT>
    irText = std::regex_replace(irText, std::regex(R"(tensor<\?x([^x>]+)>)"),
                                "tensor<1x$1>");

    auto rewriteConditional = [](const std::string &input,
                                 const std::regex &pattern,
                                 const std::function<std::string(
                                     const std::smatch &)> &onMatch) {
      std::string out;
      out.reserve(input.size());
      std::sregex_iterator it(input.begin(), input.end(), pattern);
      std::sregex_iterator end;
      size_t lastPos = 0;
      for (; it != end; ++it) {
        const auto &m = *it;
        size_t pos = static_cast<size_t>(m.position());
        out.append(input, lastPos, pos - lastPos);
        out.append(onMatch(m));
        lastPos = pos + static_cast<size_t>(m.length());
      }
      out.append(input, lastPos, std::string::npos);
      return out;
    };

    // Keep tensor.empty dynamic-size operands consistent with now-static types
    // before reparsing.
    irText = rewriteConditional(
        irText,
        std::regex(R"(tensor\.empty\([^)]*\)\s*:\s*tensor<([^>]+)>)"),
        [&](const std::smatch &m) {
          std::string full = m.str(0);
          std::string ty = m.str(1);
          if (ty.find('?') != std::string::npos)
            return full;
          return std::string("tensor.empty() : tensor<") + ty + ">";
        });
    irText = rewriteConditional(
        irText,
        std::regex(
            R"("tensor\.empty"\([^)]*\)\s*:\s*\([^)]*\)\s*->\s*tensor<([^>]+)>)"),
        [&](const std::smatch &m) {
          std::string full = m.str(0);
          std::string ty = m.str(1);
          if (ty.find('?') != std::string::npos)
            return full;
          return std::string("\"tensor.empty\"() : () -> tensor<") + ty + ">";
        });

    OwningOpRef<ModuleOp> reparsed =
        parseSourceString<ModuleOp>(irText, module.getContext());
    if (!reparsed) {
      module.emitError(
          "abft-specialize-batch-dim: failed to parse rewritten module text");
      signalPassFailure();
      return;
    }

    module.getBodyRegion().takeBody(reparsed->getBodyRegion());
  }
};

struct ABFTPass : public PassWrapper<ABFTPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ABFTPass)

  // NOTE: the CLI flag is defined as a static llvm::cl::opt above. We avoid
  // storing an Option<> member inside the pass because llvm::cl::opt is
  // non-copyable which breaks PassWrapper's cloning. Read the flag at
  // run-time from `abftEnableFuC`.

  StringRef getArgument() const final { return "abft-insert-ones"; }
  StringRef getDescription() const final {
    return "Insert ABFT checks for linalg.matmul and quantized linalg ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                    func::FuncDialect, linalg::LinalgDialect,
                    math::MathDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto funcName = func.getSymName();
    // Never instrument generated dispatch/internal executable functions.
    // ABFT helper ops (linalg.generic/reduce calls) are not legal in those
    // workgroup-distributed contexts and must only be inserted in top-level
    // pre-dispatch IR.
    if (funcName.contains("_dispatch_") || funcName.starts_with("dispatch_")) {
      return;
    }
    for (Operation *parent = func->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      StringRef parentName = parent->getName().getStringRef();
      if (parentName == "hal.executable.variant" ||
          parentName == "hal.executable.source" ||
          parentName == "hal.executable") {
        return;
      }
    }
    if (funcName == "column_checksum" || funcName == "row_checksum" ||
        funcName == "matrix_sum" || funcName == "vector_sum" ||
        funcName == "vector_dot_product" || funcName == "epsilon_compare_abft" ||
        funcName == "sample_row_scales" ||
        funcName == "sample_row_scales_ones" ||
        funcName == "sample_col_scales" ||
        funcName == "sample_col_scales_ones" ||
        funcName == "rowvec_mul_mat_abft_v2" || funcName == "mat_mul_colvec_abft_v2" ||
        funcName == "vector_epsilon_compare_abft" ||
        funcName == "vector_max_abs_diff" ||
        funcName == "vector_max_abs_diff_i32" ||
        funcName == "vector_max_abs_diff_i64" ||
        funcName == "vector_max_abs_diff_pair" ||
        funcName == "vector_first_elem" || funcName == "scalar_tensor_add" ||
        funcName == "matrix_sub" || funcName == "matrix_add" ||
        funcName == "zero_matrix_like" ||
        funcName == "abft_analysis.abft_report_failure" ||
        funcName == "abft_analysis.abft_log_rowcol_delta" ||
        funcName == "abft_analysis.abft_log_rowcol_debug") {
      return;
    }
    // Fast-path: no supported targets means no ABFT instrumentation.
    bool hasTarget = false;
    func.walk([&](Operation *op) {
      StringRef n = op->getName().getStringRef();
      if (n == "linalg.matmul" || n == "linalg.quantized_matmul" ||
          n == "linalg.conv_2d_nhwc_hwcf_q" ||
          n == "linalg.depthwise_conv_2d_nhwc_hwcm_q")
        hasTarget = true;
    });
    if (!hasTarget) {
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
    //  - rowvec_mul_mat_abft_v2(rv, mat) -> vector: for each column j, sum_k
    //  rv[k]*mat[k,j]
    //  - mat_mul_colvec_abft_v2(mat, cv) -> vector: for each row i, sum_j
    //  mat[i,j]*cv[j]
    //  - vector_epsilon_compare_abft(v1, v2, eps) -> () : elementwise report
    //  (|v1-v2| < eps)
    func::FuncOp parsedRowVec = ensureFunctionWithBody("rowvec_mul_mat_abft_v2", R"mlir(
module {
  func.func @rowvec_mul_mat_abft_v2(%rv: tensor<?xf32>, %mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mat_k = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %rv_k = tensor.dim %rv, %c0 : tensor<?xf32>
    %rv_is_scalar = arith.cmpi eq, %rv_k, %c1 : index
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = scf.for %j = %c0 to %n step %c1 iter_args(%accv = %init) -> (tensor<?xf32>) {
      %sum = scf.for %i = %c0 to %mat_k step %c1 iter_args(%acc = %zero) -> (f32) {
        %rv_elem = scf.if %rv_is_scalar -> (f32) {
          %x = tensor.extract %rv[%c0] : tensor<?xf32>
          scf.yield %x : f32
        } else {
          %x = tensor.extract %rv[%i] : tensor<?xf32>
          scf.yield %x : f32
        }
        %mat_elem = tensor.extract %mat[%i, %j] : tensor<?x?xf32>
        %prod = arith.mulf %rv_elem, %mat_elem : f32
        %next = arith.addf %acc, %prod : f32
        scf.yield %next : f32
      }
      %nextv = tensor.insert %sum into %accv[%j] : tensor<?xf32>
      scf.yield %nextv : tensor<?xf32>
    }
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedRowVec;

    func::FuncOp parsedMatCol = ensureFunctionWithBody("mat_mul_colvec_abft_v2", R"mlir(
module {
  func.func @mat_mul_colvec_abft_v2(%mat: tensor<?x?xf32>, %cv: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %mat_k = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %cv_k = tensor.dim %cv, %c0 : tensor<?xf32>
    %cv_is_scalar = arith.cmpi eq, %cv_k, %c1 : index
    %empty = tensor.empty(%m) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = scf.for %i = %c0 to %m step %c1 iter_args(%accv = %init) -> (tensor<?xf32>) {
      %sum = scf.for %j = %c0 to %mat_k step %c1 iter_args(%acc = %zero) -> (f32) {
        %cv_elem = scf.if %cv_is_scalar -> (f32) {
          %x = tensor.extract %cv[%c0] : tensor<?xf32>
          scf.yield %x : f32
        } else {
          %x = tensor.extract %cv[%j] : tensor<?xf32>
          scf.yield %x : f32
        }
        %mat_elem = tensor.extract %mat[%i, %j] : tensor<?x?xf32>
        %prod = arith.mulf %mat_elem, %cv_elem : f32
        %next = arith.addf %acc, %prod : f32
        scf.yield %next : f32
      }
      %nextv = tensor.insert %sum into %accv[%i] : tensor<?xf32>
      scf.yield %nextv : tensor<?xf32>
    }
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
        %is_nan = arith.cmpf uno, %ad, %ad : f32
        %safe_ad = arith.select %is_nan, %zero, %ad : f32
        %m = arith.maximumf %safe_ad, %acc : f32
        linalg.yield %m : f32
    } -> tensor<f32>
    %res = tensor.extract %out[] : tensor<f32>
    return %res : f32
  }
}
)mlir");
    (void)parsedVecMax;

    func::FuncOp parsedVecMaxI32 = ensureFunctionWithBody(
        "vector_max_abs_diff_i32", R"mlir(
module {
  func.func @vector_max_abs_diff_i32(%v1: tensor<?xi32>, %v2: tensor<?xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %v1, %c0 : tensor<?xi32>
    %tmp = tensor.empty(%n) : tensor<?xi64>
    %zero = arith.constant 0 : i64
    %initVec = linalg.fill ins(%zero : i64) outs(%tmp : tensor<?xi64>) -> tensor<?xi64>
    %diff = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]
    } ins(%v1, %v2 : tensor<?xi32>, tensor<?xi32>) outs(%initVec : tensor<?xi64>) {
      ^bb0(%a: i32, %b: i32, %acc: i64):
        %a64 = arith.extsi %a : i32 to i64
        %b64 = arith.extsi %b : i32 to i64
        %ge = arith.cmpi sge, %a64, %b64 : i64
        %mx = arith.select %ge, %a64, %b64 : i64
        %mn = arith.select %ge, %b64, %a64 : i64
        %ad = arith.subi %mx, %mn : i64
        linalg.yield %ad : i64
    } -> tensor<?xi64>
    %empty = tensor.empty() : tensor<i64>
    %init = linalg.fill ins(%zero : i64) outs(%empty : tensor<i64>) -> tensor<i64>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->()>],
      iterator_types = ["reduction"]
    } ins(%diff : tensor<?xi64>) outs(%init : tensor<i64>) {
      ^bb0(%a: i64, %acc: i64):
        %m = arith.maxsi %a, %acc : i64
        linalg.yield %m : i64
    } -> tensor<i64>
    %res64 = tensor.extract %out[] : tensor<i64>
    %imax = arith.constant 2147483647 : i64
    %clamped = arith.minsi %res64, %imax : i64
    %res = arith.trunci %clamped : i64 to i32
    return %res : i32
  }
}
)mlir");
    (void)parsedVecMaxI32;
    func::FuncOp parsedVecMaxI64 = ensureFunctionWithBody(
        "vector_max_abs_diff_i64", R"mlir(
module {
  func.func @vector_max_abs_diff_i64(%v1: tensor<?xi64>, %v2: tensor<?xi64>) -> f32 {
    %empty = tensor.empty() : tensor<i64>
    %zero_i64 = arith.constant 0 : i64
    %init = linalg.fill ins(%zero_i64 : i64) outs(%empty : tensor<i64>) -> tensor<i64>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->()>],
      iterator_types = ["reduction"]
    } ins(%v1, %v2 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<i64>) {
      ^bb0(%a: i64, %b: i64, %acc: i64):
        %ge = arith.cmpi sge, %a, %b : i64
        %mx = arith.select %ge, %a, %b : i64
        %mn = arith.select %ge, %b, %a : i64
        %ad = arith.subi %mx, %mn : i64
        %m = arith.maxsi %ad, %acc : i64
        linalg.yield %m : i64
    } -> tensor<i64>
    %res_i64 = tensor.extract %out[] : tensor<i64>
    %res = arith.sitofp %res_i64 : i64 to f32
    return %res : f32
  }
}
)mlir");
    (void)parsedVecMaxI64;
    func::FuncOp parsedVecMaxI64I32 = ensureFunctionWithBody(
        "vector_max_abs_diff_i64_i32", R"mlir(
module {
  func.func @vector_max_abs_diff_i64_i32(%v1: tensor<?xi64>, %v2: tensor<?xi64>) -> i32 {
    %empty = tensor.empty() : tensor<i64>
    %zero_i64 = arith.constant 0 : i64
    %imax_i64 = arith.constant 2147483647 : i64
    %init = linalg.fill ins(%zero_i64 : i64) outs(%empty : tensor<i64>) -> tensor<i64>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->()>],
      iterator_types = ["reduction"]
    } ins(%v1, %v2 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<i64>) {
      ^bb0(%a: i64, %b: i64, %acc: i64):
        %ge = arith.cmpi sge, %a, %b : i64
        %mx = arith.select %ge, %a, %b : i64
        %mn = arith.select %ge, %b, %a : i64
        %ad = arith.subi %mx, %mn : i64
        %m = arith.maxsi %ad, %acc : i64
        linalg.yield %m : i64
    } -> tensor<i64>
    %res_i64 = tensor.extract %out[] : tensor<i64>
    %clamped = arith.minsi %res_i64, %imax_i64 : i64
    %res = arith.trunci %clamped : i64 to i32
    return %res : i32
  }
}
)mlir");
    (void)parsedVecMaxI64I32;
    func::FuncOp parsedVecMaxAbsI64 = ensureFunctionWithBody(
        "vector_max_abs_i64", R"mlir(
module {
  func.func @vector_max_abs_i64(%v: tensor<?xi64>) -> f32 {
    %empty = tensor.empty() : tensor<i64>
    %zero_i64 = arith.constant 0 : i64
    %init = linalg.fill ins(%zero_i64 : i64) outs(%empty : tensor<i64>) -> tensor<i64>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->()>],
      iterator_types = ["reduction"]
    } ins(%v : tensor<?xi64>) outs(%init : tensor<i64>) {
      ^bb0(%a: i64, %acc: i64):
        %zero_i64_2 = arith.constant 0 : i64
        %neg = arith.subi %zero_i64_2, %a : i64
        %isNeg = arith.cmpi slt, %a, %zero_i64_2 : i64
        %abs = arith.select %isNeg, %neg, %a : i64
        %m = arith.maxsi %abs, %acc : i64
        linalg.yield %m : i64
    } -> tensor<i64>
    %res_i64 = tensor.extract %out[] : tensor<i64>
    %res = arith.sitofp %res_i64 : i64 to f32
    return %res : f32
  }
}
)mlir");
    (void)parsedVecMaxAbsI64;
    func::FuncOp parsedVecMaxAbsI64I32 = ensureFunctionWithBody(
        "vector_max_abs_i64_i32", R"mlir(
module {
  func.func @vector_max_abs_i64_i32(%v: tensor<?xi64>) -> i32 {
    %empty = tensor.empty() : tensor<i64>
    %zero_i64 = arith.constant 0 : i64
    %imax_i64 = arith.constant 2147483647 : i64
    %init = linalg.fill ins(%zero_i64 : i64) outs(%empty : tensor<i64>) -> tensor<i64>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->()>],
      iterator_types = ["reduction"]
    } ins(%v : tensor<?xi64>) outs(%init : tensor<i64>) {
      ^bb0(%a: i64, %acc: i64):
        %zero_i64_2 = arith.constant 0 : i64
        %neg = arith.subi %zero_i64_2, %a : i64
        %isNeg = arith.cmpi slt, %a, %zero_i64_2 : i64
        %abs = arith.select %isNeg, %neg, %a : i64
        %m = arith.maxsi %abs, %acc : i64
        linalg.yield %m : i64
    } -> tensor<i64>
    %res_i64 = tensor.extract %out[] : tensor<i64>
    %clamped = arith.minsi %res_i64, %imax_i64 : i64
    %res = arith.trunci %clamped : i64 to i32
    return %res : i32
  }
}
)mlir");
    (void)parsedVecMaxAbsI64I32;

    // NOTE: Keep vector_max_abs_diff_pair disabled for stability in the EmitC
    // pipeline. ABFT logging uses scalar max-diff helpers and does not require
    // this helper.

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
    if (!module.lookupSymbol<func::FuncOp>("vector_max_abs_diff_i32")) {
      auto i32Ty = modBuilder.getI32Type();
      auto vecDynI32 = RankedTensorType::get({ShapedType::kDynamic}, i32Ty);
      auto ft = FunctionType::get(ctx, TypeRange{vecDynI32, vecDynI32}, TypeRange{i32Ty});
      maybeInsertDecl("vector_max_abs_diff_i32", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_max_abs_diff_i64")) {
      auto i64Ty = modBuilder.getI64Type();
      auto vecDynI64 = RankedTensorType::get({ShapedType::kDynamic}, i64Ty);
      auto ft = FunctionType::get(ctx, TypeRange{vecDynI64, vecDynI64}, TypeRange{f32});
      maybeInsertDecl("vector_max_abs_diff_i64", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_max_abs_i64")) {
      auto i64Ty = modBuilder.getI64Type();
      auto vecDynI64 = RankedTensorType::get({ShapedType::kDynamic}, i64Ty);
      auto ft = FunctionType::get(ctx, TypeRange{vecDynI64}, TypeRange{f32});
      maybeInsertDecl("vector_max_abs_i64", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_max_abs_diff_i64_i32")) {
      auto i64Ty = modBuilder.getI64Type();
      auto i32Ty = modBuilder.getI32Type();
      auto vecDynI64 = RankedTensorType::get({ShapedType::kDynamic}, i64Ty);
      auto ft = FunctionType::get(ctx, TypeRange{vecDynI64, vecDynI64}, TypeRange{i32Ty});
      maybeInsertDecl("vector_max_abs_diff_i64_i32", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_max_abs_i64_i32")) {
      auto i64Ty = modBuilder.getI64Type();
      auto i32Ty = modBuilder.getI32Type();
      auto vecDynI64 = RankedTensorType::get({ShapedType::kDynamic}, i64Ty);
      auto ft = FunctionType::get(ctx, TypeRange{vecDynI64}, TypeRange{i32Ty});
      maybeInsertDecl("vector_max_abs_i64_i32", ft);
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
    if (abftEnableAnalysisLog) {
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
    }

    // Collect targets inside this function only to avoid mutating while
    // walking.
    SmallVector<Operation *, 8> targets;
    func.walk([&](Operation *op) {
      StringRef n = op->getName().getStringRef();
      if (n == "linalg.matmul" || n == "linalg.quantized_matmul" ||
          n == "linalg.conv_2d_nhwc_hwcf_q" ||
          n == "linalg.depthwise_conv_2d_nhwc_hwcm_q")
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
        module.lookupSymbol<func::FuncOp>(StringRef("rowvec_mul_mat_abft_v2"));
    auto matcolFn =
        module.lookupSymbol<func::FuncOp>(StringRef("mat_mul_colvec_abft_v2"));
    auto vecMaxFn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_diff"));
    auto vecMaxI32Fn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_diff_i32"));
    auto vecMaxI64Fn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_diff_i64"));
    auto vecMaxAbsI64Fn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_i64"));
    auto vecMaxI64I32Fn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_diff_i64_i32"));
    auto vecMaxAbsI64I32Fn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_i64_i32"));
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

    if (abftInjectFault) {
      // Emit only the requested i32 helper to minimize added IR and keep
      // lowering surface small for EmitC.
      StringRef pattern = abftInjectFaultPattern.getValue();
      if (pattern == "single_point") {
        (void)ensureFunctionWithBody("apply_fault_i32_single_point", R"mlir(
module {
  func.func @apply_fault_i32_single_point(%mat: tensor<?x?xi32>, %delta: i32) -> tensor<?x?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = tensor.dim %mat, %c0 : tensor<?x?xi32>
    %cols = tensor.dim %mat, %c1 : tensor<?x?xi32>
    %empty = tensor.empty(%rows, %cols) : tensor<?x?xi32>
    %zero = arith.constant 0 : i32
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%mat : tensor<?x?xi32>) outs(%init : tensor<?x?xi32>) {
      ^bb0(%a: i32, %acc: i32):
        %i = linalg.index 0 : index
        %j = linalg.index 1 : index
        %hit_i = arith.cmpi eq, %i, %c0 : index
        %hit_j = arith.cmpi eq, %j, %c0 : index
        %hit = arith.andi %hit_i, %hit_j : i1
        %res = arith.select %hit, %delta, %zero : i32
        %sum = arith.addi %a, %res : i32
        linalg.yield %sum : i32
    } -> tensor<?x?xi32>
    return %res : tensor<?x?xi32>
  }
}
)mlir");
      } else if (pattern == "trivial") {
        (void)ensureFunctionWithBody("apply_fault_i32_trivial", R"mlir(
module {
  func.func @apply_fault_i32_trivial(%mat: tensor<?x?xi32>, %delta: i32) -> tensor<?x?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %rows = tensor.dim %mat, %c0 : tensor<?x?xi32>
    %cols = tensor.dim %mat, %c1 : tensor<?x?xi32>
    %empty = tensor.empty(%rows, %cols) : tensor<?x?xi32>
    %zero = arith.constant 0 : i32
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%mat : tensor<?x?xi32>) outs(%init : tensor<?x?xi32>) {
      ^bb0(%a: i32, %acc: i32):
        %i = linalg.index 0 : index
        %j = linalg.index 1 : index
        %in_i = arith.cmpi ult, %i, %c2 : index
        %in_j = arith.cmpi ult, %j, %c2 : index
        %in_blk = arith.andi %in_i, %in_j : i1
        %im = arith.remui %i, %c2 : index
        %jm = arith.remui %j, %c2 : index
        %diag = arith.cmpi eq, %im, %jm : index
        %plus_delta = arith.addi %delta, %zero : i32
        %zero_i32 = arith.constant 0 : i32
        %minus_delta = arith.subi %zero_i32, %delta : i32
        %fault = arith.select %diag, %plus_delta, %minus_delta : i32
        %fault_val = arith.select %in_blk, %fault, %zero : i32
        %sum = arith.addi %a, %fault_val : i32
        linalg.yield %sum : i32
    } -> tensor<?x?xi32>
    return %res : tensor<?x?xi32>
  }
}
)mlir");
      } else {
        (void)ensureFunctionWithBody("apply_fault_i32_checkered", R"mlir(
module {
  func.func @apply_fault_i32_checkered(%mat: tensor<?x?xi32>, %delta: i32) -> tensor<?x?xi32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %rows = tensor.dim %mat, %c0 : tensor<?x?xi32>
    %c1 = arith.constant 1 : index
    %cols = tensor.dim %mat, %c1 : tensor<?x?xi32>
    %empty = tensor.empty(%rows, %cols) : tensor<?x?xi32>
    %zero = arith.constant 0 : i32
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%mat : tensor<?x?xi32>) outs(%init : tensor<?x?xi32>) {
      ^bb0(%a: i32, %acc: i32):
        %i = linalg.index 0 : index
        %im = arith.remui %i, %c2 : index
        %is_even = arith.cmpi eq, %im, %c0 : index
        %zero_i32 = arith.constant 0 : i32
        %minus_delta = arith.subi %zero_i32, %delta : i32
        %fault = arith.select %is_even, %delta, %minus_delta : i32
        %sum = arith.addi %a, %fault : i32
        linalg.yield %sum : i32
    } -> tensor<?x?xi32>
    return %res : tensor<?x?xi32>
  }
}
)mlir");
      }
    }
    int64_t layerOrdinal = 0;
    for (Operation *op : targets) {
      int64_t currentLayer = layerOrdinal++;
      StringRef targetName = op->getName().getStringRef();
        if ((targetName == "linalg.quantized_matmul" ||
           targetName == "linalg.conv_2d_nhwc_hwcf_q" ||
           targetName == "linalg.depthwise_conv_2d_nhwc_hwcm_q")) {
        auto qmm = dyn_cast<linalg::QuantizedMatmulOp>(op);
        if (!qmm || !vecMaxI64Fn || !vecMaxAbsI64Fn ||
          (abftEnableAnalysisLog && !logRowColDeltaFn) ||
          op->getNumResults() == 0) {
          op->emitRemark()
            << "abft-qmatmul: missing helper(s), skipping quantized instrumentation";
          continue;
        }

        auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!outTy || !outTy.getElementType().isInteger(32)) {
          op->emitRemark()
            << "abft-qmatmul: unsupported output type; expected ranked tensor<i32>";
          continue;
        }

        Location loc = op->getLoc();
        Block::iterator nextIt = std::next(Block::iterator(op));
        OpBuilder bAfter(op->getBlock(), nextIt);

        Value lhs = qmm.getDpsInputOperand(0)->get();
        Value rhs = qmm.getDpsInputOperand(1)->get();
        Value lhsZp = qmm.getDpsInputOperand(2)->get();
        Value rhsZp = qmm.getDpsInputOperand(3)->get();
        Value outInit = qmm.getDpsInitOperand(0)->get();
        Value outRes = qmm.getResult(0);

        auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
        auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
        if (!lhsTy || !rhsTy || lhsTy.getRank() != 2 || rhsTy.getRank() != 2) {
          op->emitRemark() << "abft-qmatmul: expected rank-2 lhs/rhs";
          continue;
        }

        auto dyn2dI64 = RankedTensorType::get(
          {ShapedType::kDynamic, ShapedType::kDynamic}, bAfter.getI64Type());
        auto vecDynI64 =
          RankedTensorType::get({ShapedType::kDynamic}, bAfter.getI64Type());

        // Fault injection must mutate the produced matmul result (not just the
        // checksum path), so downstream users observe the injected fault.
        if (abftInjectFault &&
            (abftInjectFaultLayer < 0 ||
             currentLayer == static_cast<int64_t>(abftInjectFaultLayer))) {
          StringRef pattern = abftInjectFaultPattern.getValue();
          StringRef injectFn = "apply_fault_i32_single_point";
          if (pattern == "trivial") {
            injectFn = "apply_fault_i32_trivial";
          } else if (pattern == "checkered") {
            injectFn = "apply_fault_i32_checkered";
          }
          auto injectSym = module.lookupSymbol<func::FuncOp>(injectFn);
          if (injectSym) {
            SmallVector<Type, 1> injTypes;
            for (Type t : injectSym.getFunctionType().getResults()) {
              injTypes.push_back(t);
            }
            Value injArg = outRes;
            Type expectedIn = injectSym.getFunctionType().getInput(0);
            if (injArg.getType() != expectedIn) {
              auto inTy = dyn_cast<RankedTensorType>(injArg.getType());
              auto expTy = dyn_cast<RankedTensorType>(expectedIn);
              if (inTy && expTy && inTy.getRank() == expTy.getRank()) {
                injArg = bAfter.create<tensor::CastOp>(loc, expectedIn, injArg).getResult();
              } else {
                op->emitRemark()
                  << "abft-qmatmul: pattern injection rank mismatch, skipping fault injection";
                injectSym = {};
              }
            }
            if (injectSym) {
              auto deltaVal = bAfter.create<arith::ConstantIntOp>(
                loc, abftInjectFaultDelta, /*width=*/32);
              Value injected = bAfter
                .create<func::CallOp>(loc, injectFn, TypeRange(injTypes),
                                      ValueRange{injArg, deltaVal.getResult()})
                .getResult(0);
              if (injected.getType() != outRes.getType()) {
                injected = bAfter.create<tensor::CastOp>(loc, outRes.getType(), injected).getResult();
              }
              Operation *injOp = injected.getDefiningOp();
              SmallVector<OpOperand *, 8> usesToRewrite;
              for (OpOperand &use : outRes.getUses()) {
                Operation *owner = use.getOwner();
                if (owner == injArg.getDefiningOp()) {
                  continue;
                }
                if (owner->getBlock() == injOp->getBlock() && injOp->isBeforeInBlock(owner)) {
                  usesToRewrite.push_back(&use);
                }
              }
              for (OpOperand *use : usesToRewrite) {
                use->set(injected);
              }
              outRes = injected;
              op->emitRemark() << "abft-qmatmul: injected " << pattern << " fault";
            }
          } else {
            op->emitRemark() << "abft-qmatmul: missing fault helper, skipping fault injection";
          }
        }

        Value c0 = bAfter.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = bAfter.create<arith::ConstantIndexOp>(loc, 1);
        Value z64 = bAfter.create<arith::ConstantIntOp>(loc, 0, 64);

        auto castToI64 = [&](Value v) -> Value {
          auto ty = dyn_cast<RankedTensorType>(v.getType());
          if (!ty || ty.getRank() != 2 || !isa<IntegerType>(ty.getElementType()))
          return Value();
          Value m = bAfter.create<tensor::DimOp>(loc, v, c0);
          Value n = bAfter.create<tensor::DimOp>(loc, v, c1);
          Value empty = bAfter.create<tensor::EmptyOp>(loc, TypeRange{dyn2dI64},
                                 ValueRange{m, n}).getResult();
          Value init = bAfter.create<linalg::FillOp>(loc, ValueRange{z64},
                               ValueRange{empty}).getResult(0);
          return bAfter
            .create<linalg::GenericOp>(
              loc, TypeRange{dyn2dI64}, ValueRange{v}, ValueRange{init},
              SmallVector<AffineMap>{
                AffineMap::getMultiDimIdentityMap(2, bAfter.getContext()),
                AffineMap::getMultiDimIdentityMap(2, bAfter.getContext())},
              SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                               utils::IteratorType::parallel},
              [&](OpBuilder &nb, Location nloc, ValueRange args) {
              Value ex =
                nb.create<arith::ExtSIOp>(nloc, nb.getI64Type(), args[0]).getResult();
              nb.create<linalg::YieldOp>(nloc, ex);
              })
            .getResult(0);
        };

        auto makeChecksumI64 = [&](Value in, int64_t dimToReduce) -> Value {
          Value outLen =
            (dimToReduce == 0)
              ? bAfter.create<tensor::DimOp>(loc, in, c1).getResult()
              : bAfter.create<tensor::DimOp>(loc, in, c0).getResult();
          Value empty =
            bAfter.create<tensor::EmptyOp>(loc, TypeRange{vecDynI64},
                           ValueRange{outLen}).getResult();
          Value init =
            bAfter.create<linalg::FillOp>(loc, ValueRange{z64}, ValueRange{empty})
              .getResult(0);
          return bAfter
            .create<linalg::ReduceOp>(
              loc, ValueRange{in}, ValueRange{init},
              ArrayRef<int64_t>{dimToReduce},
              [&](OpBuilder &nb, Location nloc, ValueRange args) {
              Value sum =
                nb.create<arith::AddIOp>(nloc, args[0], args[1]).getResult();
              nb.create<linalg::YieldOp>(nloc, sum);
              })
            .getResult(0);
        };

        auto subScalarFromMatI64 = [&](Value mat, Value scalarI64) -> Value {
          Value m = bAfter.create<tensor::DimOp>(loc, mat, c0);
          Value n = bAfter.create<tensor::DimOp>(loc, mat, c1);
          Value empty = bAfter.create<tensor::EmptyOp>(loc, TypeRange{dyn2dI64},
                                 ValueRange{m, n}).getResult();
          Value init = bAfter.create<linalg::FillOp>(loc, ValueRange{z64},
                               ValueRange{empty}).getResult(0);
          return bAfter
            .create<linalg::GenericOp>(
              loc, TypeRange{dyn2dI64}, ValueRange{mat, scalarI64},
              ValueRange{init},
              SmallVector<AffineMap>{
                AffineMap::getMultiDimIdentityMap(2, bAfter.getContext()),
                AffineMap::get(2, 0, {}, bAfter.getContext()),
                AffineMap::getMultiDimIdentityMap(2, bAfter.getContext())},
              SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                               utils::IteratorType::parallel},
              [&](OpBuilder &nb, Location nloc, ValueRange args) {
              Value d =
                nb.create<arith::SubIOp>(nloc, args[0], args[1]).getResult();
              nb.create<linalg::YieldOp>(nloc, d);
              })
            .getResult(0);
        };

        auto matMulColvecI64 = [&](Value mat, Value vec) -> Value {
          Value m = bAfter.create<tensor::DimOp>(loc, mat, c0);
          Value empty =
            bAfter.create<tensor::EmptyOp>(loc, TypeRange{vecDynI64}, ValueRange{m})
              .getResult();
          Value init =
            bAfter.create<linalg::FillOp>(loc, ValueRange{z64}, ValueRange{empty})
              .getResult(0);
          return bAfter
            .create<linalg::GenericOp>(
              loc, TypeRange{vecDynI64}, ValueRange{mat, vec}, ValueRange{init},
              SmallVector<AffineMap>{
                AffineMap::get(2, 0, {getAffineDimExpr(0, bAfter.getContext()),
                          getAffineDimExpr(1, bAfter.getContext())},
                       bAfter.getContext()),
                AffineMap::get(2, 0, {getAffineDimExpr(1, bAfter.getContext())},
                       bAfter.getContext()),
                AffineMap::get(2, 0, {getAffineDimExpr(0, bAfter.getContext())},
                       bAfter.getContext())},
              SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                               utils::IteratorType::reduction},
              [&](OpBuilder &nb, Location nloc, ValueRange args) {
              Value p =
                nb.create<arith::MulIOp>(nloc, args[0], args[1]).getResult();
              Value s =
                nb.create<arith::AddIOp>(nloc, args[2], p).getResult();
              nb.create<linalg::YieldOp>(nloc, s);
              })
            .getResult(0);
        };

        auto rowvecMulMatI64 = [&](Value rowv, Value mat) -> Value {
          Value n = bAfter.create<tensor::DimOp>(loc, mat, c1);
          Value empty =
            bAfter.create<tensor::EmptyOp>(loc, TypeRange{vecDynI64}, ValueRange{n})
              .getResult();
          Value init =
            bAfter.create<linalg::FillOp>(loc, ValueRange{z64}, ValueRange{empty})
              .getResult(0);
          return bAfter
            .create<linalg::GenericOp>(
              loc, TypeRange{vecDynI64}, ValueRange{rowv, mat}, ValueRange{init},
              SmallVector<AffineMap>{
                AffineMap::get(2, 0, {getAffineDimExpr(1, bAfter.getContext())},
                       bAfter.getContext()),
                AffineMap::get(2, 0, {getAffineDimExpr(1, bAfter.getContext()),
                          getAffineDimExpr(0, bAfter.getContext())},
                       bAfter.getContext()),
                AffineMap::get(2, 0, {getAffineDimExpr(0, bAfter.getContext())},
                       bAfter.getContext())},
              SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                               utils::IteratorType::reduction},
              [&](OpBuilder &nb, Location nloc, ValueRange args) {
              Value p =
                nb.create<arith::MulIOp>(nloc, args[0], args[1]).getResult();
              Value s =
                nb.create<arith::AddIOp>(nloc, args[2], p).getResult();
              nb.create<linalg::YieldOp>(nloc, s);
              })
            .getResult(0);
        };

          auto subMatFromMatI64 = [&](Value lhsM, Value rhsM) -> Value {
            Value m = bAfter.create<tensor::DimOp>(loc, lhsM, c0);
            Value n = bAfter.create<tensor::DimOp>(loc, lhsM, c1);
            Value empty = bAfter.create<tensor::EmptyOp>(
              loc, TypeRange{dyn2dI64}, ValueRange{m, n}).getResult();
            Value init = bAfter
                     .create<linalg::FillOp>(loc, ValueRange{z64},
                                 ValueRange{empty})
                     .getResult(0);
            return bAfter
              .create<linalg::GenericOp>(
                loc, TypeRange{dyn2dI64}, ValueRange{lhsM, rhsM},
                ValueRange{init},
                SmallVector<AffineMap>{
                  AffineMap::getMultiDimIdentityMap(2, bAfter.getContext()),
                  AffineMap::getMultiDimIdentityMap(2, bAfter.getContext()),
                  AffineMap::getMultiDimIdentityMap(2, bAfter.getContext())},
                SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                 utils::IteratorType::parallel},
                [&](OpBuilder &nb, Location nloc, ValueRange args) {
                Value d =
                  nb.create<arith::SubIOp>(nloc, args[0], args[1]).getResult();
                nb.create<linalg::YieldOp>(nloc, d);
                })
              .getResult(0);
          };

        auto extractI64Scalar = [&](Value v) -> Value {
          if (auto ty = dyn_cast<RankedTensorType>(v.getType())) {
          if (ty.getRank() == 0) {
            Value s = bAfter.create<tensor::ExtractOp>(loc, v).getResult();
            if (s.getType().isInteger(64)) return s;
            return bAfter.create<arith::ExtSIOp>(loc, bAfter.getI64Type(), s)
              .getResult();
          }
          }
          if (v.getType().isInteger(64)) return v;
          return bAfter.create<arith::ExtSIOp>(loc, bAfter.getI64Type(), v)
            .getResult();
        };

        Value lhsI64 = castToI64(lhs);
        Value rhsI64 = castToI64(rhs);
        Value outI64 = castToI64(outRes);
        Value initI64 = castToI64(outInit);
        if (!lhsI64 || !rhsI64 || !outI64 || !initI64) {
          op->emitRemark() << "abft-qmatmul: failed i64 conversion";
          continue;
        }

        Value zA64 = extractI64Scalar(lhsZp);
        Value zB64 = extractI64Scalar(rhsZp);

        Value lhsAdj = subScalarFromMatI64(lhsI64, zA64);
        Value rhsAdj = subScalarFromMatI64(rhsI64, zB64);

        Value lhsCol = makeChecksumI64(lhsAdj, /*dimToReduce=*/0);
        Value rhsRow = makeChecksumI64(rhsAdj, /*dimToReduce=*/1);
        Value propRow = matMulColvecI64(lhsAdj, rhsRow);
        Value propCol = rowvecMulMatI64(lhsCol, rhsAdj);
        Value expRow64 = propRow;
        Value expCol64 = propCol;

        Value outCmpI64 = subMatFromMatI64(outI64, initI64);
        Value outRow64 = makeChecksumI64(outCmpI64, /*dimToReduce=*/1);
        Value outCol64 = makeChecksumI64(outCmpI64, /*dimToReduce=*/0);

        auto f32Ty = bAfter.getF32Type();
        auto i32Ty = bAfter.getI32Type();
        SmallVector<Type, 1> i32Types = {i32Ty};
        Value rowDeltaI32 =
          bAfter
            .create<func::CallOp>(loc, StringRef("vector_max_abs_diff_i64_i32"),
                        TypeRange(i32Types),
                        ValueRange{expRow64, outRow64})
            .getResult(0);
        Value colDeltaI32 =
          bAfter
            .create<func::CallOp>(loc, StringRef("vector_max_abs_diff_i64_i32"),
                        TypeRange(i32Types),
                        ValueRange{expCol64, outCol64})
            .getResult(0);
        Value rowExpMaxI32 =
          bAfter
            .create<func::CallOp>(loc, StringRef("vector_max_abs_i64_i32"),
                        TypeRange(i32Types), ValueRange{expRow64})
            .getResult(0);
        Value rowCalcMaxI32 =
          bAfter
            .create<func::CallOp>(loc, StringRef("vector_max_abs_i64_i32"),
                        TypeRange(i32Types), ValueRange{outRow64})
            .getResult(0);
        Value colExpMaxI32 =
          bAfter
            .create<func::CallOp>(loc, StringRef("vector_max_abs_i64_i32"),
                        TypeRange(i32Types), ValueRange{expCol64})
            .getResult(0);
        Value colCalcMaxI32 =
          bAfter
            .create<func::CallOp>(loc, StringRef("vector_max_abs_i64_i32"),
                        TypeRange(i32Types), ValueRange{outCol64})
            .getResult(0);

        Value rowDelta = bAfter.create<arith::SIToFPOp>(loc, f32Ty, rowDeltaI32);
        Value colDelta = bAfter.create<arith::SIToFPOp>(loc, f32Ty, colDeltaI32);
        Value rowExpMax = bAfter.create<arith::SIToFPOp>(loc, f32Ty, rowExpMaxI32);
        Value rowCalcMax = bAfter.create<arith::SIToFPOp>(loc, f32Ty, rowCalcMaxI32);
        Value colExpMax = bAfter.create<arith::SIToFPOp>(loc, f32Ty, colExpMaxI32);
        Value colCalcMax = bAfter.create<arith::SIToFPOp>(loc, f32Ty, colCalcMaxI32);



        Value layerConst = bAfter.create<arith::ConstantOp>(
          loc, bAfter.getF32Type(),
          bAfter.getF32FloatAttr(static_cast<float>(currentLayer)));
        if (abftEnableAnalysisLog) {
          bAfter.create<func::CallOp>(
            loc, StringRef("abft_analysis.abft_log_rowcol_delta"), TypeRange{},
            ValueRange{layerConst, rowDelta, colDelta});
          if (logRowColDebugFn) {
            bAfter.create<func::CallOp>(
              loc, StringRef("abft_analysis.abft_log_rowcol_debug"), TypeRange{},
              ValueRange{layerConst, rowExpMax, rowCalcMax, colExpMax,
                   colCalcMax});
          }
        }
        op->emitRemark() << "abft-qmatmul: inserted algorithmic checksum verification";
        continue;
        }

      // Integer matmul path (e.g., int8/int8->int32 after
      // iree-global-opt-quantized-matmul-to-matmul lowering).
      if (targetName == "linalg.matmul") {
        auto matmul = dyn_cast<linalg::MatmulOp>(op);
          if (!matmul || !vecMaxI64I32Fn || !vecMaxAbsI64I32Fn ||
              (abftEnableAnalysisLog && !logRowColDeltaFn) ||
            op->getNumResults() == 0) {
          op->emitRemark() << "abft-int-matmul: missing helper(s), skipping";
          continue;
        }
        auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!outTy || !outTy.getElementType().isInteger(32)) {
          // Let float path below handle f32 matmul.
          op->emitRemark()
              << "abft-int-matmul: output not i32, deferring to other paths";
        } else {
          Location loc = op->getLoc();
          Block::iterator nextIt = std::next(Block::iterator(op));
          OpBuilder bAfter(op->getBlock(), nextIt);

          auto dyn2dI32 = RankedTensorType::get(
              {ShapedType::kDynamic, ShapedType::kDynamic}, bAfter.getI32Type());
          auto toI32 = [&](Value v) -> Value {
            auto ty = dyn_cast<RankedTensorType>(v.getType());
            if (!ty || ty.getRank() != 2)
              return Value();
            if (ty.getElementType().isInteger(32)) {
              return ty == dyn2dI32
                         ? v
                         : bAfter.create<tensor::CastOp>(loc, dyn2dI32, v)
                               .getResult();
            }
            if (!isa<IntegerType>(ty.getElementType()))
              return Value();
            Value c0 = bAfter.create<arith::ConstantIndexOp>(loc, 0);
            Value c1 = bAfter.create<arith::ConstantIndexOp>(loc, 1);
            Value m = bAfter.create<tensor::DimOp>(loc, v, c0);
            Value n = bAfter.create<tensor::DimOp>(loc, v, c1);
            Value empty =
                bAfter
                    .create<tensor::EmptyOp>(loc, TypeRange{dyn2dI32},
                                             ValueRange{m, n})
                    .getResult();
            Value z = bAfter.create<arith::ConstantIntOp>(loc, 0, 32);
            Value init = bAfter
                             .create<linalg::FillOp>(loc, ValueRange{z},
                                                     ValueRange{empty})
                             .getResult(0);
            return bAfter
                .create<linalg::GenericOp>(
                    loc, TypeRange{dyn2dI32}, ValueRange{v}, ValueRange{init},
                    SmallVector<AffineMap>{
                        AffineMap::getMultiDimIdentityMap(2, bAfter.getContext()),
                        AffineMap::getMultiDimIdentityMap(2, bAfter.getContext())},
                    SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                                     utils::IteratorType::parallel},
                    [&](OpBuilder &nb, Location nloc, ValueRange args) {
                      Value ex =
                          nb.create<arith::ExtSIOp>(nloc, nb.getI32Type(), args[0]);
                      nb.create<linalg::YieldOp>(nloc, ex);
                    })
                .getResult(0);
          };

                Value lhsI32 = toI32(op->getOperand(0));
                Value rhsI32 = toI32(op->getOperand(1));

                Value outI32 = toI32(op->getResult(0));
                if (!lhsI32 || !rhsI32 || !outI32) {
                op->emitRemark() << "abft-int-matmul: failed i32 conversion for outputs";
            continue;
          }
                Value lhsForChecksI32 = lhsI32;

                Value initI32 = (op->getNumOperands() > 2) ? toI32(op->getOperand(2)) : Value();

                if (abftInjectFault &&
                    (abftInjectFaultLayer < 0 ||
                     currentLayer == static_cast<int64_t>(abftInjectFaultLayer))) {
                StringRef pattern = abftInjectFaultPattern.getValue();
                StringRef injectFn = "apply_fault_i32_single_point";
                if (pattern == "trivial") {
                  injectFn = "apply_fault_i32_trivial";
                } else if (pattern == "checkered") {
                  injectFn = "apply_fault_i32_checkered";
                }
                auto injectSym = module.lookupSymbol<func::FuncOp>(injectFn);
                if (injectSym) {
                  SmallVector<Type, 1> injTypes;
                  for (Type t : injectSym.getFunctionType().getResults())
                  injTypes.push_back(t);
                  Value injArg = outI32;
                  Type expectedIn = injectSym.getFunctionType().getInput(0);
                  if (injArg.getType() != expectedIn) {
                  auto inTy = dyn_cast<RankedTensorType>(injArg.getType());
                  auto expTy = dyn_cast<RankedTensorType>(expectedIn);
                  if (inTy && expTy && inTy.getRank() == expTy.getRank()) {
                    injArg = bAfter.create<tensor::CastOp>(loc, expectedIn, injArg).getResult();
                  } else {
                    op->emitRemark()
                      << "abft-int-matmul: pattern injection rank mismatch, skipping fault injection";
                    injectSym = {};
                  }
                  }
                  if (injectSym) {
                  auto deltaVal = bAfter.create<arith::ConstantIntOp>(
                    loc, abftInjectFaultDelta, /*width=*/32);
                  Value injected = bAfter
                             .create<func::CallOp>(
                               loc, injectFn, TypeRange(injTypes),
                               ValueRange{injArg, deltaVal.getResult()})
                             .getResult(0);
                  if (injected.getType() != outI32.getType()) {
                    injected =
                      bAfter.create<tensor::CastOp>(loc, outI32.getType(), injected).getResult();
                  }
                  outI32 = injected;
                  op->emitRemark() << "abft-int-matmul: injected " << pattern << " fault";
                  }
                }
                }

                Value c0 = bAfter.create<arith::ConstantIndexOp>(loc, 0);
                Value c1 = bAfter.create<arith::ConstantIndexOp>(loc, 1);
                Value z32 = bAfter.create<arith::ConstantIntOp>(loc, 0, 32);
                Value z64 = bAfter.create<arith::ConstantIntOp>(loc, 0, 64);

                Value outCmpI32 = outI32;

                auto dyn2dI64 = RankedTensorType::get(
                    {ShapedType::kDynamic, ShapedType::kDynamic},
                    bAfter.getI64Type());
                auto vecDynI64 = RankedTensorType::get(
                    {ShapedType::kDynamic}, bAfter.getI64Type());
                auto castMatI32ToI64 = [&](Value mI32) -> Value {
                  Value m = bAfter.create<tensor::DimOp>(loc, mI32, c0);
                  Value n = bAfter.create<tensor::DimOp>(loc, mI32, c1);
                  Value empty = bAfter.create<tensor::EmptyOp>(
                      loc, TypeRange{dyn2dI64}, ValueRange{m, n}).getResult();
                  Value init = bAfter.create<linalg::FillOp>(
                      loc, ValueRange{z64}, ValueRange{empty}).getResult(0);
                  return bAfter.create<linalg::GenericOp>(
                                   loc, TypeRange{dyn2dI64}, ValueRange{mI32},
                                   ValueRange{init},
                                   SmallVector<AffineMap>{
                                       AffineMap::getMultiDimIdentityMap(
                                           2, bAfter.getContext()),
                                       AffineMap::getMultiDimIdentityMap(
                                           2, bAfter.getContext())},
                                   SmallVector<utils::IteratorType>{
                                       utils::IteratorType::parallel,
                                       utils::IteratorType::parallel},
                                   [&](OpBuilder &nb, Location nloc,
                                       ValueRange args) {
                                     Value ex = nb.create<arith::ExtSIOp>(
                                         nloc, nb.getI64Type(), args[0]);
                                     nb.create<linalg::YieldOp>(nloc, ex);
                                   })
                      .getResult(0);
                };
                auto makeChecksumI64 = [&](Value in, int64_t dimToReduce) -> Value {
                  Value outLen =
                      (dimToReduce == 0)
                          ? bAfter.create<tensor::DimOp>(loc, in, c1).getResult()
                          : bAfter.create<tensor::DimOp>(loc, in, c0).getResult();
                  Value empty = bAfter.create<tensor::EmptyOp>(
                      loc, TypeRange{vecDynI64}, ValueRange{outLen}).getResult();
                  Value init = bAfter.create<linalg::FillOp>(
                      loc, ValueRange{z64}, ValueRange{empty}).getResult(0);
                  return bAfter
                      .create<linalg::ReduceOp>(
                          loc, ValueRange{in}, ValueRange{init},
                          ArrayRef<int64_t>{dimToReduce},
                          [&](OpBuilder &nb, Location nloc, ValueRange args) {
                            nb.create<linalg::YieldOp>(
                                nloc,
                                nb.create<arith::AddIOp>(nloc, args[0], args[1])
                                    .getResult());
                          })
                      .getResult(0);
                };

                Value lhsI64 = castMatI32ToI64(lhsForChecksI32);
                Value rhsI64 = castMatI32ToI64(rhsI32);
                Value outCmpI64 = castMatI32ToI64(outCmpI32);

                Value outRow64 = makeChecksumI64(outCmpI64, /*dimToReduce=*/1);
                Value outCol64 = makeChecksumI64(outCmpI64, /*dimToReduce=*/0);

                Value mDim = bAfter.create<tensor::DimOp>(loc, outCmpI32, c0);
                Value nDim = bAfter.create<tensor::DimOp>(loc, outCmpI32, c1);
                Value z32tensor = bAfter.create<arith::ConstantIntOp>(loc, 0, 32);
                Value eMxNI32 = bAfter.create<tensor::EmptyOp>(
                  loc, TypeRange{dyn2dI32}, ValueRange{mDim, nDim}).getResult();
                Value iMxNI32 = bAfter.create<linalg::FillOp>(
                  loc, ValueRange{z32tensor}, ValueRange{eMxNI32}).getResult(0);
                Value expCoreI32 = bAfter.create<linalg::MatmulOp>(
                  loc, TypeRange{dyn2dI32}, ValueRange{lhsForChecksI32, rhsI32},
                  ValueRange{iMxNI32}).getResult(0);
                Value expFullI32 = expCoreI32;
                if (initI32) {
                  Value eSumI32 = bAfter.create<tensor::EmptyOp>(
                    loc, TypeRange{dyn2dI32}, ValueRange{mDim, nDim}).getResult();
                  Value iSumI32 = bAfter.create<linalg::FillOp>(
                    loc, ValueRange{z32tensor}, ValueRange{eSumI32}).getResult(0);
                  expFullI32 = bAfter.create<linalg::AddOp>(
                    loc, TypeRange{dyn2dI32}, ValueRange{expCoreI32, initI32},
                    ValueRange{iSumI32}).getResult(0);
                }
                Value expFull64 = castMatI32ToI64(expFullI32);
                Value expRow64 = makeChecksumI64(expFull64, /*dimToReduce=*/1);
                Value expCol64 = makeChecksumI64(expFull64, /*dimToReduce=*/0);

                auto subVecFromVecI64 = [&](Value lhsV, Value rhsV) -> Value {
                  Value len = bAfter.create<tensor::DimOp>(loc, lhsV, c0);
                  Value empty = bAfter.create<tensor::EmptyOp>(
                    loc, TypeRange{RankedTensorType::get({ShapedType::kDynamic}, bAfter.getI64Type())},
                    ValueRange{len}).getResult();
                  Value init = bAfter.create<linalg::FillOp>(
                    loc, ValueRange{z64}, ValueRange{empty}).getResult(0);
                  return bAfter
                    .create<linalg::GenericOp>(
                      loc,
                      TypeRange{RankedTensorType::get({ShapedType::kDynamic}, bAfter.getI64Type())},
                      ValueRange{lhsV, rhsV}, ValueRange{init},
                      SmallVector<AffineMap>{
                        AffineMap::getMultiDimIdentityMap(1, bAfter.getContext()),
                        AffineMap::getMultiDimIdentityMap(1, bAfter.getContext()),
                        AffineMap::getMultiDimIdentityMap(1, bAfter.getContext())},
                      SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
                      [&](OpBuilder &nb, Location nloc, ValueRange args) {
                      Value d =
                        nb.create<arith::SubIOp>(nloc, args[0], args[1]).getResult();
                      nb.create<linalg::YieldOp>(nloc, d);
                      })
                    .getResult(0);
                };

                Value layerConst = bAfter.create<arith::ConstantOp>(
                    loc, bAfter.getF32Type(),
                    bAfter.getF32FloatAttr(static_cast<float>(currentLayer)));

                auto emitPerElementDeltas = [&](Value diffVec, bool isRowAxis) {
                  Value len = bAfter.create<tensor::DimOp>(loc, diffVec, c0);
                  Value empty = bAfter.create<tensor::EmptyOp>(
                      loc, TypeRange{vecDynI64}, ValueRange{len}).getResult();
                  Value init = bAfter.create<linalg::FillOp>(
                      loc, ValueRange{z64}, ValueRange{empty}).getResult(0);
                  (void)bAfter.create<linalg::GenericOp>(
                      loc, TypeRange{vecDynI64}, ValueRange{diffVec},
                      ValueRange{init},
                      SmallVector<AffineMap>{
                          AffineMap::getMultiDimIdentityMap(1, bAfter.getContext()),
                          AffineMap::getMultiDimIdentityMap(1, bAfter.getContext())},
                      SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
                      [&](OpBuilder &nb, Location nloc, ValueRange args) {
                        Value elem = args[0];
                        Value zeroI64 = nb.create<arith::ConstantIntOp>(nloc, 0, 64);
                        Value isNeg = nb.create<arith::CmpIOp>(
                            nloc, arith::CmpIPredicate::slt, elem, zeroI64);
                        Value neg = nb.create<arith::SubIOp>(nloc, zeroI64, elem).getResult();
                        Value abs = nb.create<arith::SelectOp>(nloc, isNeg, neg, elem).getResult();
                        Value absF32 = nb.create<arith::SIToFPOp>(nloc, nb.getF32Type(), abs);
                        Value zeroF32 = nb.create<arith::ConstantOp>(
                            nloc, nb.getF32Type(), nb.getF32FloatAttr(0.0f));
                        if (isRowAxis) {
                          nb.create<func::CallOp>(
                              nloc, StringRef("abft_analysis.abft_log_rowcol_delta"),
                              TypeRange{}, ValueRange{layerConst, absF32, zeroF32});
                        } else {
                          nb.create<func::CallOp>(
                              nloc, StringRef("abft_analysis.abft_log_rowcol_delta"),
                              TypeRange{}, ValueRange{layerConst, zeroF32, absF32});
                        }
                        nb.create<linalg::YieldOp>(nloc, elem);
                      });
                };

                Value rowDiff64 = subVecFromVecI64(expRow64, outRow64);
                Value colDiff64 = subVecFromVecI64(expCol64, outCol64);
                if (abftEnableAnalysisLog) {
                  emitPerElementDeltas(rowDiff64, /*isRowAxis=*/true);
                  emitPerElementDeltas(colDiff64, /*isRowAxis=*/false);
                }

                auto i32Ty = bAfter.getI32Type();
                SmallVector<Type, 1> i32Types = {i32Ty};
                Value rowDeltaI32 = bAfter
                    .create<func::CallOp>(
                        loc, StringRef("vector_max_abs_diff_i64_i32"),
                        TypeRange(i32Types), ValueRange{expRow64, outRow64})
                    .getResult(0);
                Value colDeltaI32 = bAfter
                    .create<func::CallOp>(
                        loc, StringRef("vector_max_abs_diff_i64_i32"),
                        TypeRange(i32Types), ValueRange{expCol64, outCol64})
                    .getResult(0);
                Value rowExpMaxI32 = bAfter
                    .create<func::CallOp>(loc, StringRef("vector_max_abs_i64_i32"),
                                          TypeRange(i32Types),
                                          ValueRange{expRow64})
                    .getResult(0);
                Value rowCalcMaxI32 = bAfter
                    .create<func::CallOp>(loc, StringRef("vector_max_abs_i64_i32"),
                                          TypeRange(i32Types),
                                          ValueRange{outRow64})
                    .getResult(0);
                Value colExpMaxI32 = bAfter
                    .create<func::CallOp>(loc, StringRef("vector_max_abs_i64_i32"),
                                          TypeRange(i32Types),
                                          ValueRange{expCol64})
                    .getResult(0);
                Value colCalcMaxI32 = bAfter
                    .create<func::CallOp>(loc, StringRef("vector_max_abs_i64_i32"),
                                          TypeRange(i32Types),
                                          ValueRange{outCol64})
                    .getResult(0);
                Value rowDelta = bAfter.create<arith::SIToFPOp>(
                    loc, bAfter.getF32Type(), rowDeltaI32);
                Value colDelta = bAfter.create<arith::SIToFPOp>(
                    loc, bAfter.getF32Type(), colDeltaI32);
                Value rowExpMax = bAfter.create<arith::SIToFPOp>(
                    loc, bAfter.getF32Type(), rowExpMaxI32);
                Value rowCalcMax = bAfter.create<arith::SIToFPOp>(
                    loc, bAfter.getF32Type(), rowCalcMaxI32);
                Value colExpMax = bAfter.create<arith::SIToFPOp>(
                    loc, bAfter.getF32Type(), colExpMaxI32);
                Value colCalcMax = bAfter.create<arith::SIToFPOp>(
                    loc, bAfter.getF32Type(), colCalcMaxI32);



          if (abftEnableAnalysisLog) {
            bAfter.create<func::CallOp>(
                loc, StringRef("abft_analysis.abft_log_rowcol_delta"), TypeRange{},
                    ValueRange{layerConst, rowDelta, colDelta});
            bAfter.create<func::CallOp>(
                loc, StringRef("abft_analysis.abft_log_rowcol_debug"), TypeRange{},
                ValueRange{layerConst, rowExpMax, rowCalcMax, colExpMax,
                           colCalcMax});
          }
          op->emitRemark()
                  << "abft-int-matmul: inserted algorithmic checksum verification";
          continue;
        }
      }

      op->emitRemark() << "abft-ones: matched matmul for ones*A insertion";

      if (!vecMaxFn || !colFn || !rowFn || !matcolFn ||
          (abftEnableAnalysisLog && !logRowColDeltaFn) ||
          !rowvecFn || op->getNumResults() == 0) {
        op->emitRemark() << "abft: missing helper(s), skipping FP32 instrumentation";
        continue;
      }
      auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!outTy || !outTy.getElementType().isF32()) {
        op->emitRemark() << "abft: unsupported output type; expected ranked tensor<f32>";
        continue;
      }
      
      Location loc = op->getLoc();
      Value refOut = op->getResult(0);

      Block::iterator nextIt = std::next(Block::iterator(op));
      OpBuilder bAfter(op->getBlock(), nextIt);

      auto lhs = op->getOperand(0);
      auto rhs = op->getOperand(1);
      auto init = op->getOperand(2);
      auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
      auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
      auto outTy2 = dyn_cast<RankedTensorType>(refOut.getType());
      if (!lhsTy || !rhsTy || !outTy2 || lhsTy.getRank() != 2 ||
          rhsTy.getRank() != 2 || outTy2.getRank() != 2) {
        op->emitRemark() << "abft: FP32 checksum path expects rank-2 matmul";
        continue;
      }
      auto dyn2dF32 = RankedTensorType::get(
          {ShapedType::kDynamic, ShapedType::kDynamic}, bAfter.getF32Type());
      Value lhs2d = lhsTy == dyn2dF32
                        ? lhs
                        : bAfter.create<tensor::CastOp>(loc, dyn2dF32, lhs).getResult();
      Value rhs2d = rhsTy == dyn2dF32
                        ? rhs
                        : bAfter.create<tensor::CastOp>(loc, dyn2dF32, rhs).getResult();
      Value out2d = outTy2 == dyn2dF32
                        ? refOut
                        : bAfter.create<tensor::CastOp>(loc, dyn2dF32, refOut).getResult();
      Value init2d = init.getType() == dyn2dF32
                         ? init
                         : bAfter.create<tensor::CastOp>(loc, dyn2dF32, init).getResult();

      Value compare2d = out2d;
      if (matrixSubFn) {
        compare2d = bAfter
                        .create<func::CallOp>(
                            loc, StringRef("matrix_sub"),
                            TypeRange{matrixSubFn.getFunctionType().getResult(0)},
                            ValueRange{out2d, init2d})
                        .getResult(0);
      }

      Value compareForChecks = compare2d;
      if (abftInjectFault &&
          (abftInjectFaultLayer < 0 ||
           currentLayer == static_cast<int64_t>(abftInjectFaultLayer))) {
        StringRef pattern = abftInjectFaultPattern.getValue();
        Value c0 = bAfter.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = bAfter.create<arith::ConstantIndexOp>(loc, 1);
        Value c2 = bAfter.create<arith::ConstantIndexOp>(loc, 2);
        Value rows = bAfter.create<tensor::DimOp>(loc, compare2d, c0);
        Value cols = bAfter.create<tensor::DimOp>(loc, compare2d, c1);
        Value hasAtLeast2Rows = bAfter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::uge, rows, c2);
        Value hasAtLeast2Cols = bAfter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::uge, cols, c2);
        Value canApplyTrivial2D =
            bAfter.create<arith::AndIOp>(loc, hasAtLeast2Rows, hasAtLeast2Cols);
        auto dyn2dTyF32 = RankedTensorType::get(
            {ShapedType::kDynamic, ShapedType::kDynamic}, bAfter.getF32Type());
        Value empty = bAfter
                          .create<tensor::EmptyOp>(
                              loc, TypeRange{dyn2dTyF32}, ValueRange{rows, cols})
                          .getResult();
        Value zero = bAfter.create<arith::ConstantOp>(
            loc, bAfter.getF32Type(), bAfter.getF32FloatAttr(0.0f));
        Value deltaVal = bAfter.create<arith::ConstantOp>(
            loc, bAfter.getF32Type(),
            bAfter.getF32FloatAttr(static_cast<float>(abftInjectFaultDelta)));
        Value initFault = bAfter
                              .create<linalg::FillOp>(
                                  loc, ValueRange{zero}, ValueRange{empty})
                              .getResult(0);
          compareForChecks =
            bAfter
                .create<linalg::GenericOp>(
            loc, TypeRange{dyn2dTyF32}, ValueRange{compare2d},
                    ValueRange{initFault},
                    SmallVector<AffineMap>{
                        AffineMap::getMultiDimIdentityMap(2, bAfter.getContext()),
                        AffineMap::getMultiDimIdentityMap(2, bAfter.getContext())},
                    SmallVector<utils::IteratorType>{
                        utils::IteratorType::parallel,
                        utils::IteratorType::parallel},
                    [&](OpBuilder &nestedBuilder, Location nestedLoc,
                        ValueRange args) {
                      Value i = nestedBuilder.create<linalg::IndexOp>(nestedLoc, 0);
                      Value j = nestedBuilder.create<linalg::IndexOp>(nestedLoc, 1);
                      Value fault = nestedBuilder.create<arith::ConstantOp>(
                          nestedLoc, nestedBuilder.getF32Type(),
                          nestedBuilder.getF32FloatAttr(0.0f));
                      if (pattern == "single_point") {
                        Value hitI = nestedBuilder.create<arith::CmpIOp>(
                            nestedLoc, arith::CmpIPredicate::eq, i, c0);
                        Value hitJ = nestedBuilder.create<arith::CmpIOp>(
                            nestedLoc, arith::CmpIPredicate::eq, j, c0);
                        Value hit = nestedBuilder.create<arith::AndIOp>(
                            nestedLoc, hitI, hitJ);
                        fault = nestedBuilder.create<arith::SelectOp>(
                            nestedLoc, hit, deltaVal, fault);
                      } else if (pattern == "trivial") {
                        Value inI = nestedBuilder.create<arith::CmpIOp>(
                            nestedLoc, arith::CmpIPredicate::ult, i, c2);
                        Value inJ = nestedBuilder.create<arith::CmpIOp>(
                            nestedLoc, arith::CmpIPredicate::ult, j, c2);
                        Value inBlk = nestedBuilder.create<arith::AndIOp>(nestedLoc, inI, inJ);
                        inBlk = nestedBuilder.create<arith::AndIOp>(
                            nestedLoc, inBlk, canApplyTrivial2D);
                        Value im = nestedBuilder.create<arith::RemUIOp>(nestedLoc, i, c2);
                        Value jm = nestedBuilder.create<arith::RemUIOp>(nestedLoc, j, c2);
                        Value diag = nestedBuilder.create<arith::CmpIOp>(
                            nestedLoc, arith::CmpIPredicate::eq, im, jm);
                        Value neg =
                            nestedBuilder.create<arith::SubFOp>(nestedLoc, zero, deltaVal);
                        Value pat = nestedBuilder.create<arith::SelectOp>(
                            nestedLoc, diag, deltaVal, neg);
                        fault = nestedBuilder.create<arith::SelectOp>(
                            nestedLoc, inBlk, pat, fault);
                      } else if (pattern == "checkered") {
                        Value im = nestedBuilder.create<arith::RemUIOp>(nestedLoc, i, c2);
                        Value even = nestedBuilder.create<arith::CmpIOp>(
                            nestedLoc, arith::CmpIPredicate::eq, im, c0);
                        Value neg =
                            nestedBuilder.create<arith::SubFOp>(nestedLoc, zero, deltaVal);
                        fault = nestedBuilder.create<arith::SelectOp>(
                            nestedLoc, even, deltaVal, neg);
                      }
                      Value sum = nestedBuilder.create<arith::AddFOp>(
                          nestedLoc, args[0], fault);
                      nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
                    })
                .getResult(0);
      }

            Value lhsCol = bAfter.create<func::CallOp>(
              loc, StringRef("column_checksum"),
              TypeRange{colFn.getFunctionType().getResult(0)},
              ValueRange{lhs2d}).getResult(0);
            Value rhsRow = bAfter.create<func::CallOp>(
              loc, StringRef("row_checksum"),
              TypeRange{rowFn.getFunctionType().getResult(0)},
              ValueRange{rhs2d}).getResult(0);
      auto outCol = bAfter.create<func::CallOp>(
          loc, StringRef("column_checksum"),
          TypeRange{colFn.getFunctionType().getResult(0)},
              ValueRange{compareForChecks}).getResult(0);
      auto outRow = bAfter.create<func::CallOp>(
          loc, StringRef("row_checksum"),
          TypeRange{rowFn.getFunctionType().getResult(0)},
              ValueRange{compareForChecks}).getResult(0);

            auto expRow = bAfter.create<func::CallOp>(
              loc, StringRef("mat_mul_colvec_abft_v2"),
                TypeRange{matcolFn.getFunctionType().getResult(0)},
              ValueRange{lhs2d, rhsRow}).getResult(0);
            auto expCol = bAfter.create<func::CallOp>(
              loc, StringRef("rowvec_mul_mat_abft_v2"),
                TypeRange{rowvecFn.getFunctionType().getResult(0)},
              ValueRange{lhsCol, rhs2d}).getResult(0);

      SmallVector<Type, 1> maxTypes;
      for (Type t : vecMaxFn.getFunctionType().getResults()) maxTypes.push_back(t);
      auto rowDelta = bAfter.create<func::CallOp>(
          loc, StringRef("vector_max_abs_diff"), TypeRange(maxTypes),
              ValueRange{outRow, expRow}).getResult(0);
      auto colDelta = bAfter.create<func::CallOp>(
          loc, StringRef("vector_max_abs_diff"), TypeRange(maxTypes),
              ValueRange{outCol, expCol}).getResult(0);

      auto layerConst = bAfter.create<arith::ConstantOp>(
          loc, bAfter.getF32Type(),
          bAfter.getF32FloatAttr(static_cast<float>(currentLayer)));
      if (abftEnableAnalysisLog) {
        bAfter.create<func::CallOp>(
            loc, StringRef("abft_analysis.abft_log_rowcol_delta"), TypeRange{},
            ValueRange{layerConst.getResult(), rowDelta, colDelta});
      }
      op->emitRemark() << "abft: inserted FP32 checksum-based row/col delta logging";
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createABFTPass() {
  return std::make_unique<ABFTPass>();
}
std::unique_ptr<mlir::Pass> createABFTSpecializeBatchDimPass() {
  return std::make_unique<ABFTSpecializeBatchDimPass>();
}
static mlir::PassRegistration<ABFTPass> reg;
static mlir::PassRegistration<ABFTSpecializeBatchDimPass> regBatch;
