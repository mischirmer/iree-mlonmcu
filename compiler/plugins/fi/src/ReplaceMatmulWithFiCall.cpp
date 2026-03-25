// ReplaceMatmulWithFiCall.cpp
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>

using namespace mlir;

namespace {

// Create (or find) a declaration for:
//   func private @fi.fi_plugin_f32(tensor<?x?xf32>, index, index, index)
//     -> tensor<?x?xf32>
static func::FuncOp ensureFiDecl(ModuleOp module) {
  StringRef name = "fi.fi_plugin_f32";
  if (auto f = module.lookupSymbol<func::FuncOp>(name)) return f;

  MLIRContext *ctx = module.getContext();
  OpBuilder b(module.getBodyRegion());
  auto loc = b.getUnknownLoc();

  auto f32 = b.getF32Type();
  auto idx = b.getIndexType();
  auto dyn2DTensor = RankedTensorType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, f32);
  auto fnType = FunctionType::get(
      ctx, TypeRange{dyn2DTensor, idx, idx, idx}, TypeRange{dyn2DTensor});

  auto fn = b.create<func::FuncOp>(loc, name, fnType);
  fn.setPrivate();
  return fn;
}

static std::set<int64_t> parseEnvIndexSet(const char *name) {
  std::set<int64_t> values;
  const char *env = std::getenv(name);
  if (!env || env[0] == '\0') return values;
  std::string s(env);
  if (!s.empty() && s.front() == '[' && s.back() == ']') {
    s = s.substr(1, s.size() - 2);
  }
  std::stringstream ss(s);
  while (ss.good()) {
    std::string token;
    if (!std::getline(ss, token, ',')) break;
    size_t a = 0;
    while (a < token.size() && isspace((unsigned char)token[a])) ++a;
    size_t b = token.size();
    while (b > a && isspace((unsigned char)token[b - 1])) --b;
    if (b > a) {
      std::string num = token.substr(a, b - a);
      char *endptr = nullptr;
      errno = 0;
      long long v = std::strtoll(num.c_str(), &endptr, 10);
      if (endptr != num.c_str() && *endptr == '\0' && errno == 0) {
        values.insert(static_cast<int64_t>(v));
      }
    }
  }
  return values;
}

struct InsertFiCallPass
    : public PassWrapper<InsertFiCallPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertFiCallPass)

  StringRef getArgument() const final { return "fi-insert-call-after-matmul"; }
  StringRef getDescription() const final {
    return "Insert a call to fi_plugin after linalg.matmul (f32)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,
            func::FuncDialect,
            linalg::LinalgDialect,
            tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    ModuleOp module = func->getParentOfType<ModuleOp>();

    std::set<int64_t> tamperSet = parseEnvIndexSet("IREE_FI_TAMPER");

    SmallVector<linalg::MatmulOp> mats;
    func.walk([&](linalg::MatmulOp mm) { mats.push_back(mm); });

    int inserted = 0;
    int seen = 0;
    for (auto mm : mats) {
      int thisIdx = seen++;
      // Only tamper explicitly selected matmul indices.
      // This avoids accidental all-layer tampering from inherited env.
      bool doTamper = tamperSet.count(thisIdx) > 0;

      // Tensor path: call FI function on the matmul result and replace uses
      // with the returned tensor.
      if (mm->getNumResults() != 1) continue;
      Value C = mm->getResult(0);
      auto cTy = dyn_cast<RankedTensorType>(C.getType());
      if (!cTy || cTy.getRank() != 2 || !cTy.getElementType().isF32()) continue;

      OpBuilder b(mm);
      b.setInsertionPointAfter(mm);
      Location loc = mm.getLoc();
      auto dim0 = b.create<tensor::DimOp>(loc, C, 0);
      auto dim1 = b.create<tensor::DimOp>(loc, C, 1);

      auto callee = module.lookupSymbol<func::FuncOp>("fi.fi_plugin_f32");
      if (!callee) callee = ensureFiDecl(module);

        Value tamperFlag = b.create<arith::ConstantIndexOp>(
          loc, static_cast<int64_t>(doTamper ? 1 : 0));
        auto dyn2DTensor = RankedTensorType::get(
          {ShapedType::kDynamic, ShapedType::kDynamic}, cTy.getElementType());
        Value C_dyn = b.create<tensor::CastOp>(loc, dyn2DTensor, C);
        auto call = b.create<func::CallOp>(
          loc, callee.getSymName(), TypeRange{dyn2DTensor},
          ValueRange{C_dyn, dim0.getResult(), dim1.getResult(), tamperFlag});
        Value resultDyn = call.getResult(0);
        Value resultTensor = b.create<tensor::CastOp>(loc, cTy, resultDyn);

      DominanceInfo dom(func);
      C.replaceUsesWithIf(resultTensor, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        if (user == call.getOperation()) return false;
        return dom.dominates(call.getOperation(), user);
      });

      ++inserted;
    }

    if (inserted) {
      mlir::emitRemark(func.getLoc())
          << "[fi-insert-call-after-matmul] inserted=" << inserted;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createInsertFiCallPass() {
  return std::make_unique<InsertFiCallPass>();
}

static mlir::PassRegistration<InsertFiCallPass> reg;
