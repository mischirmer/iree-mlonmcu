// AByzFTPass.cpp
// Computes scaled ABFT checksums around linalg.matmul without mutating matmul.

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
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <vector>

using namespace mlir;

namespace {

constexpr StringLiteral kAbftModeAttrName = "iree.abft.mode";
constexpr StringLiteral kAbftModeNormal = "abft";
constexpr StringLiteral kAbftModeScaled = "abyzft";
constexpr StringLiteral kAbftModeFreivald = "freivald";

static llvm::cl::opt<int> abyzftScaleSamplingMode(
  "abyzft-scale-sampling-mode",
  llvm::cl::desc(
    "Scale sampling mode: 1=uniform from [-4,-0.5]U[0.5,4], "
    "2=uniform from {-4,-2,-0.5,0.5,2,4}"),
  llvm::cl::init(1));

static double sampleScaleValue(Operation *anchor) {
  constexpr double kMinAbsScale = 0.5;
  constexpr double kMaxAbsScale = 4.0;
  if (abyzftScaleSamplingMode == 1) {
    double unit = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    double magnitude = kMinAbsScale + (kMaxAbsScale - kMinAbsScale) * unit;
    double sign = (rand() % 2 == 0) ? -1.0 : 1.0;
    return sign * magnitude;
  }
  if (abyzftScaleSamplingMode == 2) {
    constexpr double kScaleChoices[] = {-4.0, -2.0, -0.5, 0.5, 2.0, 4.0};
    int choice = rand() % (sizeof(kScaleChoices) / sizeof(kScaleChoices[0]));
    return kScaleChoices[choice];
  }
  anchor->emitRemark(
      "Unknown --abyzft-scale-sampling-mode; defaulting to mode 1");
  double unit = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
  double magnitude = kMinAbsScale + (kMaxAbsScale - kMinAbsScale) * unit;
  double sign = (rand() % 2 == 0) ? -1.0 : 1.0;
  return sign * magnitude;
}

static SmallVector<double> sampleScaleVector(Operation *anchor, int64_t size) {
  SmallVector<double> values;
  values.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    values.push_back(sampleScaleValue(anchor));
  }
  return values;
}

// Returns dense float tensor values when |value| is defined by arith.constant.
static DenseFPElementsAttr getDirectDenseFloatConstant(Value value) {
  auto constantOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constantOp) return {};
  return llvm::dyn_cast_or_null<DenseFPElementsAttr>(constantOp.getValue());
}

static DenseFPElementsAttr buildDenseTensorAttr(RankedTensorType type,
                                                ArrayRef<double> values) {
  SmallVector<APFloat> apValues;
  apValues.reserve(values.size());
  for (double value : values) {
    apValues.emplace_back(static_cast<float>(value));
  }
  return llvm::cast<DenseFPElementsAttr>(
      DenseElementsAttr::get(type, apValues));
}

static DenseFPElementsAttr buildSplatTensorAttr(RankedTensorType type,
                                                double value) {
  auto numElements = type.getNumElements();
  SmallVector<double> values(numElements, value);
  return buildDenseTensorAttr(type, values);
}

static SmallVector<double> getDenseValues(DenseFPElementsAttr dense) {
  SmallVector<double> values;
  values.reserve(dense.getNumElements());
  for (const APFloat &value : dense.getValues<APFloat>()) {
    values.push_back(value.convertToDouble());
  }
  return values;
}

static DenseFPElementsAttr evaluateConstantTensor(Value value) {
  if (auto dense = getDirectDenseFloatConstant(value)) {
    return dense;
  }

  if (auto mulOp = value.getDefiningOp<arith::MulFOp>()) {
    auto resultType = llvm::dyn_cast<RankedTensorType>(mulOp.getResult().getType());
    if (!resultType) return {};

    auto lhs = evaluateConstantTensor(mulOp.getLhs());
    auto rhs = evaluateConstantTensor(mulOp.getRhs());
    if (!lhs || !rhs) return {};

    auto lhsValues = getDenseValues(lhs);
    auto rhsValues = getDenseValues(rhs);
    if (lhsValues.size() != rhsValues.size()) return {};

    SmallVector<double> resultValues;
    resultValues.reserve(lhsValues.size());
    for (size_t index = 0; index < lhsValues.size(); ++index) {
      resultValues.push_back(lhsValues[index] * rhsValues[index]);
    }
    return buildDenseTensorAttr(resultType, resultValues);
  }

  if (auto matmulOp = value.getDefiningOp<linalg::MatmulOp>()) {
    auto resultType = llvm::dyn_cast<RankedTensorType>(matmulOp.getResult(0).getType());
    auto lhsType = llvm::dyn_cast<RankedTensorType>(matmulOp.getDpsInputOperand(0)->get().getType());
    auto rhsType = llvm::dyn_cast<RankedTensorType>(matmulOp.getDpsInputOperand(1)->get().getType());
    if (!resultType || !lhsType || !rhsType) return {};

    auto lhs = evaluateConstantTensor(matmulOp.getDpsInputOperand(0)->get());
    auto rhs = evaluateConstantTensor(matmulOp.getDpsInputOperand(1)->get());
    if (!lhs || !rhs) return {};

    int64_t m = lhsType.getShape()[0];
    int64_t k = lhsType.getShape()[1];
    int64_t n = rhsType.getShape()[1];
    auto lhsValues = getDenseValues(lhs);
    auto rhsValues = getDenseValues(rhs);
    SmallVector<double> resultValues(m * n, 0.0);

    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        double sum = 0.0;
        for (int64_t inner = 0; inner < k; ++inner) {
          sum += lhsValues[row * k + inner] * rhsValues[inner * n + col];
        }
        resultValues[row * n + col] = sum;
      }
    }
    return buildDenseTensorAttr(resultType, resultValues);
  }

  return {};
}

// Emits debug output for |value| and includes literal elements for constants.
static void emitDebugValue(Operation *anchor, StringRef label, Value value) {
  llvm::errs() << "[ABFT] " << label << " type=" << value.getType();
  if (auto dense = evaluateConstantTensor(value)) {
    llvm::errs() << " values=" << dense;
  } else {
    llvm::errs() << " values=<non-constant>";
  }
  llvm::errs() << " @" << anchor->getLoc() << "\n";
}

static Value buildSplatTensorConstant(OpBuilder &builder, Location loc,
                                      RankedTensorType type, double value) {
  auto denseAttr = buildSplatTensorAttr(type, value);
  return builder.create<arith::ConstantOp>(loc, type, denseAttr);
}

static Value buildScaleRows(OpBuilder &builder, Location loc, Value matrix,
                            Value scales, RankedTensorType resultType) {
  auto elementType = resultType.getElementType();
  auto zeroAttr = builder.getFloatAttr(elementType, 0.0);
  Value init = builder.create<arith::ConstantOp>(
      loc, resultType, DenseElementsAttr::get(resultType, zeroAttr));
  return builder
      .create<linalg::GenericOp>(
          loc, TypeRange{resultType}, ValueRange{matrix, scales}, ValueRange{init},
          SmallVector<AffineMap>{
              AffineMap::getMultiDimIdentityMap(2, builder.getContext()),
              AffineMap::get(2, 0, {builder.getAffineDimExpr(0)},
                             builder.getContext()),
              AffineMap::getMultiDimIdentityMap(2, builder.getContext())},
          SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                           utils::IteratorType::parallel},
          [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            Value scaled =
                nestedBuilder.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, scaled);
          })
      .getResult(0);
}

static Value buildScaleCols(OpBuilder &builder, Location loc, Value matrix,
                            Value scales, RankedTensorType resultType) {
  auto elementType = resultType.getElementType();
  auto zeroAttr = builder.getFloatAttr(elementType, 0.0);
  Value init = builder.create<arith::ConstantOp>(
      loc, resultType, DenseElementsAttr::get(resultType, zeroAttr));
  return builder
      .create<linalg::GenericOp>(
          loc, TypeRange{resultType}, ValueRange{matrix, scales}, ValueRange{init},
          SmallVector<AffineMap>{
              AffineMap::getMultiDimIdentityMap(2, builder.getContext()),
              AffineMap::get(2, 0, {builder.getAffineDimExpr(1)},
                             builder.getContext()),
              AffineMap::getMultiDimIdentityMap(2, builder.getContext())},
          SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                           utils::IteratorType::parallel},
          [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            Value scaled =
                nestedBuilder.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, scaled);
          })
      .getResult(0);
}

// Elementwise add for rank-2 tensors with identical shapes.
static Value buildElementwiseAdd(OpBuilder &builder, Location loc, Value lhs,
                 Value rhs, RankedTensorType resultType) {
  auto elementType = resultType.getElementType();
  auto zeroAttr = builder.getFloatAttr(elementType, 0.0);
  Value init = builder.create<arith::ConstantOp>(
    loc, resultType, DenseElementsAttr::get(resultType, zeroAttr));
  return builder
    .create<linalg::GenericOp>(
      loc, TypeRange{resultType}, ValueRange{lhs, rhs}, ValueRange{init},
      SmallVector<AffineMap>{
        AffineMap::getMultiDimIdentityMap(2, builder.getContext()),
        AffineMap::getMultiDimIdentityMap(2, builder.getContext()),
        AffineMap::getMultiDimIdentityMap(2, builder.getContext())},
      SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                       utils::IteratorType::parallel},
      [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
      Value sum = nestedBuilder.create<arith::AddFOp>(nestedLoc, args[0],
                              args[1]);
      nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
      })
    .getResult(0);
}

// Builds a row-checksum tensor: tensor<Mx1> = input[MxK] * ones[Kx1].
static Value buildRowChecksum(OpBuilder &builder, Location loc, Value input,
                int64_t rows, int64_t cols,
                Type elementType) {
  auto onesType = RankedTensorType::get({cols, 1}, elementType);
  auto zerosType = RankedTensorType::get({rows, 1}, elementType);
  auto oneAttr = builder.getFloatAttr(elementType, 1.0);
  auto zeroAttr = builder.getFloatAttr(elementType, 0.0);
  Value ones = builder.create<arith::ConstantOp>(
    loc, onesType, DenseElementsAttr::get(onesType, oneAttr));
  Value zeros = builder.create<arith::ConstantOp>(
    loc, zerosType, DenseElementsAttr::get(zerosType, zeroAttr));
  return builder
    .create<linalg::MatmulOp>(loc, zerosType, ValueRange{input, ones}, zeros)
    .getResult(0);
}

// Builds a column-checksum tensor: tensor<1xN> = ones[1xM] * input[MxN].
static Value buildColumnChecksum(OpBuilder &builder, Location loc, Value input,
                 int64_t rows, int64_t cols,
                 Type elementType) {
  auto onesType = RankedTensorType::get({1, rows}, elementType);
  auto zerosType = RankedTensorType::get({1, cols}, elementType);
  auto oneAttr = builder.getFloatAttr(elementType, 1.0);
  auto zeroAttr = builder.getFloatAttr(elementType, 0.0);
  Value ones = builder.create<arith::ConstantOp>(
    loc, onesType, DenseElementsAttr::get(onesType, oneAttr));
  Value zeros = builder.create<arith::ConstantOp>(
    loc, zerosType, DenseElementsAttr::get(zerosType, zeroAttr));
  return builder
    .create<linalg::MatmulOp>(loc, zerosType, ValueRange{ones, input}, zeros)
    .getResult(0);
}

  static Value buildMatmul(OpBuilder &builder, Location loc, Value lhs, Value rhs,
               RankedTensorType resultType) {
    auto elementType = resultType.getElementType();
    auto zeroAttr = builder.getFloatAttr(elementType, 0.0);
    Value zeros = builder.create<arith::ConstantOp>(
      loc, resultType, DenseElementsAttr::get(resultType, zeroAttr));
    return builder
      .create<linalg::MatmulOp>(loc, resultType, ValueRange{lhs, rhs}, zeros)
      .getResult(0);
  }

struct AByzFTPass : public PassWrapper<AByzFTPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AByzFTPass)

  StringRef getArgument() const final { return "abyzft-insert-scale-descal"; }
  StringRef getDescription() const final {
    return "Insert ABFT checksum computation around linalg.matmul with scaled inputs and descaled outputs.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (auto modeAttr = module->getAttrOfType<StringAttr>(kAbftModeAttrName)) {
      if (modeAttr.getValue() != kAbftModeScaled) {
        module.emitError(
            "ABFT, AByzFT and Freivald are mutually exclusive; this module "
            "is already marked for a different instrumentation mode");
        signalPassFailure();
        return;
      }
    } else {
      module->setAttr(kAbftModeAttrName,
                      StringAttr::get(module.getContext(), kAbftModeScaled));
    }

    MLIRContext *ctx = module.getContext();
    ctx->getOrLoadDialect<func::FuncDialect>();
    ctx->getOrLoadDialect<linalg::LinalgDialect>();
    ctx->getOrLoadDialect<tensor::TensorDialect>();
    ctx->getOrLoadDialect<arith::ArithDialect>();
    ctx->getOrLoadDialect<math::MathDialect>();

    auto ensureFunctionWithBody = [&](StringRef name,
                                      StringRef body) -> func::FuncOp {
      if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
      OwningOpRef<ModuleOp> tmp = parseSourceString<ModuleOp>(body, ctx);
      if (!tmp) {
        module.emitRemark() << "abyzft: failed to parse helper body for " << name;
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
            << "abyzft: helper " << name << " not present in parsed body";
        return {};
      }
      Operation *cloned = srcFunc->clone();
      module.getBody()->getOperations().push_back(cloned);
      return cast<func::FuncOp>(cloned);
    };

    (void)ensureFunctionWithBody("column_checksum", R"mlir(
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
    (void)ensureFunctionWithBody("row_checksum", R"mlir(
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
    (void)ensureFunctionWithBody("vector_max_abs_diff", R"mlir(
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
    (void)ensureFunctionWithBody("vector_max_abs_diff_pair", R"mlir(
module {
  func.func @vector_max_abs_diff_pair(%v1: tensor<?xf32>, %v2: tensor<?xf32>) -> (f32, f32) {
    %neg = arith.constant -1.0 : f32
    %zero = arith.constant 0.0 : f32
    %init_diff = tensor.empty() : tensor<f32>
    %init_a = tensor.empty() : tensor<f32>
    %init_b = tensor.empty() : tensor<f32>
    %filled_diff = linalg.fill ins(%neg : f32) outs(%init_diff : tensor<f32>) -> tensor<f32>
    %filled_a = linalg.fill ins(%zero : f32) outs(%init_a : tensor<f32>) -> tensor<f32>
    %filled_b = linalg.fill ins(%zero : f32) outs(%init_b : tensor<f32>) -> tensor<f32>
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
      %d = arith.subf %a, %b : f32
      %ad = math.absf %d : f32
      %gt = arith.cmpf ogt, %ad, %cur_diff : f32
      %nd = arith.select %gt, %ad, %cur_diff : f32
      %na = arith.select %gt, %a, %cur_a : f32
      %nb = arith.select %gt, %b, %cur_b : f32
      linalg.yield %nd, %na, %nb : f32, f32, f32
    } -> (tensor<f32>, tensor<f32>, tensor<f32>)
    %best_a = tensor.extract %out_a[] : tensor<f32>
    %best_b = tensor.extract %out_b[] : tensor<f32>
    return %best_a, %best_b : f32, f32
  }
}
)mlir");

    OpBuilder modBuilder(module.getBodyRegion());
    Location modLoc = module.getLoc();
    auto f32 = modBuilder.getF32Type();
    auto dyn2d = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, f32);
    auto vecDyn = RankedTensorType::get({ShapedType::kDynamic}, f32);
    auto maybeInsertDecl = [&](StringRef name, FunctionType fnTy) {
      if (!module.lookupSymbol<func::FuncOp>(name)) {
        auto fn = modBuilder.create<func::FuncOp>(modLoc, name, fnTy);
        fn.setPrivate();
      }
    };
    if (!module.lookupSymbol<func::FuncOp>("column_checksum")) {
      maybeInsertDecl(
          "column_checksum",
          FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("row_checksum")) {
      maybeInsertDecl(
          "row_checksum",
          FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_max_abs_diff")) {
      maybeInsertDecl(
          "vector_max_abs_diff",
          FunctionType::get(ctx, TypeRange{vecDyn, vecDyn}, TypeRange{f32}));
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_max_abs_diff_pair")) {
      maybeInsertDecl(
          "vector_max_abs_diff_pair",
          FunctionType::get(ctx, TypeRange{vecDyn, vecDyn}, TypeRange{f32, f32}));
    }
    if (!module.lookupSymbol<func::FuncOp>("abft_analysis.abft_log_rowcol_delta")) {
      maybeInsertDecl(
          "abft_analysis.abft_log_rowcol_delta",
          FunctionType::get(ctx, TypeRange{f32, f32, f32}, TypeRange{}));
    }
    if (!module.lookupSymbol<func::FuncOp>("abft_analysis.abft_log_rowcol_debug")) {
      maybeInsertDecl(
          "abft_analysis.abft_log_rowcol_debug",
          FunctionType::get(ctx, TypeRange{f32, f32, f32, f32, f32}, TypeRange{}));
    }

    auto rowFn = module.lookupSymbol<func::FuncOp>(StringRef("row_checksum"));
    auto colFn = module.lookupSymbol<func::FuncOp>(StringRef("column_checksum"));
    auto vecMaxFn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_diff"));
    auto vecMaxPairFn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_diff_pair"));
    auto logRowColDeltaFn = module.lookupSymbol<func::FuncOp>(
        StringRef("abft_analysis.abft_log_rowcol_delta"));
    auto logRowColDebugFn = module.lookupSymbol<func::FuncOp>(
        StringRef("abft_analysis.abft_log_rowcol_debug"));

    SmallVector<linalg::MatmulOp> targets;
    module.walk([&](linalg::MatmulOp matmul) { targets.push_back(matmul); });

    for (auto [matmulIndex, matmul] : llvm::enumerate(targets)) {
      OpBuilder builder(matmul);
      Location loc = matmul.getLoc();

      Value lhs = matmul.getDpsInputOperand(0)->get();
      Value rhs = matmul.getDpsInputOperand(1)->get();
      Value outInit = matmul.getDpsInitOperand(0)->get();
      auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
      auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());
      auto outType = llvm::dyn_cast<RankedTensorType>(outInit.getType());
      if (!lhsType || !rhsType || !outType || !lhsType.hasStaticShape() ||
          !rhsType.hasStaticShape() || !outType.hasStaticShape() ||
          lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
          outType.getRank() != 2) {
        matmul.emitError(
            "Expected static rank-2 tensor operands/results for AByzFT");
        signalPassFailure();
        return;
      }

      int64_t M = lhsType.getShape()[0];
      int64_t K = lhsType.getShape()[1];
      int64_t rhsK = rhsType.getShape()[0];
      int64_t N = rhsType.getShape()[1];
      if (K != rhsK || outType.getShape()[0] != M || outType.getShape()[1] != N) {
        matmul.emitError("Matmul operand/result shapes are inconsistent");
        signalPassFailure();
        return;
      }

      Type elemType = lhsType.getElementType();
      auto rowScaleValues = sampleScaleVector(matmul, M);
      auto colScaleValues = sampleScaleVector(matmul, N);
      auto rowVectorType = RankedTensorType::get({M}, elemType);
      auto colVectorType = RankedTensorType::get({N}, elemType);
      Value rowScaleVector = builder.create<arith::ConstantOp>(
          loc, rowVectorType,
          buildDenseTensorAttr(rowVectorType, rowScaleValues));
      Value colScaleVector = builder.create<arith::ConstantOp>(
          loc, colVectorType,
          buildDenseTensorAttr(colVectorType, colScaleValues));
      SmallVector<double> invRowScaleValues;
      invRowScaleValues.reserve(rowScaleValues.size());
      for (double value : rowScaleValues)
        invRowScaleValues.push_back(1.0 / value);
      SmallVector<double> invColScaleValues;
      invColScaleValues.reserve(colScaleValues.size());
      for (double value : colScaleValues)
        invColScaleValues.push_back(1.0 / value);
      Value invRowScale = builder.create<arith::ConstantOp>(
          loc, rowVectorType,
          buildDenseTensorAttr(rowVectorType, invRowScaleValues));
      Value invColScale = builder.create<arith::ConstantOp>(
          loc, colVectorType,
          buildDenseTensorAttr(colVectorType, invColScaleValues));

      Value lhsColChecksum =
          buildColumnChecksum(builder, loc, lhs, M, K, elemType);
      Value rhsRowChecksum =
          buildRowChecksum(builder, loc, rhs, K, N, elemType);

      Value lhsScaled =
          buildScaleRows(builder, loc, lhs, rowScaleVector, lhsType);
      Value rhsScaled =
          buildScaleCols(builder, loc, rhs, colScaleVector, rhsType);
      Value zeroInit = buildSplatTensorConstant(builder, loc, outType, 0.0);

      matmul->setOperand(0, lhsScaled);
      matmul->setOperand(1, rhsScaled);
      matmul->setOperand(2, zeroInit);

      builder.setInsertionPointAfter(matmul);
      Value scaledMatmulResult = matmul.getResult(0);
      Value descaleRows =
          buildScaleRows(builder, loc, scaledMatmulResult, invRowScale, outType);
      Value descaleCols =
          buildScaleCols(builder, loc, descaleRows, invColScale, outType);
      Value matmulResult = buildElementwiseAdd(builder, loc, descaleCols, outInit,
                                               outType);

      Value outRowChecksum = buildMatmul(
          builder, loc, matmulResult,
          buildSplatTensorConstant(builder, loc,
                                   RankedTensorType::get({N, 1}, elemType), 1.0),
          RankedTensorType::get({M, 1}, elemType));
      Value outColChecksum = buildMatmul(
          builder, loc,
          buildSplatTensorConstant(builder, loc,
                                   RankedTensorType::get({1, M}, elemType), 1.0),
          matmulResult, RankedTensorType::get({1, N}, elemType));
      Value expectedOutRowChecksum =
          buildMatmul(builder, loc, lhs, rhsRowChecksum,
                      RankedTensorType::get({M, 1}, elemType));
      Value expectedOutColChecksum =
          buildMatmul(builder, loc, lhsColChecksum, rhs,
                      RankedTensorType::get({1, N}, elemType));

      // Include init/outs tensor contribution: C = A*B + init.
      Value initRowChecksum = buildMatmul(
          builder, loc, outInit,
          buildSplatTensorConstant(builder, loc,
                                   RankedTensorType::get({N, 1}, elemType), 1.0),
          RankedTensorType::get({M, 1}, elemType));
      Value initColChecksum = buildMatmul(
          builder, loc,
          buildSplatTensorConstant(builder, loc,
                                   RankedTensorType::get({1, M}, elemType), 1.0),
          outInit, RankedTensorType::get({1, N}, elemType));
      expectedOutRowChecksum = buildElementwiseAdd(
          builder, loc, expectedOutRowChecksum, initRowChecksum,
          RankedTensorType::get({M, 1}, elemType));
      expectedOutColChecksum = buildElementwiseAdd(
          builder, loc, expectedOutColChecksum, initColChecksum,
          RankedTensorType::get({1, N}, elemType));

      Value rowMaxDelta =
          builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                            builder.getF32FloatAttr(0.0f));
      Value colMaxDelta =
          builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                            builder.getF32FloatAttr(0.0f));
      Value rowExpMax = rowMaxDelta;
      Value rowCalcMax = rowMaxDelta;
      Value colExpMax = rowMaxDelta;
      Value colCalcMax = rowMaxDelta;

      if (rowFn && colFn && vecMaxFn && !vecMaxFn.isExternal()) {
        auto rowTy = rowFn.getFunctionType();
        Value rowExpArg = expectedOutRowChecksum;
        if (rowExpArg.getType() != rowTy.getInput(0))
          rowExpArg = builder
                          .create<tensor::CastOp>(loc, rowTy.getInput(0), rowExpArg)
                          .getResult();
        Value rowCalcArg = outRowChecksum;
        if (rowCalcArg.getType() != rowTy.getInput(0))
          rowCalcArg = builder
                           .create<tensor::CastOp>(loc, rowTy.getInput(0), rowCalcArg)
                           .getResult();

        SmallVector<Type, 1> rowRes;
        for (Type t : rowTy.getResults())
          rowRes.push_back(t);
        Value rowExpectedVec = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("row_checksum"),
                                       TypeRange(rowRes), ValueRange{rowExpArg})
                                   .getResult(0);
        Value rowCalcVec = builder
                               .create<func::CallOp>(
                                   loc, StringRef("row_checksum"),
                                   TypeRange(rowRes), ValueRange{rowCalcArg})
                               .getResult(0);

        auto colTy = colFn.getFunctionType();
        Value colExpArg = expectedOutColChecksum;
        if (colExpArg.getType() != colTy.getInput(0))
          colExpArg = builder
                          .create<tensor::CastOp>(loc, colTy.getInput(0), colExpArg)
                          .getResult();
        Value colCalcArg = outColChecksum;
        if (colCalcArg.getType() != colTy.getInput(0))
          colCalcArg = builder
                           .create<tensor::CastOp>(loc, colTy.getInput(0), colCalcArg)
                           .getResult();

        SmallVector<Type, 1> colRes;
        for (Type t : colTy.getResults())
          colRes.push_back(t);
        Value colExpectedVec = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("column_checksum"),
                                       TypeRange(colRes), ValueRange{colExpArg})
                                   .getResult(0);
        Value colCalcVec = builder
                               .create<func::CallOp>(
                                   loc, StringRef("column_checksum"),
                                   TypeRange(colRes), ValueRange{colCalcArg})
                               .getResult(0);

        auto maxTy = vecMaxFn.getFunctionType();
        SmallVector<Type, 1> maxRes;
        for (Type t : maxTy.getResults())
          maxRes.push_back(t);
        Value rowExpectedVecArg = rowExpectedVec;
        if (rowExpectedVecArg.getType() != maxTy.getInput(0))
          rowExpectedVecArg = builder.create<tensor::CastOp>(
              loc, maxTy.getInput(0), rowExpectedVecArg).getResult();
        Value rowCalcVecArg = rowCalcVec;
        if (rowCalcVecArg.getType() != maxTy.getInput(1))
          rowCalcVecArg = builder
                              .create<tensor::CastOp>(loc, maxTy.getInput(1),
                                                      rowCalcVecArg)
                              .getResult();
        rowMaxDelta = builder
                          .create<func::CallOp>(
                              loc, StringRef("vector_max_abs_diff"),
                              TypeRange(maxRes),
                              ValueRange{rowExpectedVecArg, rowCalcVecArg})
                          .getResult(0);

        Value colExpectedVecArg = colExpectedVec;
        if (colExpectedVecArg.getType() != maxTy.getInput(0))
          colExpectedVecArg = builder.create<tensor::CastOp>(
              loc, maxTy.getInput(0), colExpectedVecArg).getResult();
        Value colCalcVecArg = colCalcVec;
        if (colCalcVecArg.getType() != maxTy.getInput(1))
          colCalcVecArg = builder
                              .create<tensor::CastOp>(loc, maxTy.getInput(1),
                                                      colCalcVecArg)
                              .getResult();
        colMaxDelta = builder
                          .create<func::CallOp>(
                              loc, StringRef("vector_max_abs_diff"),
                              TypeRange(maxRes),
                              ValueRange{colExpectedVecArg, colCalcVecArg})
                          .getResult(0);

        if (vecMaxPairFn && !vecMaxPairFn.isExternal()) {
          auto pairTy = vecMaxPairFn.getFunctionType();
          SmallVector<Type, 2> pairRes;
          for (Type t : pairTy.getResults())
            pairRes.push_back(t);

          Value pairRowExpected = rowExpectedVec;
          if (pairRowExpected.getType() != pairTy.getInput(0))
            pairRowExpected = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(0), pairRowExpected).getResult();
          Value pairRowCalc = rowCalcVec;
          if (pairRowCalc.getType() != pairTy.getInput(1))
            pairRowCalc = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(1), pairRowCalc).getResult();
          auto rowPairCall = builder.create<func::CallOp>(
              loc, StringRef("vector_max_abs_diff_pair"), TypeRange(pairRes),
              ValueRange{pairRowExpected, pairRowCalc});
          rowExpMax = rowPairCall.getResult(0);
          rowCalcMax = rowPairCall.getResult(1);

          Value pairColExpected = colExpectedVec;
          if (pairColExpected.getType() != pairTy.getInput(0))
            pairColExpected = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(0), pairColExpected).getResult();
          Value pairColCalc = colCalcVec;
          if (pairColCalc.getType() != pairTy.getInput(1))
            pairColCalc = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(1), pairColCalc).getResult();
          auto colPairCall = builder.create<func::CallOp>(
              loc, StringRef("vector_max_abs_diff_pair"), TypeRange(pairRes),
              ValueRange{pairColExpected, pairColCalc});
          colExpMax = colPairCall.getResult(0);
          colCalcMax = colPairCall.getResult(1);
        }
      }

      emitDebugValue(matmul, "lhs", lhs);
      emitDebugValue(matmul, "rhs", rhs);
      emitDebugValue(matmul, "output", matmulResult);
      emitDebugValue(lhsScaled.getDefiningOp(), "scaled_lhs", lhsScaled);
      emitDebugValue(rhsScaled.getDefiningOp(), "scaled_rhs", rhsScaled);
      emitDebugValue(descaleCols.getDefiningOp(), "descaled_output", descaleCols);
      emitDebugValue(lhsColChecksum.getDefiningOp(), "lhs_col_checksum",
                     lhsColChecksum);
      emitDebugValue(rhsRowChecksum.getDefiningOp(), "rhs_row_checksum",
                     rhsRowChecksum);
      emitDebugValue(outRowChecksum.getDefiningOp(), "output_row_checksum_direct",
                     outRowChecksum);
      emitDebugValue(expectedOutRowChecksum.getDefiningOp(),
                     "output_row_checksum_expected", expectedOutRowChecksum);
      emitDebugValue(outColChecksum.getDefiningOp(), "output_col_checksum_direct",
                     outColChecksum);
      emitDebugValue(expectedOutColChecksum.getDefiningOp(),
                     "output_col_checksum_expected", expectedOutColChecksum);

      Value indexConst = builder.create<arith::ConstantOp>(
          loc, builder.getF32Type(),
          builder.getF32FloatAttr(static_cast<float>(matmulIndex)));
      if (logRowColDeltaFn) {
        builder.create<func::CallOp>(
            loc, StringRef("abft_analysis.abft_log_rowcol_delta"), TypeRange{},
            ValueRange{indexConst, rowMaxDelta, colMaxDelta});
      }
      if (logRowColDebugFn) {
        builder.create<func::CallOp>(
            loc, StringRef("abft_analysis.abft_log_rowcol_debug"), TypeRange{},
            ValueRange{indexConst, rowExpMax, rowCalcMax, colExpMax, colCalcMax});
      }

      DominanceInfo dom(module);
      matmul.getResult(0).replaceUsesWithIf(matmulResult, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return user != matmulResult.getDefiningOp() &&
               dom.properlyDominates(matmulResult.getDefiningOp(), user);
      });
    }
  }
};

struct FreivaldPass : public PassWrapper<FreivaldPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FreivaldPass)

  StringRef getArgument() const final {
    return "freivald-insert-weighted-checksums";
  }
  StringRef getDescription() const final {
    return "Insert ABFT checksum computation with weighted checksum generation without scaling the matmul operands.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (auto modeAttr = module->getAttrOfType<StringAttr>(kAbftModeAttrName)) {
      if (modeAttr.getValue() != kAbftModeFreivald) {
        module.emitError(
            "ABFT, AByzFT and Freivald are mutually exclusive; this module "
            "is already marked for a different instrumentation mode");
        signalPassFailure();
        return;
      }
    } else {
      module->setAttr(kAbftModeAttrName,
                      StringAttr::get(module.getContext(), kAbftModeFreivald));
    }

    MLIRContext *ctx = module.getContext();
    ctx->getOrLoadDialect<func::FuncDialect>();
    ctx->getOrLoadDialect<linalg::LinalgDialect>();
    ctx->getOrLoadDialect<tensor::TensorDialect>();
    ctx->getOrLoadDialect<arith::ArithDialect>();
    ctx->getOrLoadDialect<math::MathDialect>();

    auto ensureFunctionWithBody = [&](StringRef name,
                                      StringRef body) -> func::FuncOp {
      if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
      OwningOpRef<ModuleOp> tmp = parseSourceString<ModuleOp>(body, ctx);
      if (!tmp) {
        module.emitRemark() << "freivald: failed to parse helper body for "
                            << name;
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
            << "freivald: helper " << name << " not present in parsed body";
        return {};
      }
      Operation *cloned = srcFunc->clone();
      module.getBody()->getOperations().push_back(cloned);
      return cast<func::FuncOp>(cloned);
    };

    (void)ensureFunctionWithBody("column_checksum", R"mlir(
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
    (void)ensureFunctionWithBody("row_checksum", R"mlir(
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
    (void)ensureFunctionWithBody("vector_max_abs_diff", R"mlir(
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
    (void)ensureFunctionWithBody("vector_max_abs_diff_pair", R"mlir(
module {
  func.func @vector_max_abs_diff_pair(%v1: tensor<?xf32>, %v2: tensor<?xf32>) -> (f32, f32) {
    %neg = arith.constant -1.0 : f32
    %zero = arith.constant 0.0 : f32
    %init_diff = tensor.empty() : tensor<f32>
    %init_a = tensor.empty() : tensor<f32>
    %init_b = tensor.empty() : tensor<f32>
    %filled_diff = linalg.fill ins(%neg : f32) outs(%init_diff : tensor<f32>) -> tensor<f32>
    %filled_a = linalg.fill ins(%zero : f32) outs(%init_a : tensor<f32>) -> tensor<f32>
    %filled_b = linalg.fill ins(%zero : f32) outs(%init_b : tensor<f32>) -> tensor<f32>
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
      %d = arith.subf %a, %b : f32
      %ad = math.absf %d : f32
      %gt = arith.cmpf ogt, %ad, %cur_diff : f32
      %nd = arith.select %gt, %ad, %cur_diff : f32
      %na = arith.select %gt, %a, %cur_a : f32
      %nb = arith.select %gt, %b, %cur_b : f32
      linalg.yield %nd, %na, %nb : f32, f32, f32
    } -> (tensor<f32>, tensor<f32>, tensor<f32>)
    %best_a = tensor.extract %out_a[] : tensor<f32>
    %best_b = tensor.extract %out_b[] : tensor<f32>
    return %best_a, %best_b : f32, f32
  }
}
)mlir");

    OpBuilder modBuilder(module.getBodyRegion());
    Location modLoc = module.getLoc();
    auto f32 = modBuilder.getF32Type();
    auto dyn2d = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, f32);
    auto vecDyn = RankedTensorType::get({ShapedType::kDynamic}, f32);
    auto maybeInsertDecl = [&](StringRef name, FunctionType fnTy) {
      if (!module.lookupSymbol<func::FuncOp>(name)) {
        auto fn = modBuilder.create<func::FuncOp>(modLoc, name, fnTy);
        fn.setPrivate();
      }
    };
    if (!module.lookupSymbol<func::FuncOp>("column_checksum")) {
      maybeInsertDecl(
          "column_checksum",
          FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("row_checksum")) {
      maybeInsertDecl(
          "row_checksum",
          FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_max_abs_diff")) {
      maybeInsertDecl(
          "vector_max_abs_diff",
          FunctionType::get(ctx, TypeRange{vecDyn, vecDyn}, TypeRange{f32}));
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_max_abs_diff_pair")) {
      maybeInsertDecl(
          "vector_max_abs_diff_pair",
          FunctionType::get(ctx, TypeRange{vecDyn, vecDyn}, TypeRange{f32, f32}));
    }
    if (!module.lookupSymbol<func::FuncOp>("abft_analysis.abft_log_rowcol_delta")) {
      maybeInsertDecl(
          "abft_analysis.abft_log_rowcol_delta",
          FunctionType::get(ctx, TypeRange{f32, f32, f32}, TypeRange{}));
    }
    if (!module.lookupSymbol<func::FuncOp>("abft_analysis.abft_log_rowcol_debug")) {
      maybeInsertDecl(
          "abft_analysis.abft_log_rowcol_debug",
          FunctionType::get(ctx, TypeRange{f32, f32, f32, f32, f32}, TypeRange{}));
    }

    auto rowFn = module.lookupSymbol<func::FuncOp>(StringRef("row_checksum"));
    auto colFn = module.lookupSymbol<func::FuncOp>(StringRef("column_checksum"));
    auto vecMaxFn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_diff"));
    auto vecMaxPairFn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_max_abs_diff_pair"));
    auto logRowColDeltaFn = module.lookupSymbol<func::FuncOp>(
        StringRef("abft_analysis.abft_log_rowcol_delta"));
    auto logRowColDebugFn = module.lookupSymbol<func::FuncOp>(
        StringRef("abft_analysis.abft_log_rowcol_debug"));

    SmallVector<linalg::MatmulOp> targets;
    module.walk([&](linalg::MatmulOp matmul) { targets.push_back(matmul); });

    for (auto [matmulIndex, matmul] : llvm::enumerate(targets)) {
      OpBuilder builder(matmul);
      Location loc = matmul.getLoc();

      Value lhs = matmul.getDpsInputOperand(0)->get();
      Value rhs = matmul.getDpsInputOperand(1)->get();
      Value outInit = matmul.getDpsInitOperand(0)->get();
      auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
      auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());
      auto outType = llvm::dyn_cast<RankedTensorType>(outInit.getType());
      if (!lhsType || !rhsType || !outType || !lhsType.hasStaticShape() ||
          !rhsType.hasStaticShape() || !outType.hasStaticShape() ||
          lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
          outType.getRank() != 2) {
        matmul.emitError(
            "Expected static rank-2 tensor operands/results for Freivald");
        signalPassFailure();
        return;
      }

      int64_t M = lhsType.getShape()[0];
      int64_t K = lhsType.getShape()[1];
      int64_t rhsK = rhsType.getShape()[0];
      int64_t N = rhsType.getShape()[1];
      if (K != rhsK || outType.getShape()[0] != M || outType.getShape()[1] != N) {
        matmul.emitError("Matmul operand/result shapes are inconsistent");
        signalPassFailure();
        return;
      }

      Type elemType = lhsType.getElementType();
      auto rowScaleValues = sampleScaleVector(matmul, M);
      auto colScaleValues = sampleScaleVector(matmul, N);
      auto rowScaleType = RankedTensorType::get({1, M}, elemType);
      auto colScaleType = RankedTensorType::get({N, 1}, elemType);
      Value rowScale = builder.create<arith::ConstantOp>(
          loc, rowScaleType, buildDenseTensorAttr(rowScaleType, rowScaleValues));
      Value colScale = builder.create<arith::ConstantOp>(
          loc, colScaleType, buildDenseTensorAttr(colScaleType, colScaleValues));

      Value scaledLhsColChecksum = buildMatmul(
          builder, loc, rowScale, lhs, RankedTensorType::get({1, K}, elemType));
      Value scaledRhsRowChecksum = buildMatmul(
          builder, loc, rhs, colScale, RankedTensorType::get({K, 1}, elemType));

      builder.setInsertionPointAfter(matmul);
      Value matmulResult = matmul.getResult(0);

      Value outRowChecksum =
          buildMatmul(builder, loc, matmulResult, colScale,
                      RankedTensorType::get({M, 1}, elemType));
      Value outColChecksum =
          buildMatmul(builder, loc, rowScale, matmulResult,
                      RankedTensorType::get({1, N}, elemType));
      Value expectedOutRowChecksum =
          buildMatmul(builder, loc, lhs, scaledRhsRowChecksum,
                      RankedTensorType::get({M, 1}, elemType));
      Value expectedOutColChecksum =
          buildMatmul(builder, loc, scaledLhsColChecksum, rhs,
                      RankedTensorType::get({1, N}, elemType));

      Value initRowChecksum = buildMatmul(
          builder, loc, outInit, colScale, RankedTensorType::get({M, 1}, elemType));
      Value initColChecksum = buildMatmul(
          builder, loc, rowScale, outInit, RankedTensorType::get({1, N}, elemType));
      expectedOutRowChecksum = buildElementwiseAdd(
          builder, loc, expectedOutRowChecksum, initRowChecksum,
          RankedTensorType::get({M, 1}, elemType));
      expectedOutColChecksum = buildElementwiseAdd(
          builder, loc, expectedOutColChecksum, initColChecksum,
          RankedTensorType::get({1, N}, elemType));

      Value rowMaxDelta =
          builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                            builder.getF32FloatAttr(0.0f));
      Value colMaxDelta =
          builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                            builder.getF32FloatAttr(0.0f));
      Value rowExpMax = rowMaxDelta;
      Value rowCalcMax = rowMaxDelta;
      Value colExpMax = rowMaxDelta;
      Value colCalcMax = rowMaxDelta;

      if (rowFn && colFn && vecMaxFn && !vecMaxFn.isExternal()) {
        auto rowTy = rowFn.getFunctionType();
        Value rowExpArg = expectedOutRowChecksum;
        if (rowExpArg.getType() != rowTy.getInput(0))
          rowExpArg = builder
                          .create<tensor::CastOp>(loc, rowTy.getInput(0), rowExpArg)
                          .getResult();
        Value rowCalcArg = outRowChecksum;
        if (rowCalcArg.getType() != rowTy.getInput(0))
          rowCalcArg = builder
                           .create<tensor::CastOp>(loc, rowTy.getInput(0), rowCalcArg)
                           .getResult();

        SmallVector<Type, 1> rowRes;
        for (Type t : rowTy.getResults())
          rowRes.push_back(t);
        Value rowExpectedVec = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("row_checksum"),
                                       TypeRange(rowRes), ValueRange{rowExpArg})
                                   .getResult(0);
        Value rowCalcVec = builder
                               .create<func::CallOp>(
                                   loc, StringRef("row_checksum"),
                                   TypeRange(rowRes), ValueRange{rowCalcArg})
                               .getResult(0);

        auto colTy = colFn.getFunctionType();
        Value colExpArg = expectedOutColChecksum;
        if (colExpArg.getType() != colTy.getInput(0))
          colExpArg = builder
                          .create<tensor::CastOp>(loc, colTy.getInput(0), colExpArg)
                          .getResult();
        Value colCalcArg = outColChecksum;
        if (colCalcArg.getType() != colTy.getInput(0))
          colCalcArg = builder
                           .create<tensor::CastOp>(loc, colTy.getInput(0), colCalcArg)
                           .getResult();

        SmallVector<Type, 1> colRes;
        for (Type t : colTy.getResults())
          colRes.push_back(t);
        Value colExpectedVec = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("column_checksum"),
                                       TypeRange(colRes), ValueRange{colExpArg})
                                   .getResult(0);
        Value colCalcVec = builder
                               .create<func::CallOp>(
                                   loc, StringRef("column_checksum"),
                                   TypeRange(colRes), ValueRange{colCalcArg})
                               .getResult(0);

        auto maxTy = vecMaxFn.getFunctionType();
        SmallVector<Type, 1> maxRes;
        for (Type t : maxTy.getResults())
          maxRes.push_back(t);
        Value rowExpectedVecArg = rowExpectedVec;
        if (rowExpectedVecArg.getType() != maxTy.getInput(0))
          rowExpectedVecArg = builder.create<tensor::CastOp>(
              loc, maxTy.getInput(0), rowExpectedVecArg).getResult();
        Value rowCalcVecArg = rowCalcVec;
        if (rowCalcVecArg.getType() != maxTy.getInput(1))
          rowCalcVecArg = builder
                              .create<tensor::CastOp>(loc, maxTy.getInput(1),
                                                      rowCalcVecArg)
                              .getResult();
        rowMaxDelta = builder
                          .create<func::CallOp>(
                              loc, StringRef("vector_max_abs_diff"),
                              TypeRange(maxRes),
                              ValueRange{rowExpectedVecArg, rowCalcVecArg})
                          .getResult(0);

        Value colExpectedVecArg = colExpectedVec;
        if (colExpectedVecArg.getType() != maxTy.getInput(0))
          colExpectedVecArg = builder.create<tensor::CastOp>(
              loc, maxTy.getInput(0), colExpectedVecArg).getResult();
        Value colCalcVecArg = colCalcVec;
        if (colCalcVecArg.getType() != maxTy.getInput(1))
          colCalcVecArg = builder
                              .create<tensor::CastOp>(loc, maxTy.getInput(1),
                                                      colCalcVecArg)
                              .getResult();
        colMaxDelta = builder
                          .create<func::CallOp>(
                              loc, StringRef("vector_max_abs_diff"),
                              TypeRange(maxRes),
                              ValueRange{colExpectedVecArg, colCalcVecArg})
                          .getResult(0);

        if (vecMaxPairFn && !vecMaxPairFn.isExternal()) {
          auto pairTy = vecMaxPairFn.getFunctionType();
          SmallVector<Type, 2> pairRes;
          for (Type t : pairTy.getResults())
            pairRes.push_back(t);

          Value pairRowExpected = rowExpectedVec;
          if (pairRowExpected.getType() != pairTy.getInput(0))
            pairRowExpected = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(0), pairRowExpected).getResult();
          Value pairRowCalc = rowCalcVec;
          if (pairRowCalc.getType() != pairTy.getInput(1))
            pairRowCalc = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(1), pairRowCalc).getResult();
          auto rowPairCall = builder.create<func::CallOp>(
              loc, StringRef("vector_max_abs_diff_pair"), TypeRange(pairRes),
              ValueRange{pairRowExpected, pairRowCalc});
          rowExpMax = rowPairCall.getResult(0);
          rowCalcMax = rowPairCall.getResult(1);

          Value pairColExpected = colExpectedVec;
          if (pairColExpected.getType() != pairTy.getInput(0))
            pairColExpected = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(0), pairColExpected).getResult();
          Value pairColCalc = colCalcVec;
          if (pairColCalc.getType() != pairTy.getInput(1))
            pairColCalc = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(1), pairColCalc).getResult();
          auto colPairCall = builder.create<func::CallOp>(
              loc, StringRef("vector_max_abs_diff_pair"), TypeRange(pairRes),
              ValueRange{pairColExpected, pairColCalc});
          colExpMax = colPairCall.getResult(0);
          colCalcMax = colPairCall.getResult(1);
        }
      }

      emitDebugValue(matmul, "lhs", lhs);
      emitDebugValue(matmul, "rhs", rhs);
      emitDebugValue(matmul, "output", matmulResult);
      emitDebugValue(scaledLhsColChecksum.getDefiningOp(),
                     "scaled_lhs_col_checksum", scaledLhsColChecksum);
      emitDebugValue(scaledRhsRowChecksum.getDefiningOp(),
                     "scaled_rhs_row_checksum", scaledRhsRowChecksum);
      emitDebugValue(outRowChecksum.getDefiningOp(), "output_row_checksum_direct",
                     outRowChecksum);
      emitDebugValue(expectedOutRowChecksum.getDefiningOp(),
                     "output_row_checksum_expected", expectedOutRowChecksum);
      emitDebugValue(outColChecksum.getDefiningOp(), "output_col_checksum_direct",
                     outColChecksum);
      emitDebugValue(expectedOutColChecksum.getDefiningOp(),
                     "output_col_checksum_expected", expectedOutColChecksum);

      Value indexConst = builder.create<arith::ConstantOp>(
          loc, builder.getF32Type(),
          builder.getF32FloatAttr(static_cast<float>(matmulIndex)));
      if (logRowColDeltaFn) {
        builder.create<func::CallOp>(
            loc, StringRef("abft_analysis.abft_log_rowcol_delta"), TypeRange{},
            ValueRange{indexConst, rowMaxDelta, colMaxDelta});
      }
      if (logRowColDebugFn) {
        builder.create<func::CallOp>(
            loc, StringRef("abft_analysis.abft_log_rowcol_debug"), TypeRange{},
            ValueRange{indexConst, rowExpMax, rowCalcMax, colExpMax, colCalcMax});
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAByzFTPass() {
  return std::make_unique<AByzFTPass>();
}

std::unique_ptr<mlir::Pass> createFreivaldPass() {
  return std::make_unique<FreivaldPass>();
}

static mlir::PassRegistration<AByzFTPass> reg;
static mlir::PassRegistration<FreivaldPass> regFreivald;
