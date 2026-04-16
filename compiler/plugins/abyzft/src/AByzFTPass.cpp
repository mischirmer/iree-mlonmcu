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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cmath>
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

static llvm::cl::opt<bool> abyzftInjectFault(
    "abyzft-inject-fault",
    llvm::cl::desc("Enable deterministic synthetic fault injection for testing"),
    llvm::cl::init(false));

static llvm::cl::opt<int> abyzftInjectFaultDelta(
    "abyzft-inject-fault-delta",
    llvm::cl::desc("Fault injection additive delta"),
    llvm::cl::init(1));

static llvm::cl::opt<std::string> abyzftInjectFaultPattern(
    "abyzft-inject-fault-pattern",
    llvm::cl::desc("Fault injection pattern: single_point, trivial, checkered"),
    llvm::cl::init("single_point"));

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

static Value buildElementwiseSub(OpBuilder &builder, Location loc, Value lhs,
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
      Value diff = nestedBuilder.create<arith::SubFOp>(nestedLoc, args[0],
                              args[1]);
      nestedBuilder.create<linalg::YieldOp>(nestedLoc, diff);
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

static Value buildDynamicRowvecMulMat(OpBuilder &builder, Location loc, Value rv,
                                      Value mat) {
  auto matType = llvm::cast<RankedTensorType>(mat.getType());
  auto elemType = matType.getElementType();
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value n = builder.create<tensor::DimOp>(loc, mat, c1);
  auto row2dType = RankedTensorType::get({1, ShapedType::kDynamic}, elemType);
  auto out2dType = RankedTensorType::get({1, ShapedType::kDynamic}, elemType);
  auto out1dType = RankedTensorType::get({ShapedType::kDynamic}, elemType);
  Value rv2d =
      builder.create<tensor::ExpandShapeOp>(loc, row2dType, rv,
                                            ReassociationIndices{{0, 1}});
  Value empty2d =
      builder.create<tensor::EmptyOp>(loc, out2dType, ValueRange{n});
  Value zero = builder.create<arith::ConstantFloatOp>(
      loc, APFloat(0.0f), llvm::cast<FloatType>(elemType));
  Value init2d =
      builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty2d})
          .getResult(0);
  Value res2d =
      builder
          .create<linalg::MatmulOp>(loc, out2dType, ValueRange{rv2d, mat}, init2d)
          .getResult(0);
  return builder.create<tensor::CollapseShapeOp>(loc, out1dType, res2d,
                                                  ReassociationIndices{{0, 1}})
      .getResult();
}

static Value buildDynamicMatMulColvec(OpBuilder &builder, Location loc, Value mat,
                                      Value cv) {
  auto matType = llvm::cast<RankedTensorType>(mat.getType());
  auto elemType = matType.getElementType();
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value m = builder.create<tensor::DimOp>(loc, mat, c0);
  auto col2dType = RankedTensorType::get({ShapedType::kDynamic, 1}, elemType);
  auto out2dType = RankedTensorType::get({ShapedType::kDynamic, 1}, elemType);
  auto out1dType = RankedTensorType::get({ShapedType::kDynamic}, elemType);
  Value cv2d =
      builder.create<tensor::ExpandShapeOp>(loc, col2dType, cv,
                                            ReassociationIndices{{0, 1}});
  Value empty2d =
      builder.create<tensor::EmptyOp>(loc, out2dType, ValueRange{m});
  Value zero = builder.create<arith::ConstantFloatOp>(
      loc, APFloat(0.0f), llvm::cast<FloatType>(elemType));
  Value init2d =
      builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty2d})
          .getResult(0);
  Value res2d =
      builder
          .create<linalg::MatmulOp>(loc, out2dType, ValueRange{mat, cv2d}, init2d)
          .getResult(0);
  return builder.create<tensor::CollapseShapeOp>(loc, out1dType, res2d,
                                                  ReassociationIndices{{0, 1}})
      .getResult();
}

struct AByzFTPass : public PassWrapper<AByzFTPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AByzFTPass)

  StringRef getArgument() const final { return "abyzft-insert-scale-descal"; }
  StringRef getDescription() const final {
    return "Insert ABFT checksum computation around linalg.matmul with scaled inputs and descaled outputs.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    (void)abyzftInjectFaultPattern;
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

    auto symbolExists = [&](StringRef name) {
      for (Operation &op : module.getBody()->getOperations()) {
        if (auto symName =
                op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
          if (symName.getValue() == name)
            return true;
        }
      }
      return false;
    };

    auto eraseConflictingSymbol = [&](StringRef name) {
      for (Operation &op : llvm::make_early_inc_range(module.getBody()->getOperations())) {
        if (auto symName =
                op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
          if (symName.getValue() == name) {
            op.erase();
            return;
          }
        }
      }
    };

    auto ensureFunctionWithBody = [&](StringRef name,
                                      StringRef body) -> func::FuncOp {
      if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
        if (!existing.empty())
          return existing;
        existing.erase();
      } else if (symbolExists(name)) {
        eraseConflictingSymbol(name);
      }
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
    (void)ensureFunctionWithBody("sample_row_scales", R"mlir(
module {
  func.func @sample_row_scales(%mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
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
    (void)ensureFunctionWithBody("sample_col_scales", R"mlir(
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
    (void)ensureFunctionWithBody("rowvec_mul_mat", R"mlir(
module {
  func.func @rowvec_mul_mat(%rv: tensor<?xf32>, %mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %k = tensor.dim %rv, %c0 : tensor<?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %rv2d = tensor.expand_shape %rv [[0, 1]] output_shape [%c1, %k] : tensor<?xf32> into tensor<1x?xf32>
    %empty2d = tensor.empty(%n) : tensor<1x?xf32>
    %zero = arith.constant 0.0 : f32
    %init2d = linalg.fill ins(%zero : f32) outs(%empty2d : tensor<1x?xf32>) -> tensor<1x?xf32>
    %res2d = linalg.matmul ins(%rv2d, %mat : tensor<1x?xf32>, tensor<?x?xf32>)
      outs(%init2d : tensor<1x?xf32>) -> tensor<1x?xf32>
    %res = tensor.collapse_shape %res2d [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("mat_mul_colvec", R"mlir(
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
    (void)ensureFunctionWithBody("vector_add", R"mlir(
module {
  func.func @vector_add(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %lhs, %c0 : tensor<?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]
    } ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("matrix_add", R"mlir(
module {
  func.func @matrix_add(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %n = tensor.dim %lhs, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("zero_matrix_like", R"mlir(
module {
  func.func @zero_matrix_like(%mat: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %res = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("scale_matrix_rows", R"mlir(
module {
  func.func @scale_matrix_rows(%mat: tensor<?x?xf32>, %scales: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%mat, %scales : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %s: f32, %acc: f32):
        %prod = arith.mulf %a, %s : f32
        linalg.yield %prod : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("scale_matrix_cols", R"mlir(
module {
  func.func @scale_matrix_cols(%mat: tensor<?x?xf32>, %scales: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%mat, %scales : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %s: f32, %acc: f32):
        %prod = arith.mulf %a, %s : f32
        linalg.yield %prod : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("sample_row_scales", R"mlir(
module {
  func.func @sample_row_scales(%mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
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
    (void)ensureFunctionWithBody("sample_col_scales", R"mlir(
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
    (void)ensureFunctionWithBody("scale_matrix_rows", R"mlir(
module {
  func.func @scale_matrix_rows(%mat: tensor<?x?xf32>, %scales: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%mat, %scales : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %s: f32, %acc: f32):
        %prod = arith.mulf %a, %s : f32
        linalg.yield %prod : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("scale_matrix_cols", R"mlir(
module {
  func.func @scale_matrix_cols(%mat: tensor<?x?xf32>, %scales: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%mat, %scales : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %s: f32, %acc: f32):
        %prod = arith.mulf %a, %s : f32
        linalg.yield %prod : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("descale_matrix", R"mlir(
module {
  func.func @descale_matrix(%mat: tensor<?x?xf32>, %row_scales: tensor<?xf32>, %col_scales: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}
      ins(%mat, %row_scales, %col_scales : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %rs: f32, %cs: f32, %acc: f32):
        %tmp = arith.divf %a, %rs : f32
        %inv = arith.divf %tmp, %cs : f32
        linalg.yield %inv : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("rowvec_mul_mat", R"mlir(
module {
  func.func @rowvec_mul_mat(%rv: tensor<?xf32>, %mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %k = tensor.dim %rv, %c0 : tensor<?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %rv2d = tensor.expand_shape %rv [[0, 1]] output_shape [%k] : tensor<?xf32> into tensor<1x?xf32>
    %empty2d = tensor.empty(%n) : tensor<1x?xf32>
    %zero = arith.constant 0.0 : f32
    %init2d = linalg.fill ins(%zero : f32) outs(%empty2d : tensor<1x?xf32>) -> tensor<1x?xf32>
    %res2d = linalg.matmul ins(%rv2d, %mat : tensor<1x?xf32>, tensor<?x?xf32>)
      outs(%init2d : tensor<1x?xf32>) -> tensor<1x?xf32>
    %res = tensor.collapse_shape %res2d [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("mat_mul_colvec", R"mlir(
module {
  func.func @mat_mul_colvec(%mat: tensor<?x?xf32>, %cv: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %k = tensor.dim %cv, %c0 : tensor<?xf32>
    %cv2d = tensor.expand_shape %cv [[0, 1]] output_shape [%k] : tensor<?xf32> into tensor<?x1xf32>
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
    (void)ensureFunctionWithBody("vector_add", R"mlir(
module {
  func.func @vector_add(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %lhs, %c0 : tensor<?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]
    } ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("matrix_add", R"mlir(
module {
  func.func @matrix_add(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %n = tensor.dim %lhs, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("matrix_sub", R"mlir(
module {
  func.func @matrix_sub(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %n = tensor.dim %lhs, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %diff = arith.subf %a, %b : f32
        linalg.yield %diff : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("zero_matrix_like", R"mlir(
module {
  func.func @zero_matrix_like(%mat: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %res = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("matrix_add", R"mlir(
module {
  func.func @matrix_add(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %n = tensor.dim %lhs, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("zero_matrix_like", R"mlir(
module {
  func.func @zero_matrix_like(%mat: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %filled : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("scale_matrix_rows", R"mlir(
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
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]
    } ins(%mat, %scales : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %s: f32, %acc: f32):
        %prod = arith.mulf %a, %s : f32
        linalg.yield %prod : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("scale_matrix_cols", R"mlir(
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
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]
    } ins(%mat, %scales : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %s: f32, %acc: f32):
        %prod = arith.mulf %a, %s : f32
        linalg.yield %prod : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("matrix_add", R"mlir(
module {
  func.func @matrix_add(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %n = tensor.dim %lhs, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0, d1)->(d0, d1)>, affine_map<(d0, d1)->(d0, d1)>, affine_map<(d0, d1)->(d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("zero_matrix_like", R"mlir(
module {
  func.func @zero_matrix_like(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %n = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %res = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("scale_matrix_rows", R"mlir(
module {
  func.func @scale_matrix_rows(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.mulf %in, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("scale_matrix_cols", R"mlir(
module {
  func.func @scale_matrix_cols(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.mulf %in, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("matrix_add", R"mlir(
module {
  func.func @matrix_add(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %n = tensor.dim %lhs, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0, d1)->(d0, d1)>, affine_map<(d0, d1)->(d0, d1)>, affine_map<(d0, d1)->(d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("zero_matrix_like", R"mlir(
module {
  func.func @zero_matrix_like(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %n = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %res = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("scale_matrix_rows", R"mlir(
module {
  func.func @scale_matrix_rows(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.mulf %in, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("scale_matrix_cols", R"mlir(
module {
  func.func @scale_matrix_cols(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.mulf %in, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
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
      if (!symbolExists(name)) {
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
    if (!module.lookupSymbol<func::FuncOp>("sample_row_scales")) {
      maybeInsertDecl("sample_row_scales", FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("sample_col_scales")) {
      maybeInsertDecl("sample_col_scales", FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("scale_matrix_rows")) {
      maybeInsertDecl("scale_matrix_rows", FunctionType::get(ctx, TypeRange{dyn2d, vecDyn}, TypeRange{dyn2d}));
    }
    if (!module.lookupSymbol<func::FuncOp>("scale_matrix_cols")) {
      maybeInsertDecl("scale_matrix_cols", FunctionType::get(ctx, TypeRange{dyn2d, vecDyn}, TypeRange{dyn2d}));
    }
    if (!module.lookupSymbol<func::FuncOp>("descale_matrix")) {
      maybeInsertDecl("descale_matrix", FunctionType::get(ctx, TypeRange{dyn2d, vecDyn, vecDyn}, TypeRange{dyn2d}));
    }
    if (!module.lookupSymbol<func::FuncOp>("rowvec_mul_mat")) {
      maybeInsertDecl("rowvec_mul_mat", FunctionType::get(ctx, TypeRange{vecDyn, dyn2d}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("mat_mul_colvec")) {
      maybeInsertDecl("mat_mul_colvec", FunctionType::get(ctx, TypeRange{dyn2d, vecDyn}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_add")) {
      maybeInsertDecl("vector_add", FunctionType::get(ctx, TypeRange{vecDyn, vecDyn}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("matrix_add")) {
      maybeInsertDecl("matrix_add", FunctionType::get(ctx, TypeRange{dyn2d, dyn2d}, TypeRange{dyn2d}));
    }
    if (!module.lookupSymbol<func::FuncOp>("matrix_sub")) {
      maybeInsertDecl("matrix_sub", FunctionType::get(ctx, TypeRange{dyn2d, dyn2d}, TypeRange{dyn2d}));
    }
    if (!module.lookupSymbol<func::FuncOp>("zero_matrix_like")) {
      maybeInsertDecl("zero_matrix_like", FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{dyn2d}));
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
    auto sampleRowFn = module.lookupSymbol<func::FuncOp>(StringRef("sample_row_scales"));
    auto sampleColFn = module.lookupSymbol<func::FuncOp>(StringRef("sample_col_scales"));
    auto scaleRowsFn = module.lookupSymbol<func::FuncOp>(StringRef("scale_matrix_rows"));
    auto scaleColsFn = module.lookupSymbol<func::FuncOp>(StringRef("scale_matrix_cols"));
    auto descaleFn = module.lookupSymbol<func::FuncOp>(StringRef("descale_matrix"));
    auto vectorAddFn = module.lookupSymbol<func::FuncOp>(StringRef("vector_add"));
    auto matrixAddFn = module.lookupSymbol<func::FuncOp>(StringRef("matrix_add"));
    auto matrixSubFn = module.lookupSymbol<func::FuncOp>(StringRef("matrix_sub"));
    auto zeroMatrixLikeFn = module.lookupSymbol<func::FuncOp>(StringRef("zero_matrix_like"));

    auto isHelperFunction = [&](func::FuncOp func) {
      if (!func)
        return false;
      auto name = func.getSymName();
      return name == "column_checksum" || name == "row_checksum" ||
             name == "vector_max_abs_diff" ||
             name == "vector_max_abs_diff_pair" ||
             name == "abft_analysis.abft_log_rowcol_delta" ||
             name == "abft_analysis.abft_log_rowcol_debug" ||
             name == "sample_row_scales" || name == "sample_col_scales" ||
             name == "scale_matrix_rows" || name == "scale_matrix_cols" ||
             name == "descale_matrix" || name == "rowvec_mul_mat" ||
             name == "mat_mul_colvec" || name == "vector_add" ||
             name == "matrix_add" || name == "matrix_sub" ||
             name == "zero_matrix_like";
    };

    SmallVector<linalg::MatmulOp> targets;
    module.walk([&](linalg::MatmulOp matmul) {
      if (isHelperFunction(matmul->getParentOfType<func::FuncOp>()))
        return;
      targets.push_back(matmul);
    });
    if (targets.empty() && logRowColDeltaFn) {
      SmallVector<Operation *, 8> quantTargets;
      module.walk([&](Operation *op) {
        auto parentFunc = op->getParentOfType<func::FuncOp>();
        if (isHelperFunction(parentFunc))
          return;
        StringRef name = op->getName().getStringRef();
        if (name == "linalg.quantized_matmul" ||
            name == "linalg.conv_2d_nhwc_hwcf_q" ||
            name == "linalg.depthwise_conv_2d_nhwc_hwcm_q") {
          quantTargets.push_back(op);
        }
      });
      for (auto [index, op] : llvm::enumerate(quantTargets)) {
        if (op->getNumResults() == 0)
          continue;
        auto outTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!outTy || !outTy.getElementType().isInteger(32))
          continue;
        OpBuilder builder(op);
        Location loc = op->getLoc();
        Operation *dup = builder.clone(*op);
        if (!dup || dup->getNumResults() == 0)
          continue;

        Value refOut = op->getResult(0);
        Value dupOut = dup->getResult(0);
        Block::iterator nextIt = std::next(Block::iterator(op));
        OpBuilder bAfter(op->getBlock(), nextIt);

        SmallVector<Value> idx;
        idx.reserve(outTy.getRank());
        for (int64_t i = 0; i < outTy.getRank(); ++i) {
          idx.push_back(bAfter.create<arith::ConstantIndexOp>(loc, 0));
        }

        Value ref0 = bAfter.create<tensor::ExtractOp>(loc, refOut, idx);
        Value dup0 = bAfter.create<tensor::ExtractOp>(loc, dupOut, idx);
        if (abyzftInjectFault.getValue()) {
          StringRef pattern = abyzftInjectFaultPattern.getValue();
          int base = abyzftInjectFaultDelta.getValue();
          Value delta = bAfter.create<arith::ConstantIntOp>(loc, base, 32);
          // Use data-dependent magnitude instead of layer-index amplification.
          auto one = bAfter.create<arith::ConstantIntOp>(loc, 1, 32);
          auto zero = bAfter.create<arith::ConstantIntOp>(loc, 0, 32);
          auto isNeg = bAfter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, dup0, zero);
          auto negDup = bAfter.create<arith::SubIOp>(loc, zero, dup0);
          auto absDup =
              bAfter.create<arith::SelectOp>(loc, isNeg, negDup, dup0);
          auto mag = bAfter.create<arith::MaxSIOp>(loc, absDup, one);
          delta = bAfter.create<arith::MulIOp>(loc, delta, mag);
          if (pattern == "checkered") {
            delta = bAfter.create<arith::SubIOp>(loc, zero, delta);
          }
          dup0 = bAfter.create<arith::AddIOp>(loc, dup0, delta);
        }
        Value refF = bAfter.create<arith::SIToFPOp>(loc, bAfter.getF32Type(), ref0);
        Value dupF = bAfter.create<arith::SIToFPOp>(loc, bAfter.getF32Type(), dup0);
        Value diff = bAfter.create<arith::SubFOp>(loc, refF, dupF);
        Value delta = bAfter.create<math::AbsFOp>(loc, diff);
        Value layerConst = bAfter.create<arith::ConstantOp>(
            loc, bAfter.getF32Type(), bAfter.getF32FloatAttr((float)index));
        bAfter.create<func::CallOp>(
            loc, StringRef("abft_analysis.abft_log_rowcol_delta"), TypeRange{},
            ValueRange{layerConst, delta, delta});
      }
      return;
    }

    for (auto [matmulIndex, matmul] : llvm::enumerate(targets)) {
      OpBuilder builder(matmul);
      Location loc = matmul.getLoc();

      Value lhs = matmul.getDpsInputOperand(0)->get();
      Value rhs = matmul.getDpsInputOperand(1)->get();
      Value outInit = matmul.getDpsInitOperand(0)->get();
      auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
      auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());
      auto outType = llvm::dyn_cast<RankedTensorType>(outInit.getType());
      if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
          rhsType.getRank() != 2 || outType.getRank() != 2) {
        matmul.emitError("Expected rank-2 tensor operands/results for AByzFT");
        signalPassFailure();
        return;
      }

      auto dyn2dTy = RankedTensorType::get(
          {ShapedType::kDynamic, ShapedType::kDynamic},
          lhsType.getElementType());

      if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
          !outType.hasStaticShape()) {
        if (!sampleRowFn || !sampleColFn || !scaleRowsFn || !scaleColsFn ||
            !descaleFn || !rowFn || !colFn ||
            !vectorAddFn || !matrixAddFn || !zeroMatrixLikeFn || !vecMaxFn) {
          matmul.emitError("Missing dynamic AByzFT helper functions");
          signalPassFailure();
          return;
        }

        Value lhsDyn = lhsType != dyn2dTy
                           ? builder.create<tensor::CastOp>(loc, dyn2dTy, lhs).getResult()
                           : lhs;
        Value rhsDyn = rhsType != dyn2dTy
                           ? builder.create<tensor::CastOp>(loc, dyn2dTy, rhs).getResult()
                           : rhs;
        Value outDyn = outType != dyn2dTy
                           ? builder.create<tensor::CastOp>(loc, dyn2dTy, outInit).getResult()
                           : outInit;

        Value rowScales = builder
                              .create<func::CallOp>(
                                  loc, StringRef("sample_row_scales"),
                                  TypeRange{sampleRowFn.getFunctionType().getResult(0)},
                                  ValueRange{lhsDyn})
                              .getResult(0);
        Value colScales = builder
                              .create<func::CallOp>(
                                  loc, StringRef("sample_col_scales"),
                                  TypeRange{sampleColFn.getFunctionType().getResult(0)},
                                  ValueRange{rhsDyn})
                              .getResult(0);

        Value lhsScaledDyn = builder
                                 .create<func::CallOp>(
                                     loc, StringRef("scale_matrix_rows"),
                                     TypeRange{scaleRowsFn.getFunctionType().getResult(0)},
                                     ValueRange{lhsDyn, rowScales})
                                 .getResult(0);
        Value rhsScaledDyn = builder
                                 .create<func::CallOp>(
                                     loc, StringRef("scale_matrix_cols"),
                                     TypeRange{scaleColsFn.getFunctionType().getResult(0)},
                                     ValueRange{rhsDyn, colScales})
                                 .getResult(0);
        Value zeroInitDyn = builder
                                .create<func::CallOp>(
                                    loc, StringRef("zero_matrix_like"),
                                    TypeRange{zeroMatrixLikeFn.getFunctionType().getResult(0)},
                                    ValueRange{outDyn})
                                .getResult(0);

        Value lhsScaled = lhsScaledDyn.getType() != lhsType
                              ? builder.create<tensor::CastOp>(loc, lhsType, lhsScaledDyn).getResult()
                              : lhsScaledDyn;
        Value rhsScaled = rhsScaledDyn.getType() != rhsType
                              ? builder.create<tensor::CastOp>(loc, rhsType, rhsScaledDyn).getResult()
                              : rhsScaledDyn;
        Value zeroInit = zeroInitDyn.getType() != outType
                             ? builder.create<tensor::CastOp>(loc, outType, zeroInitDyn).getResult()
                             : zeroInitDyn;

        matmul->setOperand(0, lhsScaled);
        matmul->setOperand(1, rhsScaled);
        matmul->setOperand(2, zeroInit);

        builder.setInsertionPointAfter(matmul);
        Value scaledResultDyn = outType != dyn2dTy
                                    ? builder.create<tensor::CastOp>(loc, dyn2dTy, matmul.getResult(0)).getResult()
                                    : matmul.getResult(0);
        Value descaledDyn = builder
                                .create<func::CallOp>(
                                    loc, StringRef("descale_matrix"),
                                    TypeRange{descaleFn.getFunctionType().getResult(0)},
                                    ValueRange{scaledResultDyn, rowScales, colScales})
                                .getResult(0);
        Value restoredDyn = builder
                                .create<func::CallOp>(
                                    loc, StringRef("matrix_add"),
                                    TypeRange{matrixAddFn.getFunctionType().getResult(0)},
                                    ValueRange{descaledDyn, outDyn})
                                .getResult(0);
        Value matmulResult = restoredDyn.getType() != outType
                                 ? builder.create<tensor::CastOp>(loc, outType, restoredDyn).getResult()
                                 : restoredDyn;

        Value lhsColChecksum = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("column_checksum"),
                                       TypeRange{colFn.getFunctionType().getResult(0)},
                                       ValueRange{lhsDyn})
                                   .getResult(0);
        Value rhsRowChecksum = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("row_checksum"),
                                       TypeRange{rowFn.getFunctionType().getResult(0)},
                                       ValueRange{rhsDyn})
                                   .getResult(0);

        Value expectedOutColChecksum =
            buildDynamicRowvecMulMat(builder, loc, lhsColChecksum, rhsDyn);
        Value expectedOutRowChecksum =
            buildDynamicMatMulColvec(builder, loc, lhsDyn, rhsRowChecksum);

        Value resultDyn = outType != dyn2dTy
                              ? builder.create<tensor::CastOp>(loc, dyn2dTy, matmulResult).getResult()
                              : matmulResult;
        Value compareMatrixDyn = resultDyn;
        if (matrixSubFn) {
          compareMatrixDyn = builder
                                 .create<func::CallOp>(
                                     loc, StringRef("matrix_sub"),
                                     TypeRange{matrixSubFn.getFunctionType().getResult(0)},
                                     ValueRange{resultDyn, outDyn})
                                 .getResult(0);
        }
        Value outRowChecksum = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("row_checksum"),
                                       TypeRange{rowFn.getFunctionType().getResult(0)},
                                       ValueRange{compareMatrixDyn})
                                   .getResult(0);
        Value outColChecksum = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("column_checksum"),
                                       TypeRange{colFn.getFunctionType().getResult(0)},
                                       ValueRange{compareMatrixDyn})
                                   .getResult(0);

        Value rowMaxDelta = builder
                                .create<func::CallOp>(
                                    loc, StringRef("vector_max_abs_diff"),
                                    TypeRange{vecMaxFn.getFunctionType().getResult(0)},
                                    ValueRange{expectedOutRowChecksum, outRowChecksum})
                                .getResult(0);
        Value colMaxDelta = builder
                                .create<func::CallOp>(
                                    loc, StringRef("vector_max_abs_diff"),
                                    TypeRange{vecMaxFn.getFunctionType().getResult(0)},
                                    ValueRange{expectedOutColChecksum, outColChecksum})
                                .getResult(0);

        Value rowExpMax = rowMaxDelta;
        Value rowCalcMax = rowMaxDelta;
        Value colExpMax = rowMaxDelta;
        Value colCalcMax = rowMaxDelta;
        if (vecMaxPairFn && !vecMaxPairFn.isExternal()) {
          auto rowPair = builder.create<func::CallOp>(
              loc, StringRef("vector_max_abs_diff_pair"),
              TypeRange{vecMaxPairFn.getFunctionType().getResult(0),
                        vecMaxPairFn.getFunctionType().getResult(1)},
              ValueRange{expectedOutRowChecksum, outRowChecksum});
          rowExpMax = rowPair.getResult(0);
          rowCalcMax = rowPair.getResult(1);
          auto colPair = builder.create<func::CallOp>(
              loc, StringRef("vector_max_abs_diff_pair"),
              TypeRange{vecMaxPairFn.getFunctionType().getResult(0),
                        vecMaxPairFn.getFunctionType().getResult(1)},
              ValueRange{expectedOutColChecksum, outColChecksum});
          colExpMax = colPair.getResult(0);
          colCalcMax = colPair.getResult(1);
        }

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
        continue;
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

      Value lhsDyn = lhsType != dyn2d
                         ? builder.create<tensor::CastOp>(loc, dyn2d, lhs).getResult()
                         : lhs;
      Value rhsDyn = rhsType != dyn2d
                         ? builder.create<tensor::CastOp>(loc, dyn2d, rhs).getResult()
                         : rhs;
      Value outDyn = outType != dyn2d
                         ? builder.create<tensor::CastOp>(loc, dyn2d, outInit).getResult()
                         : outInit;

      Value lhsColChecksum = builder
                                 .create<func::CallOp>(
                                     loc, StringRef("column_checksum"),
                                     TypeRange{colFn.getFunctionType().getResult(0)},
                                     ValueRange{lhsDyn})
                                 .getResult(0);
      Value rhsRowChecksum = builder
                                 .create<func::CallOp>(
                                     loc, StringRef("row_checksum"),
                                     TypeRange{rowFn.getFunctionType().getResult(0)},
                                     ValueRange{rhsDyn})
                                 .getResult(0);

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

      Value compareMatrix = buildElementwiseSub(builder, loc, matmulResult,
                                                outInit, outType);
      Value compareMatrixDyn = outType != dyn2d
                                   ? builder.create<tensor::CastOp>(loc, dyn2d, compareMatrix).getResult()
                                   : compareMatrix;
      Value outRowChecksum = builder
                                 .create<func::CallOp>(
                                     loc, StringRef("row_checksum"),
                                     TypeRange{rowFn.getFunctionType().getResult(0)},
                                     ValueRange{compareMatrixDyn})
                                 .getResult(0);
      Value outColChecksum = builder
                                 .create<func::CallOp>(
                                     loc, StringRef("column_checksum"),
                                     TypeRange{colFn.getFunctionType().getResult(0)},
                                     ValueRange{compareMatrixDyn})
                                 .getResult(0);
      Value expectedOutRowChecksum =
          buildDynamicMatMulColvec(builder, loc, lhsDyn, rhsRowChecksum);
      Value expectedOutColChecksum =
          buildDynamicRowvecMulMat(builder, loc, lhsColChecksum, rhsDyn);

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

      if (vecMaxFn && !vecMaxFn.isExternal()) {
        auto maxTy = vecMaxFn.getFunctionType();
        SmallVector<Type, 1> maxRes;
        for (Type t : maxTy.getResults())
          maxRes.push_back(t);
        Value rowExpectedVecArg = expectedOutRowChecksum;
        if (rowExpectedVecArg.getType() != maxTy.getInput(0))
          rowExpectedVecArg = builder.create<tensor::CastOp>(
              loc, maxTy.getInput(0), rowExpectedVecArg).getResult();
        Value rowCalcVecArg = outRowChecksum;
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

        Value colExpectedVecArg = expectedOutColChecksum;
        if (colExpectedVecArg.getType() != maxTy.getInput(0))
          colExpectedVecArg = builder.create<tensor::CastOp>(
              loc, maxTy.getInput(0), colExpectedVecArg).getResult();
        Value colCalcVecArg = outColChecksum;
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

          Value pairRowExpected = expectedOutRowChecksum;
          if (pairRowExpected.getType() != pairTy.getInput(0))
            pairRowExpected = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(0), pairRowExpected).getResult();
          Value pairRowCalc = outRowChecksum;
          if (pairRowCalc.getType() != pairTy.getInput(1))
            pairRowCalc = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(1), pairRowCalc).getResult();
          auto rowPairCall = builder.create<func::CallOp>(
              loc, StringRef("vector_max_abs_diff_pair"), TypeRange(pairRes),
              ValueRange{pairRowExpected, pairRowCalc});
          rowExpMax = rowPairCall.getResult(0);
          rowCalcMax = rowPairCall.getResult(1);

          Value pairColExpected = expectedOutColChecksum;
          if (pairColExpected.getType() != pairTy.getInput(0))
            pairColExpected = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(0), pairColExpected).getResult();
          Value pairColCalc = outColChecksum;
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

    auto symbolExists = [&](StringRef name) {
      for (Operation &op : module.getBody()->getOperations()) {
        if (auto symName =
                op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
          if (symName.getValue() == name)
            return true;
        }
      }
      return false;
    };

    auto eraseConflictingSymbol = [&](StringRef name) {
      for (Operation &op : llvm::make_early_inc_range(module.getBody()->getOperations())) {
        if (auto symName =
                op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
          if (symName.getValue() == name) {
            op.erase();
            return;
          }
        }
      }
    };

    auto ensureFunctionWithBody = [&](StringRef name,
                                      StringRef body) -> func::FuncOp {
      if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
        if (!existing.empty())
          return existing;
        existing.erase();
      } else if (symbolExists(name)) {
        eraseConflictingSymbol(name);
      }
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
    (void)ensureFunctionWithBody("sample_row_scales", R"mlir(
module {
  func.func @sample_row_scales(%mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
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
    (void)ensureFunctionWithBody("sample_col_scales", R"mlir(
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
    (void)ensureFunctionWithBody("rowvec_mul_mat", R"mlir(
module {
  func.func @rowvec_mul_mat(%rv: tensor<?xf32>, %mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %k = tensor.dim %rv, %c0 : tensor<?xf32>
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %rv2d = tensor.expand_shape %rv [[0, 1]] output_shape [%k] : tensor<?xf32> into tensor<1x?xf32>
    %empty2d = tensor.empty(%n) : tensor<1x?xf32>
    %zero = arith.constant 0.0 : f32
    %init2d = linalg.fill ins(%zero : f32) outs(%empty2d : tensor<1x?xf32>) -> tensor<1x?xf32>
    %res2d = linalg.matmul ins(%rv2d, %mat : tensor<1x?xf32>, tensor<?x?xf32>)
      outs(%init2d : tensor<1x?xf32>) -> tensor<1x?xf32>
    %res = tensor.collapse_shape %res2d [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)ensureFunctionWithBody("mat_mul_colvec", R"mlir(
module {
  func.func @mat_mul_colvec(%mat: tensor<?x?xf32>, %cv: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %k = tensor.dim %cv, %c0 : tensor<?xf32>
    %cv2d = tensor.expand_shape %cv [[0, 1]] output_shape [%k] : tensor<?xf32> into tensor<?x1xf32>
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
    (void)ensureFunctionWithBody("vector_add", R"mlir(
module {
  func.func @vector_add(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %lhs, %c0 : tensor<?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]
    } ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
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
      if (!symbolExists(name)) {
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
    if (!module.lookupSymbol<func::FuncOp>("sample_row_scales")) {
      maybeInsertDecl("sample_row_scales",
                      FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("sample_col_scales")) {
      maybeInsertDecl("sample_col_scales",
                      FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("rowvec_mul_mat")) {
      maybeInsertDecl("rowvec_mul_mat",
                      FunctionType::get(ctx, TypeRange{vecDyn, dyn2d}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("mat_mul_colvec")) {
      maybeInsertDecl("mat_mul_colvec",
                      FunctionType::get(ctx, TypeRange{dyn2d, vecDyn}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_add")) {
      maybeInsertDecl("vector_add",
                      FunctionType::get(ctx, TypeRange{vecDyn, vecDyn}, TypeRange{vecDyn}));
    }
    if (!module.lookupSymbol<func::FuncOp>("matrix_add")) {
      maybeInsertDecl("matrix_add",
                      FunctionType::get(ctx, TypeRange{dyn2d, dyn2d}, TypeRange{dyn2d}));
    }
    if (!module.lookupSymbol<func::FuncOp>("zero_matrix_like")) {
      maybeInsertDecl("zero_matrix_like",
                      FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{dyn2d}));
    }
    if (!module.lookupSymbol<func::FuncOp>("scale_matrix_rows")) {
      maybeInsertDecl("scale_matrix_rows",
                      FunctionType::get(ctx, TypeRange{dyn2d, vecDyn}, TypeRange{dyn2d}));
    }
    if (!module.lookupSymbol<func::FuncOp>("scale_matrix_cols")) {
      maybeInsertDecl("scale_matrix_cols",
                      FunctionType::get(ctx, TypeRange{dyn2d, vecDyn}, TypeRange{dyn2d}));
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
    auto sampleRowFn =
        module.lookupSymbol<func::FuncOp>(StringRef("sample_row_scales"));
    auto sampleColFn =
        module.lookupSymbol<func::FuncOp>(StringRef("sample_col_scales"));
    auto rowvecFn =
        module.lookupSymbol<func::FuncOp>(StringRef("rowvec_mul_mat"));
    auto matcolFn =
        module.lookupSymbol<func::FuncOp>(StringRef("mat_mul_colvec"));
    auto matrixAddFn = ensureFunctionWithBody("matrix_add", R"mlir(
module {
  func.func @matrix_add(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %n = tensor.dim %lhs, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0, d1)->(d0, d1)>, affine_map<(d0, d1)->(d0, d1)>, affine_map<(d0, d1)->(d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    auto zeroMatrixLikeFn = ensureFunctionWithBody("zero_matrix_like", R"mlir(
module {
  func.func @zero_matrix_like(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %n = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %zero = arith.constant 0.0 : f32
    %res = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
)mlir");
    auto scaleRowsFn = ensureFunctionWithBody("scale_matrix_rows", R"mlir(
module {
  func.func @scale_matrix_rows(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.mulf %in, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
}
)mlir");
    auto scaleColsFn = ensureFunctionWithBody("scale_matrix_cols", R"mlir(
module {
  func.func @scale_matrix_cols(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.mulf %in, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
}
)mlir");

    auto isHelperFunction = [&](func::FuncOp func) {
      if (!func)
        return false;
      auto name = func.getSymName();
      return name == "column_checksum" || name == "row_checksum" ||
             name == "vector_max_abs_diff" ||
             name == "vector_max_abs_diff_pair" ||
             name == "abft_analysis.abft_log_rowcol_delta" ||
             name == "abft_analysis.abft_log_rowcol_debug" ||
             name == "sample_row_scales" || name == "sample_col_scales" ||
             name == "scale_matrix_rows" || name == "scale_matrix_cols" ||
             name == "descale_matrix" || name == "rowvec_mul_mat" ||
             name == "mat_mul_colvec" || name == "vector_add" ||
             name == "matrix_add" || name == "matrix_sub" ||
             name == "zero_matrix_like";
    };

    SmallVector<linalg::MatmulOp> targets;
    module.walk([&](linalg::MatmulOp matmul) {
      if (isHelperFunction(matmul->getParentOfType<func::FuncOp>()))
        return;
      targets.push_back(matmul);
    });
    if (targets.empty() && logRowColDeltaFn) {
      SmallVector<Operation *, 8> quantTargets;
      module.walk([&](Operation *op) {
        auto parentFunc = op->getParentOfType<func::FuncOp>();
        if (isHelperFunction(parentFunc))
          return;
        StringRef name = op->getName().getStringRef();
        if (name == "linalg.quantized_matmul" ||
            name == "linalg.conv_2d_nhwc_hwcf_q" ||
            name == "linalg.depthwise_conv_2d_nhwc_hwcm_q") {
          quantTargets.push_back(op);
        }
      });
      for (auto [index, op] : llvm::enumerate(quantTargets)) {
        if (op->getNumResults() == 0)
          continue;
        auto outTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!outTy || !outTy.getElementType().isInteger(32))
          continue;
        OpBuilder builder(op);
        Location loc = op->getLoc();
        Operation *dup = builder.clone(*op);
        if (!dup || dup->getNumResults() == 0)
          continue;

        Value refOut = op->getResult(0);
        Value dupOut = dup->getResult(0);
        Block::iterator nextIt = std::next(Block::iterator(op));
        OpBuilder bAfter(op->getBlock(), nextIt);

        SmallVector<Value> idx;
        idx.reserve(outTy.getRank());
        for (int64_t i = 0; i < outTy.getRank(); ++i) {
          idx.push_back(bAfter.create<arith::ConstantIndexOp>(loc, 0));
        }

        Value ref0 = bAfter.create<tensor::ExtractOp>(loc, refOut, idx);
        Value dup0 = bAfter.create<tensor::ExtractOp>(loc, dupOut, idx);
        if (abyzftInjectFault.getValue()) {
          StringRef pattern = abyzftInjectFaultPattern.getValue();
          int base = abyzftInjectFaultDelta.getValue();
          Value delta = bAfter.create<arith::ConstantIntOp>(loc, base, 32);
          auto one = bAfter.create<arith::ConstantIntOp>(loc, 1, 32);
          auto zero = bAfter.create<arith::ConstantIntOp>(loc, 0, 32);
          auto isNeg = bAfter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, dup0, zero);
          auto negDup = bAfter.create<arith::SubIOp>(loc, zero, dup0);
          auto absDup =
              bAfter.create<arith::SelectOp>(loc, isNeg, negDup, dup0);
          auto mag = bAfter.create<arith::MaxSIOp>(loc, absDup, one);
          delta = bAfter.create<arith::MulIOp>(loc, delta, mag);
          if (pattern == "checkered") {
            delta = bAfter.create<arith::SubIOp>(loc, zero, delta);
          }
          dup0 = bAfter.create<arith::AddIOp>(loc, dup0, delta);
        }
        Value refF = bAfter.create<arith::SIToFPOp>(loc, bAfter.getF32Type(), ref0);
        Value dupF = bAfter.create<arith::SIToFPOp>(loc, bAfter.getF32Type(), dup0);
        Value diff = bAfter.create<arith::SubFOp>(loc, refF, dupF);
        Value delta = bAfter.create<math::AbsFOp>(loc, diff);
        Value layerConst = bAfter.create<arith::ConstantOp>(
            loc, bAfter.getF32Type(), bAfter.getF32FloatAttr((float)index));
        bAfter.create<func::CallOp>(
            loc, StringRef("abft_analysis.abft_log_rowcol_delta"), TypeRange{},
            ValueRange{layerConst, delta, delta});
      }
      return;
    }

    for (auto [matmulIndex, matmul] : llvm::enumerate(targets)) {
      OpBuilder builder(matmul);
      Location loc = matmul.getLoc();

      Value lhs = matmul.getDpsInputOperand(0)->get();
      Value rhs = matmul.getDpsInputOperand(1)->get();
      Value outInit = matmul.getDpsInitOperand(0)->get();
      auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
      auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());
      auto outType = llvm::dyn_cast<RankedTensorType>(outInit.getType());
      if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
          rhsType.getRank() != 2 || outType.getRank() != 2) {
        matmul.emitError("Expected rank-2 tensor operands/results for Freivald");
        signalPassFailure();
        return;
      }

      auto dyn2dTy = RankedTensorType::get(
          {ShapedType::kDynamic, ShapedType::kDynamic},
          lhsType.getElementType());

      if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
          !outType.hasStaticShape()) {
        if (!sampleRowFn || !sampleColFn || !rowFn || !colFn || !rowvecFn ||
            !matcolFn || !matrixAddFn || !zeroMatrixLikeFn || !scaleRowsFn ||
            !scaleColsFn || !vecMaxFn) {
          matmul.emitError("Missing dynamic Freivald helper functions");
          signalPassFailure();
          return;
        }

        Value lhsDyn = lhsType != dyn2dTy
                           ? builder.create<tensor::CastOp>(loc, dyn2dTy, lhs).getResult()
                           : lhs;
        Value rhsDyn = rhsType != dyn2dTy
                           ? builder.create<tensor::CastOp>(loc, dyn2dTy, rhs).getResult()
                           : rhs;
        Value outDyn = outType != dyn2dTy
                           ? builder.create<tensor::CastOp>(loc, dyn2dTy, outInit).getResult()
                           : outInit;

        Value rowScales = builder
                              .create<func::CallOp>(
                                  loc, StringRef("sample_row_scales"),
                                  TypeRange{sampleRowFn.getFunctionType().getResult(0)},
                                  ValueRange{lhsDyn})
                              .getResult(0);
        Value colScales = builder
                              .create<func::CallOp>(
                                  loc, StringRef("sample_col_scales"),
                                  TypeRange{sampleColFn.getFunctionType().getResult(0)},
                                  ValueRange{rhsDyn})
                              .getResult(0);
        Value zeroInitDyn = builder
                                .create<func::CallOp>(
                                    loc, StringRef("zero_matrix_like"),
                                    TypeRange{zeroMatrixLikeFn.getFunctionType().getResult(0)},
                                    ValueRange{outDyn})
                                .getResult(0);
        Value zeroInit = zeroInitDyn.getType() != outType
                             ? builder.create<tensor::CastOp>(loc, outType, zeroInitDyn).getResult()
                             : zeroInitDyn;
        matmul->setOperand(2, zeroInit);

        builder.setInsertionPointAfter(matmul);
        Value resultContributionDyn = outType != dyn2dTy
                              ? builder.create<tensor::CastOp>(loc, dyn2dTy, matmul.getResult(0)).getResult()
                              : matmul.getResult(0);

        Value lhsScaledForChecksum = builder
                                         .create<func::CallOp>(
                                             loc, StringRef("scale_matrix_rows"),
                                             TypeRange{scaleRowsFn.getFunctionType().getResult(0)},
                                             ValueRange{lhsDyn, rowScales})
                                         .getResult(0);
        Value rhsScaledForChecksum = builder
                                         .create<func::CallOp>(
                                             loc, StringRef("scale_matrix_cols"),
                                             TypeRange{scaleColsFn.getFunctionType().getResult(0)},
                                             ValueRange{rhsDyn, colScales})
                                         .getResult(0);
        Value scaledLhsColChecksum = builder
                                         .create<func::CallOp>(
                                             loc, StringRef("column_checksum"),
                                             TypeRange{colFn.getFunctionType().getResult(0)},
                                             ValueRange{lhsScaledForChecksum})
                                         .getResult(0);
        Value scaledRhsRowChecksum = builder
                                         .create<func::CallOp>(
                                             loc, StringRef("row_checksum"),
                                             TypeRange{rowFn.getFunctionType().getResult(0)},
                                             ValueRange{rhsScaledForChecksum})
                                         .getResult(0);

        Value resultScaledCols = builder
                                     .create<func::CallOp>(
                                         loc, StringRef("scale_matrix_cols"),
                                         TypeRange{scaleColsFn.getFunctionType().getResult(0)},
                                         ValueRange{resultContributionDyn, colScales})
                                     .getResult(0);
        Value resultScaledRows = builder
                                     .create<func::CallOp>(
                                         loc, StringRef("scale_matrix_rows"),
                                         TypeRange{scaleRowsFn.getFunctionType().getResult(0)},
                                         ValueRange{resultContributionDyn, rowScales})
                                     .getResult(0);
        Value outRowChecksum = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("row_checksum"),
                                       TypeRange{rowFn.getFunctionType().getResult(0)},
                                       ValueRange{resultScaledCols})
                                   .getResult(0);
        Value outColChecksum = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("column_checksum"),
                                       TypeRange{colFn.getFunctionType().getResult(0)},
                                       ValueRange{resultScaledRows})
                                   .getResult(0);

        Value expectedOutRowChecksum = builder
                                           .create<func::CallOp>(
                                               loc, StringRef("mat_mul_colvec"),
                                               TypeRange{matcolFn.getFunctionType().getResult(0)},
                                               ValueRange{lhsDyn, scaledRhsRowChecksum})
                                           .getResult(0);
        Value expectedOutColChecksum = builder
                                           .create<func::CallOp>(
                                               loc, StringRef("rowvec_mul_mat"),
                                               TypeRange{rowvecFn.getFunctionType().getResult(0)},
                                               ValueRange{scaledLhsColChecksum, rhsDyn})
                                           .getResult(0);
        Value restoredDyn = builder
                                .create<func::CallOp>(
                                    loc, StringRef("matrix_add"),
                                    TypeRange{matrixAddFn.getFunctionType().getResult(0)},
                                    ValueRange{resultContributionDyn, outDyn})
                                .getResult(0);
        Value restoredResult = restoredDyn.getType() != outType
                                   ? builder.create<tensor::CastOp>(loc, outType, restoredDyn).getResult()
                                   : restoredDyn;

        Value rowMaxDelta = builder
                                .create<func::CallOp>(
                                    loc, StringRef("vector_max_abs_diff"),
                                    TypeRange{vecMaxFn.getFunctionType().getResult(0)},
                                    ValueRange{expectedOutRowChecksum, outRowChecksum})
                                .getResult(0);
        Value colMaxDelta = builder
                                .create<func::CallOp>(
                                    loc, StringRef("vector_max_abs_diff"),
                                    TypeRange{vecMaxFn.getFunctionType().getResult(0)},
                                    ValueRange{expectedOutColChecksum, outColChecksum})
                                .getResult(0);

        Value rowExpMax = rowMaxDelta;
        Value rowCalcMax = rowMaxDelta;
        Value colExpMax = rowMaxDelta;
        Value colCalcMax = rowMaxDelta;
        if (vecMaxPairFn && !vecMaxPairFn.isExternal()) {
          auto rowPair = builder.create<func::CallOp>(
              loc, StringRef("vector_max_abs_diff_pair"),
              TypeRange{vecMaxPairFn.getFunctionType().getResult(0),
                        vecMaxPairFn.getFunctionType().getResult(1)},
              ValueRange{expectedOutRowChecksum, outRowChecksum});
          rowExpMax = rowPair.getResult(0);
          rowCalcMax = rowPair.getResult(1);
          auto colPair = builder.create<func::CallOp>(
              loc, StringRef("vector_max_abs_diff_pair"),
              TypeRange{vecMaxPairFn.getFunctionType().getResult(0),
                        vecMaxPairFn.getFunctionType().getResult(1)},
              ValueRange{expectedOutColChecksum, outColChecksum});
          colExpMax = colPair.getResult(0);
          colCalcMax = colPair.getResult(1);
        }

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
        matmul.getResult(0).replaceUsesWithIf(restoredResult, [&](OpOperand &use) {
          Operation *user = use.getOwner();
          return user != restoredResult.getDefiningOp() &&
                 dom.properlyDominates(restoredResult.getDefiningOp(), user);
        });
        continue;
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
      auto rowScaleType = RankedTensorType::get({M}, elemType);
      auto colScaleType = RankedTensorType::get({N}, elemType);
      Value rowScale = builder.create<arith::ConstantOp>(
          loc, rowScaleType, buildDenseTensorAttr(rowScaleType, rowScaleValues));
      Value colScale = builder.create<arith::ConstantOp>(
          loc, colScaleType, buildDenseTensorAttr(colScaleType, colScaleValues));
      auto vecDynTy = RankedTensorType::get({ShapedType::kDynamic}, elemType);
      auto dyn2dTyStatic =
          RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                elemType);
      Value lhsDyn = lhs.getType() != dyn2dTyStatic
                         ? builder.create<tensor::CastOp>(loc, dyn2dTyStatic, lhs)
                               .getResult()
                         : lhs;
      Value rhsDyn = rhs.getType() != dyn2dTyStatic
                         ? builder.create<tensor::CastOp>(loc, dyn2dTyStatic, rhs)
                               .getResult()
                         : rhs;
      Value outDyn =
          outInit.getType() != dyn2dTyStatic
              ? builder.create<tensor::CastOp>(loc, dyn2dTyStatic, outInit)
                    .getResult()
              : outInit;
      Value rowScaleDyn = rowScale.getType() != vecDynTy
                              ? builder.create<tensor::CastOp>(loc, vecDynTy, rowScale)
                                    .getResult()
                              : rowScale;
      Value colScaleDyn = colScale.getType() != vecDynTy
                              ? builder.create<tensor::CastOp>(loc, vecDynTy, colScale)
                                    .getResult()
                              : colScale;
      Value zeroInitDyn = builder
                              .create<func::CallOp>(
                                  loc, StringRef("zero_matrix_like"),
                                  TypeRange{zeroMatrixLikeFn.getFunctionType().getResult(0)},
                                  ValueRange{outDyn})
                              .getResult(0);
      Value zeroInit = zeroInitDyn.getType() != outType
                           ? builder.create<tensor::CastOp>(loc, outType, zeroInitDyn).getResult()
                           : zeroInitDyn;
      matmul->setOperand(2, zeroInit);

      builder.setInsertionPointAfter(matmul);
      Value contributionResult = matmul.getResult(0);
      Value resultContributionDyn =
          contributionResult.getType() != dyn2dTyStatic
              ? builder.create<tensor::CastOp>(loc, dyn2dTyStatic,
                                               contributionResult)
                    .getResult()
              : contributionResult;

      Value lhsScaledForChecksum = builder
                                       .create<func::CallOp>(
                                           loc, StringRef("scale_matrix_rows"),
                                           TypeRange{scaleRowsFn.getFunctionType().getResult(0)},
                                           ValueRange{lhsDyn, rowScaleDyn})
                                       .getResult(0);
      Value rhsScaledForChecksum = builder
                                       .create<func::CallOp>(
                                           loc, StringRef("scale_matrix_cols"),
                                           TypeRange{scaleColsFn.getFunctionType().getResult(0)},
                                           ValueRange{rhsDyn, colScaleDyn})
                                       .getResult(0);
      Value scaledLhsColChecksum = builder
                                       .create<func::CallOp>(
                                           loc, StringRef("column_checksum"),
                                           TypeRange{colFn.getFunctionType().getResult(0)},
                                           ValueRange{lhsScaledForChecksum})
                                       .getResult(0);
      Value scaledRhsRowChecksum = builder
                                       .create<func::CallOp>(
                                           loc, StringRef("row_checksum"),
                                           TypeRange{rowFn.getFunctionType().getResult(0)},
                                           ValueRange{rhsScaledForChecksum})
                                       .getResult(0);

      Value resultScaledCols = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("scale_matrix_cols"),
                                       TypeRange{scaleColsFn.getFunctionType().getResult(0)},
                                       ValueRange{resultContributionDyn, colScaleDyn})
                                   .getResult(0);
      Value resultScaledRows = builder
                                   .create<func::CallOp>(
                                       loc, StringRef("scale_matrix_rows"),
                                       TypeRange{scaleRowsFn.getFunctionType().getResult(0)},
                                       ValueRange{resultContributionDyn, rowScaleDyn})
                                   .getResult(0);
      Value outRowChecksum = builder
                                 .create<func::CallOp>(
                                     loc, StringRef("row_checksum"),
                                     TypeRange{rowFn.getFunctionType().getResult(0)},
                                     ValueRange{resultScaledCols})
                                 .getResult(0);
      Value outColChecksum = builder
                                 .create<func::CallOp>(
                                     loc, StringRef("column_checksum"),
                                     TypeRange{colFn.getFunctionType().getResult(0)},
                                     ValueRange{resultScaledRows})
                                 .getResult(0);
      Value expectedOutRowChecksum = builder
                                         .create<func::CallOp>(
                                             loc, StringRef("mat_mul_colvec"),
                                             TypeRange{matcolFn.getFunctionType().getResult(0)},
                                             ValueRange{lhsDyn, scaledRhsRowChecksum})
                                         .getResult(0);
      Value expectedOutColChecksum = builder
                                         .create<func::CallOp>(
                                             loc, StringRef("rowvec_mul_mat"),
                                             TypeRange{rowvecFn.getFunctionType().getResult(0)},
                                             ValueRange{scaledLhsColChecksum, rhsDyn})
                                         .getResult(0);
      Value restoredDyn = builder
                              .create<func::CallOp>(
                                  loc, StringRef("matrix_add"),
                                  TypeRange{matrixAddFn.getFunctionType().getResult(0)},
                                  ValueRange{resultContributionDyn, outDyn})
                              .getResult(0);
      Value matmulResult = restoredDyn.getType() != outType
                               ? builder.create<tensor::CastOp>(loc, outType, restoredDyn).getResult()
                               : restoredDyn;

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

      if (vecMaxFn && !vecMaxFn.isExternal()) {
        auto maxTy = vecMaxFn.getFunctionType();
        SmallVector<Type, 1> maxRes;
        for (Type t : maxTy.getResults())
          maxRes.push_back(t);
        Value rowExpectedVecArg = expectedOutRowChecksum;
        if (rowExpectedVecArg.getType() != maxTy.getInput(0))
          rowExpectedVecArg = builder.create<tensor::CastOp>(
              loc, maxTy.getInput(0), rowExpectedVecArg).getResult();
        Value rowCalcVecArg = outRowChecksum;
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

        Value colExpectedVecArg = expectedOutColChecksum;
        if (colExpectedVecArg.getType() != maxTy.getInput(0))
          colExpectedVecArg = builder.create<tensor::CastOp>(
              loc, maxTy.getInput(0), colExpectedVecArg).getResult();
        Value colCalcVecArg = outColChecksum;
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

          Value pairRowExpected = expectedOutRowChecksum;
          if (pairRowExpected.getType() != pairTy.getInput(0))
            pairRowExpected = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(0), pairRowExpected).getResult();
          Value pairRowCalc = outRowChecksum;
          if (pairRowCalc.getType() != pairTy.getInput(1))
            pairRowCalc = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(1), pairRowCalc).getResult();
          auto rowPairCall = builder.create<func::CallOp>(
              loc, StringRef("vector_max_abs_diff_pair"), TypeRange(pairRes),
              ValueRange{pairRowExpected, pairRowCalc});
          rowExpMax = rowPairCall.getResult(0);
          rowCalcMax = rowPairCall.getResult(1);

          Value pairColExpected = expectedOutColChecksum;
          if (pairColExpected.getType() != pairTy.getInput(0))
            pairColExpected = builder.create<tensor::CastOp>(
                loc, pairTy.getInput(0), pairColExpected).getResult();
          Value pairColCalc = outColChecksum;
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
      DominanceInfo dom(module);
      matmul.getResult(0).replaceUsesWithIf(matmulResult, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return user != matmulResult.getDefiningOp() &&
               dom.properlyDominates(matmulResult.getDefiningOp(), user);
      });
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
