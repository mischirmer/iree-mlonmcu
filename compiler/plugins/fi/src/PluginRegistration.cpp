#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

std::unique_ptr<mlir::Pass> createInsertFiCallPass();

namespace mlir::iree_compiler::plugins::fi {

struct InsertFiCallSession
    : public mlir::iree_compiler::PluginSession<InsertFiCallSession> {
  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    auto &funcPM = pm.nest<func::FuncOp>();
    funcPM.addPass(createInsertFiCallPass());
  }
};

extern "C" bool iree_register_compiler_plugin_fi(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<InsertFiCallSession>("insert_fi_call");
  return true;
}

}  // namespace mlir::iree_compiler::plugins::fi
