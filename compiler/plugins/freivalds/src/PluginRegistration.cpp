#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

std::unique_ptr<mlir::Pass> createFreivaldsPass();

namespace mlir::iree_compiler::plugins::freivalds {

struct FreivaldsPassSession
    : public mlir::iree_compiler::PluginSession<FreivaldsPassSession> {
  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    auto &funcPM = pm.nest<func::FuncOp>();
    funcPM.addPass(createFreivaldsPass());
  }
};

extern "C" bool iree_register_compiler_plugin_freivalds_pass(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<FreivaldsPassSession>("freivalds_pass");
  return true;
}

extern "C" bool __attribute__((weak)) iree_register_compiler_plugin_freivalds(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  return iree_register_compiler_plugin_freivalds_pass(registrar);
}

}  // namespace mlir::iree_compiler::plugins::freivalds

