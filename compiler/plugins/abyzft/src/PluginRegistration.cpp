#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"


// Factory from AByzFTPass.cpp
std::unique_ptr<mlir::Pass> createAByzFTPass();

namespace mlir::iree_compiler::plugins::abyzft {

struct AByzFTPassSession
    : public mlir::iree_compiler::PluginSession<AByzFTPassSession> {
  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    pm.addPass(createAByzFTPass());
  }
};

extern "C" bool iree_register_compiler_plugin_abyzft_pass(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<AByzFTPassSession>("abyzft_pass");
  return true;
}

// Provide a weak compatibility wrapper for the older symbol name
// `iree_register_compiler_plugin_abyzft` which may be referenced by the
// generated StaticLinkedPlugins.inc in some build configurations. We define
// it as a weak symbol so that if another (strong) definition exists elsewhere
// (for example the samples plugin), the strong definition wins and this
// fallback is ignored. When used, forward to the actual `abyzft_pass`
// registration function.
extern "C" bool __attribute__((weak)) iree_register_compiler_plugin_abyzft(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  // Forward to the canonical registration function.
  return iree_register_compiler_plugin_abyzft_pass(registrar);
}

}  // namespace mlir::iree_compiler::plugins::abyzft
