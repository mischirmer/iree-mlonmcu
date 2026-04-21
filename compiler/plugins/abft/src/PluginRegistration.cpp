#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// Factory from ABFTPass.cpp
std::unique_ptr<mlir::Pass> createABFTPass();
std::unique_ptr<mlir::Pass> createABFTSpecializeBatchDimPass();

namespace mlir::iree_compiler::plugins::abft {

struct ABFTPassSession
    : public mlir::iree_compiler::PluginSession<ABFTPassSession> {
  // Insert the pass after GlobalOptimization so that earlier high-level
  // canonicalization/generalization has completed and we are past phases that
  // consider certain bufferized ops (e.g. memref.dim) illegal. This lets the
  // pass safely introduce memref-based helper calls without tripping early
  // legality checks.
  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    pm.addPass(createABFTSpecializeBatchDimPass());
    auto &funcPM = pm.nest<func::FuncOp>();
    funcPM.addPass(createABFTPass());
  }
};

extern "C" bool iree_register_compiler_plugin_abft_pass(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<ABFTPassSession>("abft_pass");
  return true;
}

// Provide a weak compatibility wrapper for the older symbol name
// `iree_register_compiler_plugin_abft` which may be referenced by the
// generated StaticLinkedPlugins.inc in some build configurations. We define
// it as a weak symbol so that if another (strong) definition exists elsewhere
// (for example the samples plugin), the strong definition wins and this
// fallback is ignored. When used, forward to the actual `abft_pass`
// registration function.
extern "C" bool __attribute__((weak)) iree_register_compiler_plugin_abft(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  // Forward to the canonical registration function.
  return iree_register_compiler_plugin_abft_pass(registrar);
}

}  // namespace mlir::iree_compiler::plugins::abft
