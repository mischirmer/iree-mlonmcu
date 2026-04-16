################################################################################
# Freivalds plugin currently aliases the ABFT pass implementation so we can
# benchmark a clean ABFT-equivalent path under a separate plugin ID.
################################################################################

iree_cc_library(
  NAME
    registration
  SRCS
    "src/FreivaldsPass.cpp"
    "src/PluginRegistration.cpp"
  DEPS
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTransforms
    MLIRSupport
    iree::compiler::PluginAPI
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    freivalds
  TARGET
    ::registration
)
