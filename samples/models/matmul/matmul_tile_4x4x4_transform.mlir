module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.consumed}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [256, 256, 256]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}