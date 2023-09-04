#loc = loc(unknown)
#loc1 = loc("hidden_states")
module attributes {module.FLOPs = 0 : i64, module.asymmetric = false, module.chip = "bm1684x", module.coeff_addr = 4294967296 : i64, module.coeff_size = 1069285376 : i64, module.mode = "F16", module.name = "embedding", module.neuron_addr = 5364252672 : i64, module.neuron_size = 1576960 : i64, module.platform = "ONNX", module.state = "TPU_ADDRESSED", module.weight_file = "embedding_tpu_addressed_bm1684x_f16_weight.npz"} {
  func.func @main(%arg0: tensor<64xsi32> loc(unknown)) -> tensor<64x4096xf32, 5364781056 : i64> {
    %0 = "top.Input"(%arg0) : (tensor<64xsi32>) -> tensor<64xsi32, 5364252672 : i64> loc(#loc1)
    %1 = call @subfunc_0(%0) : (tensor<64xsi32, 5364252672 : i64>) -> tensor<64x4096xf32, 5364781056 : i64> loc(#loc)
    return %1 : tensor<64x4096xf32, 5364781056 : i64> loc(#loc)
  } loc(#loc)
  func.func @subfunc_0(%arg0: tensor<64xsi32, 5364252672 : i64> loc("hidden_states")) -> tensor<64x4096xf32, 5364781056 : i64> attributes {id = 0 : i64, mode = #tpu<run_mode TPU_STATIC>, next_index = array<i32: -1>} {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Weight"() : () -> tensor<130528x4096xf16, 4294967296 : i64> loc(#loc2)
    %2 = "tpu.Gather"(%1, %arg0, %0) {axis = 0 : i64} : (tensor<130528x4096xf16, 4294967296 : i64>, tensor<64xsi32, 5364252672 : i64>, none) -> tensor<64x4096xf16, 5364256768 : i64> loc(#loc3)
    %3 = "tpu.Cast"(%2) {with_scale = true} : (tensor<64x4096xf16, 5364256768 : i64>) -> tensor<64x4096xf32, 5364781056 : i64> loc(#loc4)
    return %3 : tensor<64x4096xf32, 5364781056 : i64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("logits_Gathermodel.transformer.word_embeddings.weight_f16")
#loc3 = loc("logits_Gather")
#loc4 = loc("logits_Gather_f32")

