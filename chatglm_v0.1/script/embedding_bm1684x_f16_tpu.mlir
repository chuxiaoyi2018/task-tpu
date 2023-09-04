#loc = loc(unknown)
module attributes {module.FLOPs = 0 : i64, module.asymmetric = false, module.chip = "bm1684x", module.mode = "F16", module.name = "embedding", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = "embedding_tpu_lowered_bm1684x_f16_weight.npz"} {
  func.func @main(%arg0: tensor<64xsi32> loc(unknown)) -> tensor<64x4096xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) : (tensor<64xsi32>) -> tensor<64xsi32> loc(#loc1)
    %2 = "top.Weight"() : () -> tensor<130528x4096xf16> loc(#loc2)
    %3 = "tpu.Gather"(%2, %1, %0) {axis = 0 : i64} : (tensor<130528x4096xf16>, tensor<64xsi32>, none) -> tensor<64x4096xf16> loc(#loc3)
    %4 = "tpu.Cast"(%3) {with_scale = true} : (tensor<64x4096xf16>) -> tensor<64x4096xf32> loc(#loc4)
    return %4 : tensor<64x4096xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("hidden_states")
#loc2 = loc("logits_Gathermodel.transformer.word_embeddings.weight_f16")
#loc3 = loc("logits_Gather")
#loc4 = loc("logits_Gather_f32")

