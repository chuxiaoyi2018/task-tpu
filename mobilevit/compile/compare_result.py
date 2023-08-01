import numpy as np
from tpu_perf.infer import SGInfer


def cos_sim(a, b):
  """计算两个向量a和b的余弦相似度"""

  a = np.array(a)
  b = np.array(b)

  inner_product = np.dot(a, b)
  # 内积
  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)
  # 模长
  cos_sim = inner_product / (norm_a * norm_b)

  return cos_sim

# bad for random, great for normal picture
#inputs = np.random.randn(1,3,256,256).astype(np.float32)
inputs = np.load('inputs.npy')

int8_model = SGInfer("tmp/mobilevit_int8.bmodel", devices=[16])
f32_model = SGInfer("tmp/mobilevit_f32.bmodel", devices=[16])

task_id = int8_model.put(inputs)
task_id, int8_results, valid = int8_model.get()

task_id = f32_model.put(inputs)
task_id, f32_results, valid = f32_model.get()

outputs = np.load("outputs.npy")
print("FP32 Cosine Similarity:", cos_sim(outputs.flatten(), f32_results[0].flatten()))
print("INT8 Cosine Similarity:", cos_sim(outputs.flatten(), int8_results[0].flatten()))

print("FP32&INT8 Cosine Similarity:", cos_sim(f32_results[0].flatten(), int8_results[0].flatten()))
