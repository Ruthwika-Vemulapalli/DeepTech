import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("models/defect_8class.onnx")

x = np.random.randn(1,3,224,224).astype(np.float32)

out = sess.run(None, {"input": x})

print("ONNX works. Output shape:", out[0].shape)
