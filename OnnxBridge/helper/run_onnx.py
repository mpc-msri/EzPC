import numpy as np
import onnxruntime as rt
import sys, os

if __name__ == "__main__":
    input_np_arr = np.load(sys.argv[1], allow_pickle=True)
    sess = rt.InferenceSession("../../Secfloat/demo/model.onnx")
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: input_np_arr})[0]
    print("Output:\n", pred_onx.flatten())
    output_dir = "onnx_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_name = os.path.join(output_dir, os.path.basename(sys.argv[1]))
    np.save(output_name, pred_onx.flatten())