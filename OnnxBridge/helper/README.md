# OnnxBridge: Helper scripts

These are some scripts to increase productivity while using, developing and testing OnnxBridge, they enable you to create an arbitrary onnx file, input for it and other major tasks.

- **compare_np_arrs.py :** A script if two numpy array are similar and if they match upto which decimal place.
```bash
# usage
python .../helper/compare_np_arrs.py -i $expected.npy $output.npy

# further add a `-v` flag to print the two array
```

- **convert_np_to_float_inp.py :** A script to dump a numpy array to `.inp` file as floating point values.
```bash
# usage
python .../helper/convert_np_to_float_inp.py --inp $actualFileName.npy --output ${actualFileName}_input.inp
```

- **create_np_array.py :** A script to create a numpy array of desired size, also prints the values which can then be saved in a `.inp` file for ezpc inference.
```bash
# usage
python .../helper/create_np_array.py > conv2d_input.inp
# saves conv2d_input.npy and conv2d_input.inp files
```

- **make_model.py :** A script to create custom ONNX files.
```bash
# examples given in the file
```

- **make_np_arr.py :** A script to extract output from th eoutput logs of inference from ezpc.
```bash
# usage
python .../helper/make_np_arr.py "output.txt"
```

- **pre_process.py :** A script to preprocess and save an input image file as numpy array.
```bash
# usage
python .../helper/pre_process.py $image_path
```

- **run_onnx.py :** A script to run ONNX files with a given numpy array.
```bash
# usage
python .../helper/run_onnx.py model.onnx "input.npy"
```
