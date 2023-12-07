#!/bin/bash

input_file=${1}
batch_size=${2:-"1"}
predictions_file=${3:-"ezpc_prediction.dat"}

work_dir="EzPC/OnnxBridge/workdir"
cd $work_dir

# ------------------CLIENT------------------
case $input_file in
  *.npy)
    input_file_basename=$(basename $input_file)
    preprocessed_file=${input_file_basename%".npy"}.inp
    # Use below command to dump the numpy file as a custom .inp format file
    python ../helper/convert_np_to_float_inp.py --inp $input_file --output $preprocessed_file
    ;;
  *)
    preprocessed_file=$input_file
    ;;
esac

# ------------------SERVER------------------
# run server
./mlp_model_LLAMA_15 2 mlp_model_input_weights.dat &

# ------------------CLIENT------------------
#run client
./mlp_model_LLAMA_15 3 127.0.0.1 <  $preprocessed_file |& tee >(tail -n $batch_size > $predictions_file); sync

#./mlp_model_LLAMA_15 3 127.0.0.1 <  $preprocessed_file | tee ezpc_client.log
# tail -n 10 ezpc_client.log > "ezpc_prediction.dat"

#tail 10 | tee ezpc_prediction.dat

#./script.sh |& tee >(tail -10 >file.txt)
##tee ezpc_client.log
#tail -n 10 ezpc_client.log > ezpc_prediction.dat



