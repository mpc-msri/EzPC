from tf_graph_io import *
from tf_graph_trans import *
import sys
import time
import os 
# Transpose nodes require perm as compile time constants for parametric codegen
# So we don't eliminate the constants we need dring compile time
def get_const_names(graph):
  transp_perm_ops = set(i.inputs[1].op.name for i in graph.get_operations() if i.type == 'Transpose')
  padding_ops = set(i.inputs[1].op.name for i in graph.get_operations() if i.type == 'Pad')
  slice_begin_ops = set(i.inputs[1].op.name for i in graph.get_operations() if i.type == 'Slice')
  slice_size_ops = set(i.inputs[2].op.name for i in graph.get_operations() if i.type == 'Slice')
  mean_axes_ops = set(i.inputs[1].op.name for i in graph.get_operations() if i.type == 'Mean')
  white_list = transp_perm_ops | padding_ops | slice_begin_ops | slice_size_ops | mean_axes_ops
  all_const_ops = set(i.name for i in graph.get_operations() if i.type == 'Const')
  return list(all_const_ops - white_list)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python preprocess_frozen_tf_graph.py tf_model_name.pb")
    sys.exit()
  else:
    input_fname = sys.argv[1]

  actual_fname = os.path.basename(input_fname) 
  dirname = os.path.dirname(input_fname) 
  output_fname = os.path.join(dirname, "mpc_processed_" + actual_fname)
  print("Loading ", input_fname, "for processing.")
  exec_graph = load_pb(input_fname)
  
  print("\n\nThis process will take some time to run as we execute portions of the graph.\n\n")
  time.sleep(5)
  # Fold away all static computations
  print("Running constant folding")
  exec_graph = fold_splits(exec_graph)
  exec_graph = fold_constants(exec_graph)
  # Convert constants to variables so as to separate the data and the generated code
  # Otherwise huge arrays will show up as constants in the generated code, thereby
  # increasing binary size.
  print("Convert frozen constants to variables")
  exec_graph = convert_consts_to_var(exec_graph, get_const_names(exec_graph))
  
  # At this stage the graph still has constants embedded in it
  # in the assign nodes for variables. We cannot execute the graph without
  # these constants. However after inferring the size, we can call remove_dead_nodes
  # to optimize away the constants and assign nodes and make the graph amenable
  # for codegen
  dump_pb(exec_graph, output_fname)
  print("The processed graph is dumped in ", output_fname)
