import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.tools.graph_transforms import TransformGraph

def delete_nodes(graph, ops):
  gd = graph.as_graph_def()
  nodes_to_delete = set(op.name for op in ops)
  new_gd = tf.compat.v1.GraphDef()
  nodes_to_keep = []
  for n in gd.node:
    if not n.name in nodes_to_delete:
      nodes_to_keep.append(n)
  new_gd.node.extend(nodes_to_keep)
  new_graph = tf.Graph()
  with new_graph.as_default():
    tf.import_graph_def(new_gd, name="")
  return new_graph

def remove_dead_nodes(graph, input_tensors, output_tensors):
  transforms = ['remove_nodes(op=Identity)', 'strip_unused_nodes']
  in_list = [i.name for i in input_tensors]
  out_list = [i.name for i in output_tensors]
  optimized_graph_def = TransformGraph(graph.as_graph_def(), in_list, out_list, transforms)
  with tf.Graph().as_default() as opt_graph:
    tf.import_graph_def(optimized_graph_def, name="")
    return opt_graph

def convert_consts_to_var(graph, const_names_list):
  const_var_names_pairs = []
  ops_to_delete = []
  with graph.as_default():
    var_list = []
    for name in const_names_list:
      #tensor = graph.get_tensor_by_name('{}:0'.format(name))
      tensor = graph.get_operation_by_name(name).outputs[0]
      with tf.Session() as sess:
        t_value = sess.run(tensor)
      t_name = '{}_const_var'.format(name)
      var = tf.Variable(t_value, name=t_name)
      const_var_names_pairs.append((name, t_name))
      var_list.append(var)
    
    for const_name, var_name in const_var_names_pairs:
      const_op = graph.get_operation_by_name(const_name)
      var_op = graph.get_operation_by_name('{}/read'.format(var_name))
      ge.swap_outputs(ge.sgv(const_op), ge.sgv(var_op))
      ops_to_delete.append(const_op)
    tf.compat.v1.variables_initializer(var_list, 'init_constvars')
  return delete_nodes(graph, ops_to_delete)

def get_inputs(op):
  return set(input.op for input in op.inputs)

def replace_node_with_const(node):
  print("Trying to execute node {}".format(node.name))
  graph = node.graph
  with graph.as_default():
    const_lists = []
    with tf.Session() as sess:
      for out_t in node.outputs:
        const_val = sess.run(out_t)
        const_op = tf.constant(const_val).op
        const_lists.append(const_op)
      ge.swap_outputs(ge.sgv(node), ge.sgv(const_lists))

def DFS(node, visited, const_map, deleted_nodes):
  print("Visiting node {}".format(node.name))
  visited.add(node)
  if node.type == "Const":
    const_map[node.name] = True
    return True
  if len(node.inputs) == 0:
    const_map[node.name] = False
    return False
  for inp_node in get_inputs(node):
    if not inp_node in visited:
      isConst = DFS(inp_node, visited, const_map, deleted_nodes) 
      const_map[inp_node.name] = isConst
  all_inputs_const = True
  for inp_node in get_inputs(node):
    all_inputs_const = all_inputs_const and const_map[inp_node.name] 
  if all_inputs_const:
    const_map[node.name] = True
    replace_node_with_const(node)
    deleted_nodes.add(node)
    return True
  const_map[node.name] = False
  return False

def get_dangling_consts_old(graph):
  consts = [ i for i in graph.get_operations() if i.type == 'Const' ]
  def has_users(op):
    for i in op.outputs:
      for j in i.consumers():
        if j.type != 'Const':
          return True
    return False
  return [ i for i in consts if not has_users(i)]

def get_dangling_consts(graph, deleted_nodes):
  consts = [ i for i in graph.get_operations() if i.type == 'Const' ]
  def has_users(op):
    for i in op.outputs:
      for j in i.consumers():
        if j.type != 'Const' and (j not in deleted_nodes):
          return True
    return False
  return [ i for i in consts if not has_users(i)] 
 
def fold_constants(graph):
  visited = set({})
  const_map = {}
  deleted_nodes = set({})
  with graph.as_default():
    for node in graph.get_operations():
      if not node in visited:
        isConst = DFS(node, visited, const_map, deleted_nodes)
        if isConst:
          replace_node_with_const(node)
          deleted_nodes.add(node)
  useless_consts = get_dangling_consts(graph, deleted_nodes)
  print("No. of consts to be removed = {}".format(len(useless_consts)))
  deleted_nodes.update(useless_consts)
  graph = delete_nodes(graph, deleted_nodes)
  consts = [ i for i in graph.get_operations() if i.type == 'Const' ]
  print("No. of total consts still remaining = {}".format(len(consts)))
  dang_consts = get_dangling_consts_old(graph)
  print("No. of dang consts still remaining = {}".format(len(dang_consts)))
  return graph

def replace_nodes_with_identity(graph, nop_splits):
  with graph.as_default():
    for split in nop_splits:
      inp_var = split.inputs[1]
      identity = tf.identity(inp_var).op
      ge.swap_outputs(ge.sgv(split), ge.sgv(identity))
  return graph
  
def fold_splits(graph):
  with graph.as_default():
    nop_splits = []
    for node in graph.get_operations():
      if node.type != "Split":
        continue
      if node.get_attr("num_split") == 1:
        nop_splits.append(node)
    replace_nodes_with_identity(graph, nop_splits)
  graph = delete_nodes(graph, set(nop_splits))
  return graph 
