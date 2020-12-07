import tensorflow as tf
from tensorflow.python.platform import gfile

def display_graph(graph, tensorboard_log_dir):
  writer = tf.summary.FileWriter(tensorboard_log_dir, graph)
  writer.close()

def load_pb(path_to_pb):
  with tf.io.gfile.GFile(path_to_pb, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
    return graph

def dump_pb(graph, filename):
  with tf.io.gfile.GFile(filename, 'wb') as f:
    graph_def = graph.as_graph_def()
    f.write(graph_def.SerializeToString())

def save_model(graph, model_name):
  with graph.as_default():
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      save_path = tf.train.Saver().save(sess, model_name)
      print("Model saved in path: %s" % save_path)
