import tensorflow as tf

model_filename = 'chest_xray_covid19_model.h5'
output_filename = 'covid_resnet.pb'

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

tf.keras.backend.set_learning_phase(0)

with tf.keras.utils.CustomObjectScope({'GlorotUniform': tf.keras.initializers.glorot_uniform()}):
    model = tf.keras.models.load_model(model_filename)
    frozen_graph = freeze_session(tf.keras.backend.get_session(),
                              output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, ".", output_filename, as_text=False)
