import os, sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TFCompiler'))
import DumpTFMtData

inp = tf.constant(1.,shape=(1,32,32,32,24))
filt = tf.constant(1.,shape=(3,3,3,24,12))

sess = tf.Session()
convOp = tf.nn.conv3d(inp, filt, strides=[1,1,1,1,1], padding="VALID")
outp = sess.run(convOp)
print(outp.shape)
print(outp)

optimized_graph_def = DumpTFMtData.save_graph_metadata(convOp, sess, feed_dict={})
