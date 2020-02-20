import os, sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TFCompiler'))
import DumpTFMtData

inp = tf.constant(1.,shape=(1,4,16,1))
filt = tf.constant(1.,shape=(4,4,256,1))

sess = tf.Session()
convOp = tf.nn.conv2d_transpose(inp, filt, output_shape=[1,8,32,256], strides=[1,2,2,1], padding="SAME")
outp = sess.run(convOp)
print(outp.shape)
print(outp)

optimized_graph_def = DumpTFMtData.save_graph_metadata(convOp, sess, feed_dict={})

for i in range(8):
	for j in range(32):
		print(outp[0][i][j][2], end=' ')
	print('')